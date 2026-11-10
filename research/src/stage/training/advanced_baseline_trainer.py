"""Training loops for DIDS-MFL and GraphIDS.

These methods have objectives that do not fit the legacy cross-entropy-only
baseline trainer, so they intentionally use a separate, explicit path.
"""

import gc
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.baselines.dids_mfl.config import Config as DIDSConfig
from stage.baselines.graphids.config import Config as GraphIDSConfig
from stage.baselines.registry import build_baseline, normalize_baseline_name
from stage.data.baseline_protocol import (
    GraphTargetDataset,
    infer_dimensions,
    load_baseline_dataset,
    load_baseline_split,
    native_num_classes,
)
from stage.training.trainer import DEVICE
from stage.utils.config import get_path


ADVANCED_BASELINES = ("DIDS-MFL", "GraphIDS")


def _set_seed(seed):
    """Make each fold repeatable without forcing deterministic slow kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_slug(name):
    return name.lower().replace("+", "_plus_").replace("-", "_")


def advanced_checkpoint_path(output_dir, name, dataset_name, fold_id):
    output_dir = Path(output_dir)
    if dataset_name == "ID-APT":
        filename = f"{name}_fold_{fold_id}.pth"
    else:
        filename = f"{name}_{dataset_slug(dataset_name)}_fold_{fold_id}.pth"
    return output_dir / filename


def _loader(dataset, indices, task, batch_size, shuffle):
    wrapped = GraphTargetDataset(dataset, task=task, indices=indices)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _save_checkpoint(path, network, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": network.state_dict(), **metadata}, path)


def _progress_path(path):
    path = Path(path)
    return path.with_name(f"{path.stem}.progress{path.suffix}")


def _restore_progress(path, model, optimizer, device):
    """Restore an interrupted fold without treating the progress file as best."""
    progress = _progress_path(path)
    source = progress if progress.exists() else Path(path)
    if not source.exists():
        return 0, None
    state = torch.load(source, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    completed = int(state.get("completed_epoch", state.get("epoch", 0)))
    logging.info("Resuming %s after epoch %s", source, completed)
    return completed, state


def _save_progress(path, network, optimizer, completed_epoch, **metadata):
    _save_checkpoint(
        _progress_path(path),
        network,
        optimizer_state_dict=optimizer.state_dict(),
        completed_epoch=int(completed_epoch),
        **metadata,
    )


def _dids_validation(model, loader, device):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            true.extend(batch.graph_y.view(-1).cpu().tolist())
            pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return f1_score(true, pred, average="macro", zero_division=0)


def train_dids_mfl_fold(fold_id, dataset_name, output_dir, batch_size, epochs, device):
    seed = 42 + int(fold_id)
    _set_seed(seed)
    spec, dataset = load_baseline_dataset(dataset_name)
    split = load_baseline_split(spec, dataset, fold_id)
    task = "native" if spec.name == "ID-APT" else "binary"
    in_channels, edge_dim = infer_dimensions(dataset)
    num_classes = native_num_classes(spec, dataset) if task == "native" else 2

    train_loader = _loader(dataset, split["train"], task, batch_size, True)
    val_loader = _loader(dataset, split["val"], task, batch_size, False)
    model = build_baseline(
        "DIDS-MFL",
        num_classes=num_classes,
        in_channels=in_channels,
        edge_dim=edge_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DIDSConfig.learning_rate)
    criterion = nn.CrossEntropyLoss()
    path = advanced_checkpoint_path(output_dir, "DIDS-MFL", spec.name, fold_id)
    start_epoch, restored = _restore_progress(path, model, optimizer, device)
    best_f1 = float(restored.get("best_validation_macro_f1", restored.get("validation_macro_f1", -1.0))) if restored else -1.0

    for epoch in range(start_epoch, int(epochs or DIDSConfig.epochs)):
        model.train()
        progress = tqdm(train_loader, desc=f"DIDS-MFL fold {fold_id} epoch {epoch + 1}")
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, aux = model(batch, return_aux=True)
            loss = criterion(logits, batch.graph_y.view(-1))
            loss = loss + DIDSConfig.disentangle_gamma * aux["orthogonality"]
            loss = loss + DIDSConfig.disentangle_eta * aux["decorrelation"]
            loss = loss + DIDSConfig.disentangle_alpha * aux["smoothness"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_f1 = _dids_validation(model, val_loader, device)
        logging.info("DIDS-MFL fold=%s epoch=%s val_macro_f1=%.4f", fold_id, epoch + 1, val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            _save_checkpoint(
                path,
                model,
                model="DIDS-MFL",
                dataset=spec.name,
                task=task,
                fold=fold_id,
                epoch=epoch + 1,
                validation_macro_f1=val_f1,
                in_channels=in_channels,
                edge_dim=edge_dim,
                num_classes=num_classes,
            )
        _save_progress(
            path, model, optimizer, epoch + 1,
            model="DIDS-MFL", dataset=spec.name, task=task, fold=fold_id,
            best_validation_macro_f1=max(best_f1, val_f1),
            in_channels=in_channels, edge_dim=edge_dim, num_classes=num_classes,
        )
        best_f1 = max(best_f1, val_f1)
    return path


def find_graphids_threshold(scores, labels, steps=500):
    if scores.numel() == 0:
        raise ValueError("Cannot calibrate GraphIDS threshold on an empty validation set.")
    if float(scores.min()) == float(scores.max()):
        return float(scores.mean())
    best_threshold = float(scores.mean())
    best_f1 = -1.0
    for threshold in torch.linspace(scores.min(), scores.max(), steps=steps):
        pred = (scores > threshold).long()
        score = f1_score(labels.numpy(), pred.numpy(), average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


@torch.no_grad()
def _graphids_scores(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        scores.append(model.anomaly_score(batch).cpu())
        labels.append(batch.graph_y.view(-1).cpu())
    return torch.cat(scores), torch.cat(labels)


def train_graphids_fold(fold_id, dataset_name, output_dir, batch_size, epochs, device):
    seed = 42 + int(fold_id)
    _set_seed(seed)
    spec, dataset = load_baseline_dataset(dataset_name)
    split = load_baseline_split(spec, dataset, fold_id)
    in_channels, edge_dim = infer_dimensions(dataset)

    benign_train = [
        index
        for index in split["train"]
        if int(dataset[int(index)].y.view(-1).ne(0).any().item()) == 0
    ]
    if not benign_train:
        raise ValueError(f"GraphIDS fold {fold_id} has no benign training graphs.")

    effective_batch = max(1, min(int(batch_size), 4))
    train_loader = _loader(dataset, benign_train, "binary", effective_batch, True)
    val_loader = _loader(dataset, split["val"], "binary", effective_batch, False)
    model = build_baseline(
        "GraphIDS", num_classes=2, in_channels=in_channels, edge_dim=edge_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "weight_decay": GraphIDSConfig.encoder_weight_decay,
            },
            {
                "params": model.transformer.parameters(),
                "weight_decay": GraphIDSConfig.ae_weight_decay,
            },
        ],
        lr=GraphIDSConfig.learning_rate,
    )
    path = advanced_checkpoint_path(output_dir, "GraphIDS", spec.name, fold_id)
    start_epoch, restored = _restore_progress(path, model, optimizer, device)
    best_pr_auc = float(restored.get("best_validation_pr_auc", restored.get("validation_pr_auc", -float("inf")))) if restored else -float("inf")
    wait = int(restored.get("patience_wait", 0)) if restored else 0

    for epoch in range(start_epoch, int(epochs or GraphIDSConfig.epochs)):
        model.train()
        progress = tqdm(train_loader, desc=f"GraphIDS fold {fold_id} epoch {epoch + 1}")
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.reconstruction_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scores, labels = _graphids_scores(model, val_loader, device)
        threshold = find_graphids_threshold(
            scores, labels, steps=GraphIDSConfig.threshold_steps
        )
        pr_auc = average_precision_score(labels.numpy(), scores.numpy())
        logging.info(
            "GraphIDS fold=%s epoch=%s val_pr_auc=%.4f threshold=%.6f",
            fold_id,
            epoch + 1,
            pr_auc,
            threshold,
        )
        if pr_auc >= best_pr_auc:
            _save_checkpoint(
                path,
                model,
                model="GraphIDS",
                dataset=spec.name,
                task="binary",
                fold=fold_id,
                epoch=epoch + 1,
                validation_pr_auc=float(pr_auc),
                threshold=float(threshold),
                in_channels=in_channels,
                edge_dim=edge_dim,
                num_classes=2,
            )
        if pr_auc > best_pr_auc:
            best_pr_auc = float(pr_auc)
            wait = 0
        else:
            wait += 1
        _save_progress(
            path, model, optimizer, epoch + 1,
            model="GraphIDS", dataset=spec.name, task="binary", fold=fold_id,
            best_validation_pr_auc=max(best_pr_auc, float(pr_auc)),
            patience_wait=wait, threshold=float(threshold),
            in_channels=in_channels, edge_dim=edge_dim, num_classes=2,
        )
        if wait >= GraphIDSConfig.patience:
            break
    return path


def train_advanced_baseline_kfold(
    name,
    dataset_name="ID-APT",
    fold=-1,
    batch_size=16,
    epochs=None,
    output_dir=None,
    device=DEVICE,
):
    canonical = normalize_baseline_name(name)
    if canonical not in ADVANCED_BASELINES:
        raise ValueError(f"Not an advanced baseline: {name}")
    output_dir = Path(output_dir or get_path("outputs", "baseline_checkpoint_dir"))
    folds = range(5) if int(fold) == -1 else [int(fold)]
    paths = []
    for fold_id in folds:
        if canonical == "DIDS-MFL":
            path = train_dids_mfl_fold(
                fold_id, dataset_name, output_dir, batch_size, epochs, device
            )
        else:
            path = train_graphids_fold(
                fold_id, dataset_name, output_dir, batch_size, epochs, device
            )
        paths.append(path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return paths


def print_advanced_training_plan(name, dataset_name, fold, output_dir=None):
    canonical = normalize_baseline_name(name)
    config = DIDSConfig if canonical == "DIDS-MFL" else GraphIDSConfig
    folds = "0, 1, 2, 3, 4" if int(fold) == -1 else str(fold)
    print("Advanced baseline training plan")
    print("model:", canonical)
    print("dataset:", dataset_name)
    print("folds:", folds)
    print("epochs:", config.epochs)
    print("learning_rate:", config.learning_rate)
    print("output_dir:", output_dir or get_path("outputs", "baseline_checkpoint_dir"))
