"""Train shortcut-visible advanced baselines and evaluate their domain shift.

The original ('w/o') graph construction exposes stable identity fields; the
de-identified ('w/') condition uses the normal ID-APT graphs. This script never
changes reference graphs/checkpoints and never tunes on an outer shortcut test
fold. GraphIDS remains binary, so it has no malware-family classification row.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from shortcut_ablation.models.identity_encoder import ShortcutFeatureAdapter
from stage.baselines.dids_mfl.config import Config as DIDSConfig
from stage.baselines.graphids.config import Config as GraphConfig
from stage.baselines.registry import build_baseline
from stage.data.baseline_protocol import GraphTargetDataset, binary_graph_target
from stage.data.dataset import IDAPTDataset
from stage.evaluation.metrics import calculate_metrics_dict
from stage.training.advanced_baseline_trainer import find_graphids_threshold
from stage.utils.config import get_path


ROOT = Path(__file__).resolve().parents[1]
SHORTCUT = ROOT / "outputs" / "shortcut_ablation"
WORKSPACE = ROOT / "outputs" / "workspace"
CKPT_DIR = WORKSPACE / "checkpoints" / "advanced_robustness"
METRIC_DIR = WORKSPACE / "metrics" / "advanced_robustness"
MODELS = ("DIDS-MFL", "GraphIDS")


class IdentityVisibleModel(nn.Module):
    """Use the same trainable identity representation as legacy shortcut models."""

    def __init__(self, name, base_dim, edge_dim, classes, ip_vocab_size):
        super().__init__()
        self.adapter = ShortcutFeatureAdapter(ip_vocab_size, base_dim=base_dim)
        self.base = build_baseline(
            name, num_classes=classes, in_channels=self.adapter.output_dim,
            edge_dim=edge_dim,
        )

    def _adapt(self, data):
        graph = copy.copy(data)
        graph.x = self.adapter(data)
        return graph

    def forward(self, data, return_aux=False):
        if return_aux:
            return self.base(self._adapt(data), return_aux=True)
        return self.base(self._adapt(data))

    def reconstruction_loss(self, data):
        return self.base.reconstruction_loss(self._adapt(data))

    def anomaly_score(self, data):
        return self.base.anomaly_score(self._adapt(data))


def _seed(fold):
    seed = 2026 + int(fold)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, indices, task, batch_size, shuffle=False):
    return DataLoader(
        GraphTargetDataset(dataset, task=task, indices=indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _shortcut_split(fold):
    path = SHORTCUT / "splits" / f"fold_{fold + 1}.json"
    with path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    return split["train_indices"], split["test_indices"]


def _ip_vocab_size():
    with (SHORTCUT / "meta" / "ip_vocab.json").open(encoding="utf-8") as handle:
        return max(json.load(handle).values()) + 1


def _checkpoint(name, fold):
    return CKPT_DIR / f"{name}_shortcut_fold_{fold}.pth"


def _progress_checkpoint(name, fold):
    path = _checkpoint(name, fold)
    return path.with_name(f"{path.stem}.progress{path.suffix}")


@torch.no_grad()
def _predict(model, name, dataset, indices, task, device, threshold=None):
    model.eval()
    batch_size = 4 if name == "GraphIDS" else 16
    true, predicted, scores = [], [], []
    for batch in _loader(dataset, indices, task, batch_size):
        batch = batch.to(device)
        if name == "GraphIDS":
            error = model.anomaly_score(batch).cpu()
            scores.extend(error.tolist())
            if threshold is not None:
                predicted.extend((error > threshold).long().tolist())
        else:
            predicted.extend(model(batch).argmax(dim=-1).cpu().tolist())
        true.extend(batch.graph_y.view(-1).cpu().tolist())
    return np.asarray(true), np.asarray(predicted), np.asarray(scores)


def _train_shortcut_fold(name, fold, dataset, epochs, device):
    _seed(fold)
    outer_train, _ = _shortcut_split(fold)
    task = "binary" if name == "GraphIDS" else "native"
    labels = [binary_graph_target(dataset[i]) if task == "binary" else int(dataset[i].y.item())
              for i in outer_train]
    # The shortcut fold's test indices are NEVER used for checkpoint selection.
    inner_train, inner_val = train_test_split(
        outer_train, test_size=0.1, random_state=2026 + fold, stratify=labels
    )
    base_dim = dataset[0].x.size(-1)
    edge_dim = dataset[0].edge_attr.size(-1)
    classes = 2 if task == "binary" else max(int(dataset[i].y.item()) for i in range(len(dataset))) + 1
    vocab = _ip_vocab_size()
    model = IdentityVisibleModel(name, base_dim, edge_dim, classes, vocab).to(device)
    if name == "GraphIDS":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.adapter.parameters(), "weight_decay": 0.0},
                {"params": model.base.encoder.parameters(), "weight_decay": GraphConfig.encoder_weight_decay},
                {"params": model.base.transformer.parameters(), "weight_decay": GraphConfig.ae_weight_decay},
            ], lr=GraphConfig.learning_rate
        )
        train_idx = [i for i in inner_train if binary_graph_target(dataset[i]) == 0]
        if not train_idx:
            raise ValueError(f"GraphIDS shortcut fold {fold} has no benign training graphs")
        limit = int(epochs or GraphConfig.epochs)
        batch_size = 4
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=DIDSConfig.learning_rate)
        criterion = nn.CrossEntropyLoss()
        train_idx = inner_train
        limit = int(epochs or DIDSConfig.epochs)
        batch_size = 16
    loader = _loader(dataset, train_idx, task, batch_size, shuffle=True)
    path = _checkpoint(name, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = _progress_checkpoint(name, fold)
    resume_path = progress_path if progress_path.exists() else path
    start_epoch, best, stale = 0, -float("inf"), 0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        if resume.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("completed_epoch", resume.get("epoch", 0)))
        best = float(resume.get("best_inner_validation", resume.get("inner_validation", -float("inf"))))
        stale = int(resume.get("patience_stale", 0))
        print(f"Resuming {name} shortcut fold={fold} after epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, limit):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if name == "GraphIDS":
                loss = model.reconstruction_loss(batch)
            else:
                logits, aux = model(batch, return_aux=True)
                loss = criterion(logits, batch.graph_y.view(-1))
                loss = loss + DIDSConfig.disentangle_gamma * aux["orthogonality"]
                loss = loss + DIDSConfig.disentangle_eta * aux["decorrelation"]
                loss = loss + DIDSConfig.disentangle_alpha * aux["smoothness"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        true, pred, scores = _predict(model, name, dataset, inner_val, task, device)
        if name == "GraphIDS":
            threshold = find_graphids_threshold(torch.tensor(scores), torch.tensor(true), GraphConfig.threshold_steps)
            score = float(average_precision_score(true, scores))
        else:
            threshold = None
            score = float(f1_score(true, pred, average="macro", zero_division=0))
        print(f"{name} shortcut fold={fold} epoch={epoch + 1}/{limit} "
              f"loss={np.mean(losses):.5f} inner_validation={score:.5f}", flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({
                "model_state_dict": model.state_dict(), "model": name,
                "condition": "w/o", "fold": fold, "epoch": epoch + 1,
                "task": task, "threshold": threshold, "inner_validation": score,
                "base_dim": base_dim, "edge_dim": edge_dim,
                "classes": classes, "ip_vocab_size": vocab,
            }, path)
        else:
            stale += 1
        torch.save({
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "model": name, "condition": "w/o", "fold": fold,
            "completed_epoch": epoch + 1, "task": task, "threshold": threshold,
            "best_inner_validation": best, "patience_stale": stale,
            "base_dim": base_dim, "edge_dim": edge_dim,
            "classes": classes, "ip_vocab_size": vocab,
        }, progress_path)
        if name == "GraphIDS" and stale >= GraphConfig.patience:
            break
    return path


def _load_model(name, fold, condition, dataset, device):
    if condition == "w/o":
        path = _checkpoint(name, fold)
    else:
        from stage.training.advanced_baseline_trainer import advanced_checkpoint_path
        path = advanced_checkpoint_path(get_path("outputs", "baseline_checkpoint_dir"), name, "ID-APT", fold)
    if not path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    if condition == "w/o":
        model = IdentityVisibleModel(
            name, state["base_dim"], state["edge_dim"], state["classes"], state["ip_vocab_size"]
        )
    else:
        model = build_baseline(name, num_classes=state["num_classes"],
                               in_channels=state["in_channels"], edge_dim=state["edge_dim"])
    model.load_state_dict(state["model_state_dict"])
    return model.to(device), state


def _domain_data(condition, fold, original):
    if condition == "w/o":
        shifted_dir = SHORTCUT / "graphs_shifted"
        indices = _shortcut_split(fold)[1]
    else:
        shifted_dir = Path(get_path("data", "shifted_graph_dir")) / f"fold_{fold}"
        if not shifted_dir.exists() and fold == 1:
            shifted_dir = WORKSPACE / "graphs_shifted_generated" / "fold_1"
        with (WORKSPACE / "splits" / f"fold_{fold}.json").open(encoding="utf-8") as handle:
            indices = json.load(handle)["val"]
    if not shifted_dir.exists():
        raise FileNotFoundError(f"Missing shifted-domain graphs: {shifted_dir}")
    if condition == "w/o":
        shifted = IDAPTDataset(root_dir=shifted_dir)
    else:
        # The preserved shifted-domain artifact uses shifted_chunk_*.pt names,
        # not the IDAPTDataset graphs_chunk_*.pt convention.
        shifted = []
        def chunk_order(path):
            suffix = path.stem.rsplit("_", 1)[-1]
            return int(suffix) if suffix.isdigit() else float("inf")

        for path in sorted(shifted_dir.glob("*.pt"), key=chunk_order):
            shifted.extend(torch.load(path, map_location="cpu", weights_only=False))
        if not shifted:
            raise FileNotFoundError(f"No shifted graph chunks in {shifted_dir}")
    original_labels = [int(original[i].y.view(-1)[0]) for i in indices]
    shifted_labels = [int(shifted[i].y.view(-1)[0]) for i in
                      (indices if condition == "w/o" else range(len(shifted)))]
    if original_labels != shifted_labels:
        raise ValueError(f"Original/shifted label order mismatch: {condition} fold={fold}")
    shifted_indices = indices if condition == "w/o" else range(len(shifted))
    return indices, shifted, shifted_indices


def _evaluate(name, fold, condition, original, device):
    model, state = _load_model(name, fold, condition, original, device)
    indices, shifted, shifted_indices = _domain_data(condition, fold, original)
    task = state["task"]
    threshold = state.get("threshold")
    o_true, o_pred, _ = _predict(model, name, original, indices, task, device, threshold)
    s_true, s_pred, _ = _predict(model, name, shifted, shifted_indices, task, device, threshold)
    if not np.array_equal(o_true, s_true):
        raise ValueError(f"Domain test labels do not align: {condition} fold={fold}")
    rows = []
    original_f1 = calculate_metrics_dict(o_true, o_pred)["F1_Macro"]
    for domain, predicted in (("Original", o_pred), ("Shifted", s_pred)):
        metrics = calculate_metrics_dict(o_true, predicted)
        rows.append({"Model": name, "Condition": condition, "Fold": fold,
                     "Task": task, "Test Domain": domain,
                     "F1-Score": metrics["F1_Macro"], "Recall": metrics["Recall"],
                     "Precision": metrics["Precision"], "Accuracy": metrics["Accuracy"],
                     "Delta F1 (%)": 100 * (metrics["F1_Macro"] - original_f1) / original_f1
                     if domain == "Shifted" and original_f1 else 0.0})
    return rows, (o_true, o_pred, s_pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--condition", choices=("w/o", "w/"), required=True)
    parser.add_argument("--train", action="store_true", help="Train a shortcut-visible w/o checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.train and args.condition != "w/o":
        parser.error("--train only applies to the shortcut-visible w/o condition")
    if not args.train and not args.evaluate:
        parser.error("select --train, --evaluate, or both")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dir = SHORTCUT / "graphs" if args.condition == "w/o" else get_path("data", "graph_dir")
    original = IDAPTDataset(root_dir=graph_dir)
    if args.train:
        print("Saved shortcut checkpoint:", _train_shortcut_fold(args.model, args.fold, original, args.epochs, device))
    if args.evaluate:
        rows, predictions = _evaluate(args.model, args.fold, args.condition, original, device)
        METRIC_DIR.mkdir(parents=True, exist_ok=True)
        out = METRIC_DIR / f"table6_{args.model.lower().replace('-', '_')}_{args.condition.replace('/', '')}_fold_{args.fold}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print("Saved Table 6 fold metrics:", out)
        if args.model == "DIDS-MFL" and args.condition == "w/":
            target, original_pred, shifted_pred = predictions
            with (ROOT / "data" / "reference" / "graphs" / "label_mapping.json").open(encoding="utf-8") as handle:
                label_map = json.load(handle)
            families = []
            classes = sorted(c for c in label_map.values() if c != 0)
            for domain, predicted in (("Original", original_pred), ("Shifted", shifted_pred)):
                precision, recall, f1, support = precision_recall_fscore_support(
                    target, predicted, labels=classes, zero_division=0
                )
                for i, family in enumerate(classes):
                    families.append({"Model": args.model, "Fold": args.fold,
                                     "Malware Family": next(k for k, v in label_map.items() if v == family),
                                     "Test Domain": domain, "F1-Score": f1[i],
                                     "Recall": recall[i], "Precision": precision[i],
                                     "Support": support[i]})
            family_out = METRIC_DIR / f"table7_dids_mfl_fold_{args.fold}.csv"
            pd.DataFrame(families).to_csv(family_out, index=False)
            print("Saved Table 7 family metrics:", family_out)


if __name__ == "__main__":
    main()
