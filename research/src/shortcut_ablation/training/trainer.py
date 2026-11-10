import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.baselines.et_graphsage.config import Config as ETGraphSAGEConfig
from stage.baselines.flow_transformer.config import Config as FlowTransformerConfig
from stage.baselines.ts_ids.config import Config as TSIDSConfig
from stage.training.trainer import BATCH_SIZE

from shortcut_ablation.config import CHECKPOINT_OUTPUT_DIR
from shortcut_ablation.data.dataset import (
    get_num_classes,
    get_split_graphs,
    load_original_graphs,
)
from shortcut_ablation.models.registry import (
    build_shortcut_model,
    list_shortcut_models,
    normalize_model_name,
)


STAGE_TRAIN_CONFIG = {
    "epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
}

BASELINE_TRAIN_CONFIGS = {
    "TS-IDS": {
        "epochs": 30,
        "learning_rate": TSIDSConfig.learning_rate,
        "weight_decay": 1e-4,
    },
    "FlowTransformer": {
        "epochs": FlowTransformerConfig.epochs,
        "learning_rate": FlowTransformerConfig.learning_rate,
        "weight_decay": 1e-4,
    },
    "E-T-GraphSAGE": {
        "epochs": ETGraphSAGEConfig.epochs,
        "learning_rate": ETGraphSAGEConfig.learning_rate,
        "weight_decay": 1e-4,
    },
}


def _safe_model_name(name):
    return name.replace("/", "_").replace(" ", "_")


def _resolve_model_names(model):
    model = str(model).strip()
    if model.lower() in {"all", "models"}:
        return list_shortcut_models()
    return [normalize_model_name(model)]


def _resolve_folds(fold):
    value = str(fold).strip().lower()
    if value == "all":
        split_files = sorted(Path("outputs/shortcut_ablation/splits").glob("fold_*.json"))
        folds = []
        for path in split_files:
            try:
                folds.append(int(path.stem.split("_")[1]))
            except Exception:
                pass
        if not folds:
            raise FileNotFoundError("No fold_*.json files found under outputs/shortcut_ablation/splits")
        return folds

    return [int(value)]


def _get_train_config(model_name):
    canonical = normalize_model_name(model_name)
    if canonical == "STAGE":
        return STAGE_TRAIN_CONFIG
    return BASELINE_TRAIN_CONFIGS[canonical]


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_weighted_sampler(graphs):
    labels = [int(g.y.item()) for g in graphs]
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1

    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def _run_one_epoch(model, loader, criterion, optimizer, device, train=True, desc=""):
    if train:
        model.train()
    else:
        model.eval()

    preds = []
    targets = []
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in progress:
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            if train:
                optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(batch.y.detach().cpu().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_batches, 1)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
    }


def train_one_model_fold(
    model_name,
    fold,
    epochs=None,
    batch_size=BATCH_SIZE,
    learning_rate=None,
    weight_decay=None,
):
    canonical = normalize_model_name(model_name)
    config = _get_train_config(canonical)

    epochs = int(config["epochs"] if epochs is None else epochs)
    batch_size = int(BATCH_SIZE if batch_size is None else batch_size)
    learning_rate = float(config["learning_rate"] if learning_rate is None else learning_rate)
    weight_decay = float(config["weight_decay"] if weight_decay is None else weight_decay)

    device = _get_device()

    all_graphs = load_original_graphs()
    num_classes = get_num_classes(all_graphs)

    train_graphs, test_graphs = get_split_graphs(fold=fold, domain="original")

    model = build_shortcut_model(canonical, num_classes=num_classes).to(device)

    sampler = _build_weighted_sampler(train_graphs)
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
    )

    safe_name = _safe_model_name(canonical)
    save_dir = CHECKPOINT_OUTPUT_DIR / safe_name
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"{safe_name}_fold_{fold}.pth"
    metadata_path = save_dir / f"{safe_name}_fold_{fold}_metadata.json"

    history = []

    print(f"[INFO] Train shortcut model: {canonical}")
    print(f"[INFO] Fold: {fold}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")
    print(f"[INFO] Num classes: {num_classes}")
    print(f"[INFO] Epochs: {epochs} | Batch size: {batch_size}")
    print(f"[INFO] Learning rate: {learning_rate} | Weight decay: {weight_decay}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")

    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            desc=f"{canonical} fold {fold} epoch {epoch}/{epochs} [train]",
        )

        eval_metrics = _run_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
            desc=f"{canonical} fold {fold} epoch {epoch}/{epochs} [original-test]",
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "original_test_loss": eval_metrics["loss"],
            "original_test_macro_f1": eval_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_macro_f1']:.4f} "
            f"original_test_f1={row['original_test_macro_f1']:.4f}"
        )

    torch.save(
        {
            "model_name": canonical,
            "fold": fold,
            "num_classes": num_classes,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    metadata = {
        "model_name": canonical,
        "fold": fold,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "num_classes": int(num_classes),
        "train_graphs": len(train_graphs),
        "test_graphs": len(test_graphs),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved checkpoint: {checkpoint_path}")
    print(f"[OK] Saved metadata: {metadata_path}")

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train_shortcut_models(
    model="all",
    fold=1,
    epochs=None,
    batch_size=BATCH_SIZE,
    learning_rate=None,
    weight_decay=None,
):
    names = _resolve_model_names(model)
    folds = _resolve_folds(fold)

    for fold_id in folds:
        for name in names:
            train_one_model_fold(
                model_name=name,
                fold=fold_id,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
