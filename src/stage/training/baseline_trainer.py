"""Training utilities for baseline models.

This module trains the baseline models used by the checkpoint-based
evaluation code. Checkpoints are saved directly under:

    get_path("outputs", "baseline_checkpoint_dir")

with names such as:

    TS-IDS_fold_0.pth
    FlowTransformer_fold_0.pth
    E-T-GraphSAGE_fold_0.pth
"""

import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.baselines.registry import (
    CANONICAL_BASELINE_NAMES,
    build_baseline,
    normalize_baseline_name,
)
from stage.baselines.et_graphsage.config import Config as ETGraphSAGEConfig
from stage.baselines.flow_transformer.config import Config as FlowTransformerConfig
from stage.baselines.ts_ids.config import Config as TSIDSConfig
from stage.data.dataset import IDAPTDataset
from stage.training.trainer import (
    BATCH_SIZE,
    DEVICE,
    get_sampler,
    load_num_classes,
    load_split_indices,
    setup_logger,
)
from stage.utils.config import get_path

BASELINE_TRAIN_CONFIGS = {
    "TS-IDS": {
        "learning_rate": TSIDSConfig.learning_rate,
        "epochs": TSIDSConfig.epochs,
    },
    "FlowTransformer": {
        "learning_rate": FlowTransformerConfig.learning_rate,
        "epochs": FlowTransformerConfig.epochs,
    },
    "E-T-GraphSAGE": {
        "learning_rate": ETGraphSAGEConfig.learning_rate,
        "epochs": ETGraphSAGEConfig.epochs,
    },
}


def get_baseline_train_config(name):
    """Return training defaults for one canonical baseline name."""
    canonical_name = normalize_baseline_name(name)
    return BASELINE_TRAIN_CONFIGS[canonical_name]


def train_baseline_fold(
    name,
    fold_id,
    save_root_dir,
    full_dataset,
    batch_size=BATCH_SIZE,
    epochs=None,
    learning_rate=None,
    device=DEVICE,
):
    """Train one baseline model on one fold."""
    canonical_name = normalize_baseline_name(name)
    config = get_baseline_train_config(canonical_name)

    epochs = int(config["epochs"] if epochs is None else epochs)
    learning_rate = float(config["learning_rate"] if learning_rate is None else learning_rate)

    save_root_dir = Path(save_root_dir)
    save_root_dir.mkdir(parents=True, exist_ok=True)

    safe_name = canonical_name.replace("/", "_")
    log_file = save_root_dir / f"train_{safe_name}_fold_{fold_id}.log"
    logger = setup_logger(f"Baseline_{safe_name}_{fold_id}", log_file)

    logger.info(f"Start Fold {fold_id} | Baseline: {canonical_name}")
    logger.info(f"Learning rate: {learning_rate} | Epochs: {epochs}")

    indices = load_split_indices(fold_id)

    train_dataset = full_dataset[indices["train"]]
    val_dataset = full_dataset[indices["val"]]

    num_classes = load_num_classes()

    sampler = get_sampler(full_dataset, indices["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_baseline(
        canonical_name,
        num_classes=num_classes,
        in_channels=8,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    model_save_path = save_root_dir / f"{canonical_name}_fold_{fold_id}.pth"

    for epoch in range(epochs):
        model.train()
        train_preds, train_targets = [], []

        train_progress = tqdm(
            train_loader,
            desc=f"{canonical_name} Fold {fold_id} Epoch {epoch + 1}/{epochs} [train]",
            leave=False,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )

        for batch in train_progress:
            batch = batch.to(device, non_blocking=True)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_progress.set_postfix(loss=f"{loss.item():.4f}")

            train_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
            train_targets.extend(batch.y.detach().cpu().numpy())

        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            val_progress = tqdm(
                val_loader,
                desc=f"{canonical_name} Fold {fold_id} Epoch {epoch + 1}/{epochs} [val]",
                leave=False,
                dynamic_ncols=True,
                disable=not sys.stderr.isatty(),
            )

            for batch in val_progress:
                batch = batch.to(device, non_blocking=True)
                batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

                logits = model(batch)
                val_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
                val_targets.extend(batch.y.detach().cpu().numpy())

        train_f1 = f1_score(train_targets, train_preds, average="macro")
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}: Train F1={train_f1:.4f} | Val F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model: F1={best_val_f1:.4f}")

    logger.info(f"Fold {fold_id} done. Best F1: {best_val_f1:.4f}")

    del model, optimizer, scheduler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


def train_baseline_kfold(
    name,
    fold=-1,
    batch_size=BATCH_SIZE,
    epochs=None,
    output_dir=None,
):
    """Train one baseline on one fold or all five folds."""
    canonical_name = normalize_baseline_name(name)

    if output_dir is None:
        output_dir = get_path("outputs", "baseline_checkpoint_dir")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = IDAPTDataset()

    if fold != -1:
        train_baseline_fold(
            name=canonical_name,
            fold_id=fold,
            save_root_dir=output_dir,
            full_dataset=full_dataset,
            batch_size=batch_size,
            epochs=epochs,
        )
        return

    for fold_id in range(5):
        train_baseline_fold(
            name=canonical_name,
            fold_id=fold_id,
            save_root_dir=output_dir,
            full_dataset=full_dataset,
            batch_size=batch_size,
            epochs=epochs,
        )


def train_all_baselines(
    fold=-1,
    batch_size=BATCH_SIZE,
    output_dir=None,
    names=None,
):
    """Train all registered baseline models."""
    names = list(names or CANONICAL_BASELINE_NAMES)

    for name in names:
        train_baseline_kfold(
            name=name,
            fold=fold,
            batch_size=batch_size,
            output_dir=output_dir,
        )


def print_baseline_training_plan(output_dir=None, names=None, fold=-1):
    """Print the baseline training plan without loading graph data."""
    names = list(names or CANONICAL_BASELINE_NAMES)
    output_dir = Path(output_dir or get_path("outputs", "baseline_checkpoint_dir"))

    if fold == -1:
        fold_text = "0, 1, 2, 3, 4"
    else:
        fold_text = str(fold)

    print("Baseline training plan")
    print("output_dir:", output_dir)
    print("folds:", fold_text)
    print("models:")

    for name in names:
        canonical_name = normalize_baseline_name(name)
        config = get_baseline_train_config(canonical_name)
        print(
            f"  - {canonical_name}: "
            f"epochs={config['epochs']}, "
            f"learning_rate={config['learning_rate']}"
        )
