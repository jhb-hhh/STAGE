"""Training utilities for STAGE and ablation variants."""

import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.data.dataset import IDAPTDataset
from stage.models.variants import MODEL_REGISTRY, build_model
from stage.utils.config import get_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 20


def get_sampler(dataset, indices):
    """Create a weighted sampler for class-balanced training."""
    y_train = np.array([dataset[i].y.item() for i in indices])
    class_sample_counts = np.bincount(y_train)

    with np.errstate(divide="ignore"):
        weight_per_class = 1.0 / class_sample_counts

    weight_per_class = np.where(weight_per_class == float("inf"), 0, weight_per_class)
    sample_weights = torch.from_numpy(weight_per_class[y_train]).type(torch.FloatTensor)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def load_split_indices(fold_id):
    """Load train and validation indices for one fold."""
    split_path = get_path("data", "split_dir") / f"fold_{fold_id}.json"

    with open(split_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_num_classes():
    """Return the number of labels in the ID-APT label map."""
    with open(get_path("data", "label_map"), "r", encoding="utf-8") as file:
        label_map = json.load(file)

    return len(label_map)


def setup_logger(name, log_file):
    """Create a logger that writes to both file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def normalize_mode_name(mode_name):
    """Normalize public model names to registry names."""
    if mode_name == "stage":
        return "stage_final"

    return mode_name


def train_fold(
    fold_id,
    mode_name,
    save_root_dir,
    full_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    device=DEVICE,
):
    """Train one model variant on one fold."""
    mode_name = normalize_mode_name(mode_name)
    save_root_dir = Path(save_root_dir)
    save_root_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_root_dir / f"train_fold_{fold_id}.log"
    logger = setup_logger(f"STAGE_{mode_name}_{fold_id}", log_file)

    logger.info(f"Start Fold {fold_id} | Mode: {mode_name}")

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

    try:
        model = build_model(mode_name, num_classes=num_classes).to(device)
    except Exception as exc:
        logger.error(f"Model initialization failed: {exc}")
        raise

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    model_save_path = save_root_dir / f"best_model_fold_{fold_id}.pth"

    for epoch in range(epochs):
        model.train()
        train_preds, train_targets = [], []

        train_progress = tqdm(
            train_loader,
            desc=f"Fold {fold_id} Epoch {epoch + 1}/{epochs} [train]",
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
                desc=f"Fold {fold_id} Epoch {epoch + 1}/{epochs} [val]",
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

        logger.info(f"Epoch {epoch + 1}: Train F1={train_f1:.4f} | Val F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model: F1={best_val_f1:.4f}")

    logger.info(f"Fold {fold_id} done. Best F1: {best_val_f1:.4f}")

    del model, optimizer, scheduler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


def train_kfold(
    mode_name,
    fold=-1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    output_root=None,
):
    """Train one model variant on one fold or all five folds."""
    mode_name = normalize_mode_name(mode_name)

    if mode_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown mode: {mode_name}. Available: {available}")

    if output_root is None:
        output_root = get_path("outputs", "checkpoint_dir") / "training_runs"

    save_dir = Path(output_root) / f"kfold_experiment_{mode_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = IDAPTDataset()

    if fold != -1:
        train_fold(
            fold_id=fold,
            mode_name=mode_name,
            save_root_dir=save_dir,
            full_dataset=full_dataset,
            batch_size=batch_size,
            epochs=epochs,
        )
    else:
        for fold_id in range(5):
            train_fold(
                fold_id=fold_id,
                mode_name=mode_name,
                save_root_dir=save_dir,
                full_dataset=full_dataset,
                batch_size=batch_size,
                epochs=epochs,
            )
