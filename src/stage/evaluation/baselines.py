"""Evaluation utilities for baseline models."""

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.baselines.registry import build_baseline
from stage.data.dataset import IDAPTDataset
from stage.evaluation.common import load_label_map, load_split_indices, resolve_device
from stage.evaluation.metrics import calculate_metrics_dict, summarize_fold_metrics
from stage.utils.config import get_path


def get_baseline_checkpoint_path(name, fold_id):
    """Return the checkpoint path for one baseline and fold."""
    return get_path("outputs", "baseline_checkpoint_dir") / f"{name}_fold_{fold_id}.pth"


def load_baseline_checkpoint(model, path, device):
    """Load a baseline checkpoint into a model instance."""
    state_dict = torch.load(path, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    return model.load_state_dict(state_dict, strict=False)


def evaluate_baseline_fold(name, fold_id, dataset=None, batch_size=16, device=None):
    """Evaluate one baseline model on one validation fold."""
    device = resolve_device(device)

    label_map = load_label_map()
    num_classes = len(label_map)

    if dataset is None:
        dataset = IDAPTDataset()

    indices = load_split_indices(fold_id)
    val_dataset = dataset[indices["val"]]
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_baseline(name, num_classes=num_classes, in_channels=8).to(device)
    ckpt_path = get_baseline_checkpoint_path(name, fold_id)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_result = load_baseline_checkpoint(model, ckpt_path, device)

    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"[WARN] checkpoint key mismatch for {name} fold {fold_id}")
        print("missing:", load_result.missing_keys)
        print("unexpected:", load_result.unexpected_keys)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{name} fold {fold_id}"):
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            logits = model(batch)
            preds = logits.argmax(dim=1)

            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return calculate_metrics_dict(y_true, y_pred)


def evaluate_baseline_all_folds(name, folds=(0, 1, 2, 3, 4), dataset=None, batch_size=16, device=None):
    """Evaluate one baseline model over multiple folds."""
    if dataset is None:
        dataset = IDAPTDataset()

    fold_metrics = []

    for fold_id in folds:
        metrics = evaluate_baseline_fold(
            name=name,
            fold_id=fold_id,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )
        fold_metrics.append(metrics)

    return summarize_fold_metrics(fold_metrics), fold_metrics


if __name__ == "__main__":
    metrics = evaluate_baseline_fold("FlowTransformer", fold_id=2, batch_size=16)
    print(metrics)
