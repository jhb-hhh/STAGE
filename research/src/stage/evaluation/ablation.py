"""Evaluation utilities for STAGE ablation variants."""

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.data.dataset import IDAPTDataset
from stage.evaluation.common import load_label_map, load_split_indices, resolve_device
from stage.evaluation.metrics import calculate_metrics_dict, summarize_fold_metrics
from stage.models.variants import build_model
from stage.training.checkpoint import load_model_checkpoint
from stage.utils.config import get_path


def get_ablation_checkpoint_path(variant, fold_id):
    """Return the checkpoint path for one ablation variant and fold."""
    root = get_path("outputs", "ablation_checkpoint_dir")
    return root / f"kfold_experiment_{variant}" / f"best_model_fold_{fold_id}.pth"


def evaluate_variant_fold(variant, fold_id, dataset=None, batch_size=16, device=None):
    """Evaluate one ablation variant on one validation fold."""
    device = resolve_device(device)

    label_map = load_label_map()
    num_classes = len(label_map)

    if dataset is None:
        dataset = IDAPTDataset()

    indices = load_split_indices(fold_id)
    val_dataset = dataset[indices["val"]]
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(variant, num_classes=num_classes).to(device)
    ckpt_path = get_ablation_checkpoint_path(variant, fold_id)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_result = load_model_checkpoint(model, ckpt_path, map_location=device, strict=False)

    allowed_missing = {"pos_encoder"}
    missing = [key for key in load_result.missing_keys if key not in allowed_missing]
    unexpected = load_result.unexpected_keys

    if missing or unexpected:
        print(f"[WARN] checkpoint key mismatch for {variant} fold {fold_id}")
        print("missing:", load_result.missing_keys)
        print("unexpected:", load_result.unexpected_keys)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{variant} fold {fold_id}"):
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            logits = model(batch)
            preds = logits.argmax(dim=1)

            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return calculate_metrics_dict(y_true, y_pred)


def evaluate_variant_all_folds(variant, folds=(0, 1, 2, 3, 4), dataset=None, batch_size=16, device=None):
    """Evaluate one ablation variant over multiple folds."""
    if dataset is None:
        dataset = IDAPTDataset()

    fold_metrics = []

    for fold_id in folds:
        metrics = evaluate_variant_fold(
            variant=variant,
            fold_id=fold_id,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )
        fold_metrics.append(metrics)

    return summarize_fold_metrics(fold_metrics), fold_metrics


if __name__ == "__main__":
    summary, fold_metrics = evaluate_variant_all_folds("stage_final", folds=(2,), batch_size=16)
    print("fold metrics:", fold_metrics)
    print("summary:", summary)
