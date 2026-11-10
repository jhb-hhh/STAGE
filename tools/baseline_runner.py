"""Compact baseline runner for the STAGE artifact."""

import csv
import json
import shutil
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.baselines.registry import (
    CANONICAL_BASELINE_NAMES,
    build_baseline,
    normalize_baseline_name,
)
from stage.data.dataset import IDAPTDataset
from stage.evaluation.common import load_label_map, load_split_indices
from stage.evaluation.metrics import calculate_metrics_dict
from stage.evaluation.baselines import load_baseline_checkpoint
from stage.training.baseline_trainer import train_baseline_kfold


def parse_models(models):
    if models is None or models == "all":
        return list(CANONICAL_BASELINE_NAMES)

    selected = [item.strip() for item in models.split(",") if item.strip()]
    return [normalize_baseline_name(item) for item in selected]


def get_checkpoint(root, name, checkpoint_type):
    canonical = normalize_baseline_name(name)

    if checkpoint_type == "reference":
        return root / "reference_artifacts/checkpoints/baselines" / f"{canonical}_reference_checkpoint.pth"

    if checkpoint_type == "trained":
        return root / "outputs/workspace/checkpoints/baselines" / f"{canonical}_trained_checkpoint.pth"

    return Path(checkpoint_type)


def train_baselines(root, models="all", epochs=1, batch_size=4, fold_id=2):
    names = parse_models(models)
    output_dir = root / "outputs/workspace/checkpoints/baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("== STAGE compact baseline training ==")
    print("This validates baseline training on the compact artifact subset.")
    print("It is not intended to reproduce full-scale paper metrics.")
    print("baselines:", ", ".join(names))

    for name in names:
        print()
        print(f"== Train baseline: {name} ==")

        train_baseline_kfold(
            name=name,
            fold=fold_id,
            batch_size=batch_size,
            epochs=epochs,
            output_dir=output_dir,
        )

        produced = output_dir / f"{name}_fold_{fold_id}.pth"
        if not produced.exists():
            print(f"[MISSING] expected checkpoint: {produced}")
            return 1

        canonical = output_dir / f"{name}_trained_checkpoint.pth"
        shutil.copy2(produced, canonical)

        try:
            produced.unlink()
        except OSError:
            pass

        print(f"Trained checkpoint: {canonical}")

    return 0


def evaluate_one_baseline(root, name, checkpoint_type="reference", fold_id=2, batch_size=8, device="cpu"):
    canonical = normalize_baseline_name(name)
    checkpoint_path = get_checkpoint(root, canonical, checkpoint_type)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {canonical}: {checkpoint_path}")

    label_map = load_label_map()
    model = build_baseline(canonical, num_classes=len(label_map), in_channels=8).to(device)

    result = load_baseline_checkpoint(model, checkpoint_path, device)
    if result.missing_keys or result.unexpected_keys:
        print(f"[WARN] checkpoint key differences for {canonical}")
        print("missing:", result.missing_keys)
        print("unexpected:", result.unexpected_keys)

    dataset = IDAPTDataset()
    indices = load_split_indices(fold_id)
    val_dataset = dataset[indices["val"]]
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{canonical} compact eval"):
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            logits = model(batch)
            preds = logits.argmax(dim=1)

            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    metrics = calculate_metrics_dict(y_true, y_pred)
    return checkpoint_path, metrics


def eval_baselines(root, models="all", checkpoint_type="reference", fold_id=2, batch_size=8, device="cpu"):
    names = parse_models(models)

    print("== STAGE compact baseline evaluation ==")
    print("This evaluates baseline models on the compact artifact subset.")
    print("It validates the baseline evaluation pipeline, not full-scale paper metrics.")
    print("checkpoint:", checkpoint_type)
    print("baselines:", ", ".join(names))

    rows = []
    results = {}

    for name in names:
        print()
        print(f"== Evaluate baseline: {name} ==")

        checkpoint_path, metrics = evaluate_one_baseline(
            root=root,
            name=name,
            checkpoint_type=checkpoint_type,
            fold_id=fold_id,
            batch_size=batch_size,
            device=device,
        )

        results[name] = {
            "checkpoint": str(checkpoint_path.relative_to(root)) if checkpoint_path.is_relative_to(root) else str(checkpoint_path),
            "metrics": metrics,
        }

        row = {"Model": name}
        row.update(metrics)
        rows.append(row)

        for key, value in metrics.items():
            print(f"{key}: {value}")

    out_dir = root / "outputs/reproduce/baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"baseline_metrics_{checkpoint_type}.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": checkpoint_type,
                "fold": fold_id,
                "note": "Compact baseline evaluation on sanitized sample data.",
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = out_dir / f"baseline_metrics_{checkpoint_type}.csv"
    fieldnames = ["Model", "Accuracy", "Precision", "Recall", "F1_Macro", "FPR", "MCC"]
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")

    from table_utils import print_csv_table
    print_csv_table(csv_path, title="Baseline evaluation table")

    return 0
