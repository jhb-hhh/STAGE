"""Compact ablation runner for the STAGE artifact."""

import csv
import json
import shutil
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.data.dataset import IDAPTDataset
from stage.evaluation.common import load_label_map, load_split_indices
from stage.evaluation.metrics import calculate_metrics_dict
from stage.models.variants import build_model
from stage.training.checkpoint import load_model_checkpoint
from stage.training.trainer import train_kfold


ABLATION_VARIANTS = [
    "stage_final",
    "gcn_only",
    "lstm_only",
    "inter_serial",
    "fusion_add",
    "fusion_concat",
    "fusion_cross_attn",
    "gcn_trans_rvg",
    "gat_lstm_rvg",
]


def parse_models(models):
    if models is None or models == "all":
        return list(ABLATION_VARIANTS)

    selected = [item.strip() for item in models.split(",") if item.strip()]
    unknown = [item for item in selected if item not in ABLATION_VARIANTS]

    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}. Available: {ABLATION_VARIANTS}")

    return selected


def get_checkpoint(root, variant, checkpoint_type):
    if checkpoint_type == "reference":
        return root / "reference_artifacts/checkpoints/ablation" / f"{variant}_reference_checkpoint.pth"

    if checkpoint_type == "trained":
        return root / "outputs/workspace/checkpoints/ablation" / f"{variant}_trained_checkpoint.pth"

    return Path(checkpoint_type)


def train_ablations(root, models="all", epochs=1, batch_size=4, fold_id=2):
    variants = parse_models(models)
    output_root = root / "outputs/workspace/checkpoints/ablation"
    output_root.mkdir(parents=True, exist_ok=True)

    print("== STAGE compact ablation training ==")
    print("This validates ablation training on the compact artifact subset.")
    print("It is not intended to reproduce full-scale paper metrics.")
    print("variants:", ", ".join(variants))

    for variant in variants:
        print()
        print(f"== Train ablation variant: {variant} ==")

        train_kfold(
            mode_name=variant,
            fold=fold_id,
            batch_size=batch_size,
            epochs=epochs,
            output_root=output_root,
        )

        produced = output_root / f"kfold_experiment_{variant}" / f"best_model_fold_{fold_id}.pth"
        if not produced.exists():
            print(f"[MISSING] expected checkpoint: {produced}")
            return 1

        canonical = output_root / f"{variant}_trained_checkpoint.pth"
        shutil.copy2(produced, canonical)

        try:
            produced.unlink()
        except OSError:
            pass

        print(f"Trained checkpoint: {canonical}")

    return 0


def evaluate_one_variant(root, variant, checkpoint_type="reference", fold_id=2, batch_size=8, device="cpu"):
    checkpoint_path = get_checkpoint(root, variant, checkpoint_type)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {variant}: {checkpoint_path}")

    label_map = load_label_map()
    model = build_model(variant, num_classes=len(label_map)).to(device)

    result = load_model_checkpoint(model, checkpoint_path, map_location=device, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"[WARN] checkpoint key differences for {variant}")
        print("missing:", result.missing_keys)
        print("unexpected:", result.unexpected_keys)

    dataset = IDAPTDataset()
    indices = load_split_indices(fold_id)
    val_dataset = dataset[indices["val"]]
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{variant} compact eval"):
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            logits = model(batch)
            preds = logits.argmax(dim=1)

            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    metrics = calculate_metrics_dict(y_true, y_pred)
    return checkpoint_path, metrics


def eval_ablations(root, models="all", checkpoint_type="reference", fold_id=2, batch_size=8, device="cpu"):
    variants = parse_models(models)

    print("== STAGE compact ablation evaluation ==")
    print("This evaluates ablation variants on the compact artifact subset.")
    print("It validates the ablation evaluation pipeline, not full-scale paper metrics.")
    print("checkpoint:", checkpoint_type)
    print("variants:", ", ".join(variants))

    rows = []
    results = {}

    for variant in variants:
        print()
        print(f"== Evaluate ablation variant: {variant} ==")
        checkpoint_path, metrics = evaluate_one_variant(
            root=root,
            variant=variant,
            checkpoint_type=checkpoint_type,
            fold_id=fold_id,
            batch_size=batch_size,
            device=device,
        )

        results[variant] = {
            "checkpoint": str(checkpoint_path.relative_to(root)) if checkpoint_path.is_relative_to(root) else str(checkpoint_path),
            "metrics": metrics,
        }

        row = {"Variant": variant}
        row.update(metrics)
        rows.append(row)

        for key, value in metrics.items():
            print(f"{key}: {value}")

    out_dir = root / "outputs/reproduce/ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"ablation_metrics_{checkpoint_type}.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": checkpoint_type,
                "fold": fold_id,
                "note": "Compact ablation evaluation on sanitized sample data.",
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = out_dir / f"ablation_metrics_{checkpoint_type}.csv"
    fieldnames = ["Variant", "Accuracy", "Precision", "Recall", "F1_Macro", "FPR", "MCC"]
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")

    from table_utils import print_csv_table
    print_csv_table(csv_path, title="Ablation evaluation table")

    return 0
