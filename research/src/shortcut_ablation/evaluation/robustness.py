import csv
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from shortcut_ablation.config import CHECKPOINT_OUTPUT_DIR, METRIC_OUTPUT_DIR
from shortcut_ablation.data.dataset import (
    get_num_classes,
    get_split_graphs,
    load_original_graphs,
)
from shortcut_ablation.evaluation.metrics import compute_metrics
from shortcut_ablation.models.registry import (
    build_shortcut_model,
    list_shortcut_models,
    normalize_model_name,
)


def _safe_model_name(name):
    return name.replace("/", "_").replace(" ", "_")


def _resolve_model_names(model):
    model = str(model).strip()
    if model.lower() in {"all", "models"}:
        return list_shortcut_models()
    return [normalize_model_name(model)]


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_path(model_name, fold):
    safe_name = _safe_model_name(model_name)
    return CHECKPOINT_OUTPUT_DIR / safe_name / f"{safe_name}_fold_{fold}.pth"


def _predict(model, graphs, batch_size, device):
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            logits = model(batch)
            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(batch.y.detach().cpu().tolist())

    return targets, preds


def evaluate_one_model_fold(model_name, fold, batch_size=2):
    canonical = normalize_model_name(model_name)
    device = _get_device()

    original_graphs = load_original_graphs()
    num_classes = get_num_classes(original_graphs)

    model = build_shortcut_model(canonical, num_classes=num_classes).to(device)

    ckpt_path = _checkpoint_path(canonical, fold)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    _, original_test = get_split_graphs(fold=fold, domain="original")
    _, shifted_test = get_split_graphs(fold=fold, domain="shifted")

    rows = []

    for domain, graphs in [
        ("Original", original_test),
        ("Shifted", shifted_test),
    ]:
        y_true, y_pred = _predict(
            model=model,
            graphs=graphs,
            batch_size=batch_size,
            device=device,
        )

        metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
        row = {
            "model": canonical,
            "fold": fold,
            "domain": domain,
            "num_graphs": len(graphs),
            **metrics,
        }
        rows.append(row)

    original_f1 = rows[0]["f1_macro"]
    shifted_f1 = rows[1]["f1_macro"]
    delta_f1 = shifted_f1 - original_f1

    for row in rows:
        row["delta_f1_shifted_minus_original"] = float(delta_f1)

    return rows


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


def run_robustness(model="all", fold=1, batch_size=2):
    names = _resolve_model_names(model)
    folds = _resolve_folds(fold)

    METRIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_suffix = "all" if str(fold).strip().lower() == "all" else str(fold)
    output_csv = METRIC_OUTPUT_DIR / f"shortcut_robustness_fold_{fold_suffix}.csv"
    output_json = METRIC_OUTPUT_DIR / f"shortcut_robustness_fold_{fold_suffix}.json"

    all_rows = []
    for fold_id in folds:
        for name in names:
            print(f"[INFO] Evaluating {name} fold {fold_id}")
            rows = evaluate_one_model_fold(
                model_name=name,
                fold=fold_id,
                batch_size=batch_size,
            )
            all_rows.extend(rows)

            original = rows[0]
            shifted = rows[1]
            print(
                f"  Original F1={original['f1_macro']:.4f} | "
                f"Shifted F1={shifted['f1_macro']:.4f} | "
                f"Delta={shifted['delta_f1_shifted_minus_original']:.4f}"
            )

    fieldnames = [
        "model",
        "fold",
        "domain",
        "num_graphs",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "fpr_macro",
        "mcc",
        "delta_f1_shifted_minus_original",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print(f"[OK] Robustness CSV saved to: {output_csv}")
    print(f"[OK] Robustness JSON saved to: {output_json}")
