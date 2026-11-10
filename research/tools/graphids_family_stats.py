"""Family-conditioned GraphIDS detection metrics for Table 7.

For each malware family, evaluate binary GraphIDS on that family versus benign
graphs from the same outer test fold. This uses family labels only to stratify
the test set; the model still predicts only benign/attack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from advanced_robustness import _domain_data, _load_model, _predict  # noqa: E402
from stage.data.dataset import IDAPTDataset  # noqa: E402
from stage.utils.config import get_path  # noqa: E402


OUT = ROOT / "outputs" / "workspace" / "metrics" / "advanced_robustness"


def evaluate(fold: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = IDAPTDataset(root_dir=get_path("data", "graph_dir"))
    model, state = _load_model("GraphIDS", fold, "w/", original, device)
    indices, shifted, shifted_indices = _domain_data("w/", fold, original)
    threshold = state["threshold"]
    _, original_pred, _ = _predict(
        model, "GraphIDS", original, indices, "binary", device, threshold
    )
    _, shifted_pred, _ = _predict(
        model, "GraphIDS", shifted, shifted_indices, "binary", device, threshold
    )
    native = np.asarray([int(original[i].y.view(-1)[0]) for i in indices])

    with (ROOT / "data" / "reference" / "graphs" / "label_mapping.json").open(encoding="utf-8") as handle:
        label_map = json.load(handle)
    inverse = {int(value): key for key, value in label_map.items()}
    rows = []
    for family in sorted(value for value in inverse if value != 0):
        mask = (native == 0) | (native == family)
        target = (native[mask] == family).astype(np.int64)
        for domain, predictions in (("Original", original_pred), ("Shifted", shifted_pred)):
            precision, recall, f1, _ = precision_recall_fscore_support(
                target,
                np.asarray(predictions)[mask],
                average="binary",
                zero_division=0,
            )
            rows.append(
                {
                    "Model": "GraphIDS",
                    "Fold": fold,
                    "Malware Family": inverse[family],
                    "Test Domain": domain,
                    "F1-Score": f1,
                    "Recall": recall,
                    "Precision": precision,
                    "Support": int(target.sum()),
                    "Protocol": "family-vs-benign binary detection",
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"table7_graphids_fold_{fold}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Saved", path)


def aggregate():
    paths = [OUT / f"table7_graphids_fold_{fold}.csv" for fold in range(5)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fold files: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    mean = frame.groupby(["Malware Family", "Test Domain"], as_index=False)[
        ["F1-Score", "Recall", "Precision", "Support"]
    ].mean()
    original = mean[mean["Test Domain"] == "Original"].set_index("Malware Family")
    shifted = mean[mean["Test Domain"] == "Shifted"].set_index("Malware Family")
    delta = (shifted["F1-Score"] - original["F1-Score"]) / original["F1-Score"].replace(0, np.nan) * 100
    rows = []
    for family in delta.sort_values().head(3).index:
        for domain, values in (("Original", original), ("Shifted", shifted)):
            item = values.loc[family]
            rows.append(
                {
                    "Model": "GraphIDS",
                    "Malware Family": family,
                    "Test Domain": domain,
                    "F1-Score": f"{item['F1-Score']:.4f}",
                    "Recall": f"{item['Recall']:.4f}",
                    "Precision": f"{item['Precision']:.4f}",
                    "Delta F1 (%)": f"{delta.loc[family]:+.2f}%" if domain == "Shifted" else "",
                    "Task": "family-vs-benign binary detection",
                }
            )
    path = OUT / "table7_graphids_top3_fivefold.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Saved", path)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.fold is not None:
        evaluate(args.fold)
    else:
        parser.error("select --fold or --aggregate")


if __name__ == "__main__":
    main()
