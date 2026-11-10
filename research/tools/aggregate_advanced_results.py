"""Create paper-ready new-baseline tables only when every required fold exists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "workspace" / "metrics"
ROBUST = METRICS / "advanced_robustness"
SOURCE = METRICS / "source_metrics"
MODELS = ("DIDS-MFL", "GraphIDS")
CONDITIONS = ("w/o", "w/")
DOMAINS = ("Original", "Shifted")
FIELDS = ("F1-Score", "Recall", "Precision", "Accuracy")


def _fold_file(name, condition, fold):
    safe_name = name.lower().replace("-", "_")
    safe_condition = condition.replace("/", "")
    return ROBUST / f"table6_{safe_name}_{safe_condition}_fold_{fold}.csv"


def _require(paths):
    missing = [path for path in paths if not path.is_file()]
    if missing:
        print("Missing required results:")
        for path in missing:
            print(path)
        raise FileNotFoundError(f"{len(missing)} required result files missing; no final table written")


def aggregate_robustness():
    paths = [_fold_file(name, condition, fold)
             for name in MODELS for condition in CONDITIONS for fold in range(5)]
    family_paths = [ROBUST / f"table7_dids_mfl_fold_{fold}.csv" for fold in range(5)]
    _require(paths + family_paths)
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    table6 = []
    for (name, condition, task), group in rows.groupby(["Model", "Condition", "Task"]):
        if set(group["Fold"]) != set(range(5)):
            raise ValueError(f"Incomplete fivefold metrics for {name} {condition}")
        for domain in DOMAINS:
            part = group[group["Test Domain"] == domain]
            row = {"Model": name, "Condition": condition, "Task": task,
                   "Test Domain": domain, "Folds": len(part)}
            for field in FIELDS:
                row[field] = f"{part[field].mean():.4f}±{part[field].std(ddof=1):.4f}"
            row["Delta F1 (%)"] = (
                f"{part['Delta F1 (%)'].mean():+.2f}%" if domain == "Shifted" else ""
            )
            table6.append(row)
    ROBUST.mkdir(parents=True, exist_ok=True)
    table6_path = ROBUST / "table6_new_baselines_fivefold.csv"
    pd.DataFrame(table6).to_csv(table6_path, index=False)
    print("Saved", table6_path)

    families = pd.concat([pd.read_csv(path) for path in family_paths], ignore_index=True)
    aggregated = families.groupby(["Malware Family", "Test Domain"], as_index=False)[
        ["F1-Score", "Recall", "Precision", "Support"]
    ].mean()
    original = aggregated[aggregated["Test Domain"] == "Original"].set_index("Malware Family")
    shifted = aggregated[aggregated["Test Domain"] == "Shifted"].set_index("Malware Family")
    drops = ((shifted["F1-Score"] - original["F1-Score"])
             / original["F1-Score"].replace(0, float("nan")) * 100)
    table7 = []
    for family in drops.sort_values().head(3).index:
        for domain, frame in (("Original", original), ("Shifted", shifted)):
            item = frame.loc[family]
            table7.append({"Model": "DIDS-MFL", "Malware Family": family,
                           "Test Domain": domain, "F1-Score": f"{item['F1-Score']:.4f}",
                           "Recall": f"{item['Recall']:.4f}",
                           "Precision": f"{item['Precision']:.4f}",
                           "Delta F1 (%)": f"{drops.loc[family]:+.2f}%" if domain == "Shifted" else "",
                           "Task": "13-class ID-APT family classification"})
    table7_path = ROBUST / "table7_dids_mfl_top3_fivefold.csv"
    pd.DataFrame(table7).to_csv(table7_path, index=False)
    print("Saved", table7_path)
    print("GraphIDS Table 7: N/A (original method outputs only Benign/Attack)")


def aggregate_detection():
    paths = []
    for dataset in ("id_apt", "dapt2020", "cic_ids_2018_plus_sandworm"):
        for name in MODELS:
            safe_name = name.lower().replace("-", "_")
            paths.append(SOURCE / f"baseline_detection_{dataset}_{safe_name}.csv")
    _require(paths)
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(combined) != 6 or set(combined["Model"]) != set(MODELS):
        raise ValueError("Unexpected final detection rows")
    out = SOURCE / "final_detection_new_baselines_three_datasets.csv"
    combined.to_csv(out, index=False)
    print("Saved", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("detection", "robustness", "all"), default="all")
    args = parser.parse_args()
    if args.task in ("robustness", "all"):
        aggregate_robustness()
    if args.task in ("detection", "all"):
        aggregate_detection()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[NOT READY] {error}", file=sys.stderr)
        sys.exit(1)
