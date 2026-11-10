"""Generate source metrics from reference or workspace checkpoints.

This script is a low-level evaluation entry point used by tools/run.py.
It evaluates STAGE, baseline, ablation, and robustness tasks and writes
source metric CSV files to the selected output directory.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from stage.baselines.registry import CANONICAL_BASELINE_NAMES, LEGACY_BASELINE_NAMES
from stage.data.dataset import IDAPTDataset
from stage.data.baseline_protocol import load_baseline_dataset, normalize_dataset_name
from stage.evaluation.ablation import evaluate_variant_all_folds, evaluate_variant_fold
from stage.evaluation.baselines import evaluate_baseline_all_folds, evaluate_baseline_fold
from stage.evaluation.detection import evaluate_stage_fold
from stage.evaluation.metrics import summarize_fold_metrics
from stage.evaluation.robustness import (
    evaluate_stage_robustness,
    evaluate_baseline_robustness_all,
    load_original_val_graphs,
    load_shifted_val_graphs,
)
from stage.training.advanced_baseline_trainer import ADVANCED_BASELINES
from stage.utils.config import get_path


METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1_Macro", "FPR", "MCC"]

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

PRESERVED_SOURCE_FILES = [
    "baseline_detection.csv",
    "ablation_summary.csv",
    "robustness_stage.csv",
    "robustness_baselines.csv",
    "explanation_benchmark.csv",
]


def get_folds(scope, fold):
    if scope == "all":
        return [0, 1, 2, 3, 4]
    return [fold]


def format_metric_value(value, digits=4):
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def normalize_metric_row(metrics):
    return {col: format_metric_value(metrics[col]) for col in METRIC_COLUMNS}


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"saved: {path}")
    print(df.to_string(index=False))
    print()



def evaluate_stage_detection(scope, fold, batch_size, output_dir, dataset=None):
    folds = get_folds(scope, fold)
    fold_metrics = []

    for fold_id in folds:
        metrics = evaluate_stage_fold(fold_id=fold_id, dataset=dataset, batch_size=batch_size)
        row = {"Model": "STAGE", "Fold": fold_id}
        row.update(normalize_metric_row(metrics))
        fold_metrics.append(metrics)

    if scope == "all":
        summary = summarize_fold_metrics(fold_metrics)
        rows = [{"Model": "STAGE", "Fold": "all", **summary}]
    else:
        rows = [{"Model": "STAGE", "Fold": folds[0], **normalize_metric_row(fold_metrics[0])}]

    df = pd.DataFrame(rows)
    write_csv(df, Path(output_dir) / "stage_detection.csv")
    return df


def evaluate_baseline_detection(
    scope,
    fold,
    batch_size,
    output_dir,
    dataset=None,
    model_name=None,
    dataset_name="ID-APT",
    include_advanced=False,
):
    rows = []

    baseline_names = list(
        CANONICAL_BASELINE_NAMES if include_advanced else LEGACY_BASELINE_NAMES
    )
    if model_name:
        if model_name not in baseline_names:
            raise ValueError(
                f"Unknown baseline model: {model_name}. "
                f"Valid baselines: {', '.join(baseline_names)}"
            )
        baseline_names = [model_name]

    for name in baseline_names:
        if scope == "all":
            summary, _ = evaluate_baseline_all_folds(
                name=name,
                folds=(0, 1, 2, 3, 4),
                dataset=dataset,
                batch_size=batch_size,
                dataset_name=dataset_name,
            )
            row = {"Model": name}
            row.update(summary)
        else:
            metrics = evaluate_baseline_fold(
                name=name,
                fold_id=fold,
                dataset=dataset,
                batch_size=batch_size,
                dataset_name=dataset_name,
            )
            row = {
                "Model": name,
                "Dataset": metrics.get("Dataset", dataset_name),
                "Task": metrics.get("Task", "native"),
            }
            row.update(normalize_metric_row(metrics))

        rows.append(row)

    df = pd.DataFrame(rows)
    if dataset_name == "ID-APT" and model_name in ADVANCED_BASELINES:
        safe_model = model_name.lower().replace("-", "_")
        filename = f"baseline_detection_id_apt_{safe_model}.csv"
    elif dataset_name == "ID-APT":
        filename = "baseline_detection.csv"
    else:
        safe_dataset = dataset_name.lower().replace("+", "_plus_").replace("-", "_")
        safe_model = (model_name or "all").lower().replace("-", "_")
        filename = f"baseline_detection_{safe_dataset}_{safe_model}.csv"
    write_csv(df, Path(output_dir) / filename)
    return df


def evaluate_ablation_summary(scope, fold, batch_size, output_dir, dataset=None, model_name=None):
    rows = []

    variants = list(ABLATION_VARIANTS)
    if model_name:
        if model_name not in variants:
            raise ValueError(
                f"Unknown ablation model: {model_name}. "
                f"Valid ablations: {', '.join(variants)}"
            )
        variants = [model_name]

    for variant in variants:
        if scope == "all":
            summary, _ = evaluate_variant_all_folds(
                variant=variant,
                folds=(0, 1, 2, 3, 4),
                dataset=dataset,
                batch_size=batch_size,
            )
            row = {"Mode": variant}
            row.update(summary)
        else:
            metrics = evaluate_variant_fold(
                variant=variant,
                fold_id=fold,
                dataset=dataset,
                batch_size=batch_size,
            )
            row = {"Mode": variant}
            row.update(normalize_metric_row(metrics))

        rows.append(row)

    df = pd.DataFrame(rows)
    write_csv(df, Path(output_dir) / "ablation_summary.csv")
    return df


def evaluate_robustness(scope, fold, batch_size, output_dir):
    if scope == "all":
        raise ValueError("Robustness source metrics are fold-specific. Use --scope fold or --scope smoke.")

    original_graphs = load_original_val_graphs(fold)
    shifted_graphs = load_shifted_val_graphs(fold)

    stage_df = evaluate_stage_robustness(
        fold_id=fold,
        batch_size=batch_size,
        save=False,
        original_graphs=original_graphs,
        shifted_graphs=shifted_graphs,
    )
    write_csv(stage_df, Path(output_dir) / "robustness_stage.csv")

    baseline_df = evaluate_baseline_robustness_all(
        fold_id=fold,
        batch_size=batch_size,
        save=False,
        original_graphs=original_graphs,
        shifted_graphs=shifted_graphs,
    )

    if "Fold" in baseline_df.columns:
        baseline_df = baseline_df.drop(columns=["Fold"])

    write_csv(baseline_df, Path(output_dir) / "robustness_baselines.csv")

    return {
        "stage": stage_df,
        "baselines": baseline_df,
    }


def write_manifest(output_dir, manifest):
    path = Path(output_dir) / "evaluate_all_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest saved: {path}")


def print_plan(args):
    print("Checkpoint-based source-metric evaluation plan")
    print("scope:", args.scope)
    print("fold:", args.fold)
    print("batch_size:", args.batch_size)
    print("output_dir:", args.output_dir)
    print("tasks:", ", ".join(args.tasks))
    print("model_name:", args.model_name or "all")
    print("dataset:", args.dataset)
    print()

    if args.scope == "all":
        print("folds: 0, 1, 2, 3, 4")
    else:
        print("folds:", args.fold)

    print()
    print("This command can recompute STAGE, baseline, and ablation metrics from checkpoints.")


def build_parser():
    parser = argparse.ArgumentParser(description="Regenerate source metrics from checkpoints.")
    parser.add_argument(
        "--scope",
        type=str,
        default="smoke",
        choices=["smoke", "fold", "all"],
        help="Evaluation scope. smoke only writes an evaluation manifest.",
    )
    parser.add_argument(
        "--include-advanced",
        action="store_true",
        help="Include DIDS-MFL and GraphIDS when --model-name is not set.",
    )
    parser.add_argument(
        "--dataset",
        default="ID-APT",
        help="Dataset for advanced baselines: ID-APT, DAPT2020, or CIC-IDS-2018+Sandworm.",
    )
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(get_path("outputs", "metrics_dir") / "source_metrics"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["stage", "baselines", "ablations"],
        choices=["stage", "baselines", "ablations", "robustness"],
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model filter for baselines or ablations.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the plan without running evaluation.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    args.output_dir = str(output_dir)

    print_plan(args)

    manifest = {
        "scope": args.scope,
        "fold": args.fold,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
        "tasks": args.tasks,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "include_advanced": args.include_advanced,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "planned",
    }

    if args.plan_only:
        write_manifest(output_dir, manifest)
        return

    args.dataset = normalize_dataset_name(args.dataset)
    if args.dataset != "ID-APT" and any(task in args.tasks for task in ["stage", "ablations"]):
        raise ValueError(
            "External-dataset evaluation is currently supported for the advanced "
            "baselines only; select --tasks baselines."
        )

    shared_dataset = None
    if args.scope != "smoke" and args.dataset == "ID-APT" and any(task in args.tasks for task in ["stage", "baselines", "ablations"]):
        print("Loading shared graph dataset once for checkpoint evaluation...")
        shared_dataset = IDAPTDataset()
    elif args.scope != "smoke" and "baselines" in args.tasks:
        _, shared_dataset = load_baseline_dataset(args.dataset)

    if args.scope == "smoke":
        manifest["status"] = "ok"
        write_manifest(output_dir, manifest)
        return

    if "stage" in args.tasks:
        evaluate_stage_detection(
            args.scope,
            args.fold,
            args.batch_size,
            output_dir,
            dataset=shared_dataset,
        )

    if "baselines" in args.tasks:
        evaluate_baseline_detection(
            args.scope,
            args.fold,
            args.batch_size,
            output_dir,
            dataset=shared_dataset,
            model_name=args.model_name,
            dataset_name=args.dataset,
            include_advanced=args.include_advanced,
        )

    if "ablations" in args.tasks:
        evaluate_ablation_summary(
            args.scope,
            args.fold,
            args.batch_size,
            output_dir,
            dataset=shared_dataset,
            model_name=args.model_name,
        )

    if "robustness" in args.tasks:
        evaluate_robustness(args.scope, args.fold, args.batch_size, output_dir)


    manifest["status"] = "ok"
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_manifest(output_dir, manifest)


if __name__ == "__main__":
    main()
