"""Train baseline models.

This is a low-level entry point used by tools/run.py.

Common usage:
    python tools/train_baselines.py
    python tools/train_baselines.py --model FlowTransformer
    python tools/train_baselines.py --model FlowTransformer --fold 2

Plan only:
    python tools/train_baselines.py --dry-run
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stage.baselines.registry import (  # noqa: E402
    CANONICAL_BASELINE_NAMES,
    normalize_baseline_name,
)
from stage.training.baseline_trainer import (  # noqa: E402
    print_baseline_training_plan,
    train_all_baselines,
)
from stage.training.advanced_baseline_trainer import (  # noqa: E402
    ADVANCED_BASELINES,
    print_advanced_training_plan,
    train_advanced_baseline_kfold,
)


BASELINE_GROUP_NAMES = {"all", "baselines"}


def parse_fold(value):
    """Parse fold value: all or one fold id in [0, 4]."""
    value = str(value).strip().lower()

    if value == "all":
        return -1

    try:
        fold = int(value)
    except ValueError as error:
        raise ValueError("--fold must be one of: all, 0, 1, 2, 3, 4") from error

    if fold < 0 or fold > 4:
        raise ValueError("--fold must be one of: all, 0, 1, 2, 3, 4")

    return fold


def resolve_baseline_names(model):
    """Resolve a model selector to canonical baseline names."""
    model = str(model).strip()

    if model.lower() in BASELINE_GROUP_NAMES:
        return list(CANONICAL_BASELINE_NAMES)

    try:
        return [normalize_baseline_name(model)]
    except KeyError as error:
        valid = ", ".join(["all", "baselines", *CANONICAL_BASELINE_NAMES])
        raise ValueError(f"Unknown baseline model: {model}. Valid values: {valid}") from error


def build_parser():
    parser = argparse.ArgumentParser(description="Train STAGE baseline models.")
    parser.add_argument(
        "--model",
        default="all",
        help="Baseline model to train, including DIDS-MFL and GraphIDS.",
    )
    parser.add_argument(
        "--dataset",
        default="ID-APT",
        help="Dataset: ID-APT, DAPT2020, or CIC-IDS-2018+Sandworm.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--fold",
        default="all",
        help="Fold to train: all, 0, 1, 2, 3, or 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training plan without loading data or training.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    fold = parse_fold(args.fold)
    names = resolve_baseline_names(args.model)

    advanced = [name for name in names if name in ADVANCED_BASELINES]
    legacy = [name for name in names if name not in ADVANCED_BASELINES]

    if args.dataset != "ID-APT" and legacy:
        raise ValueError(
            "External-dataset adapters are currently defined for DIDS-MFL and "
            "GraphIDS only. Select one of those models."
        )

    if args.dry_run:
        if legacy:
            print_baseline_training_plan(names=legacy, fold=fold)
        for name in advanced:
            print_advanced_training_plan(name, args.dataset, fold)
        print("dry-run only; baseline training is not started.")
        return

    if legacy:
        train_all_baselines(
            fold=fold,
            names=legacy,
            batch_size=args.batch_size,
        )
    for name in advanced:
        train_advanced_baseline_kfold(
            name=name,
            dataset_name=args.dataset,
            fold=fold,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as error:
        print()
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("[ERROR] Interrupted by user.", file=sys.stderr)
        sys.exit(130)
