"""Training entry point for STAGE and ablation variants."""

import argparse
from pathlib import Path

from stage.models.variants import MODEL_REGISTRY
from stage.training.trainer import normalize_mode_name, train_kfold
from stage.utils.config import get_path


def build_parser():
    """Build the command-line parser."""
    available_modes = sorted(MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="Train STAGE or ablation variants.")
    parser.add_argument("--mode", type=str, required=True, choices=available_modes)
    parser.add_argument("--fold", type=int, default=-1, help="Use -1 for all five folds.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Default: outputs/checkpoints/training_runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print training plan and paths without loading the full dataset.",
    )

    return parser


def main():
    """Print the training plan and optionally start training."""
    parser = build_parser()
    args = parser.parse_args()
    mode_name = normalize_mode_name(args.mode)

    if args.output_root is None:
        output_root = get_path("outputs", "checkpoint_dir") / "training_runs"
    else:
        output_root = Path(args.output_root)

    save_dir = output_root / f"kfold_experiment_{mode_name}"

    print("Training plan")
    print("mode:", mode_name)
    print("fold:", args.fold)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("output_dir:", save_dir)
    print("note: reference paper checkpoints are not overwritten.")

    if args.dry_run:
        print("dry-run only; training is not started.")
        return

    train_kfold(
        mode_name=mode_name,
        fold=args.fold,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
