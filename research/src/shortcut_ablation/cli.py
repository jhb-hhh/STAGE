import argparse

from shortcut_ablation.config import DEFAULT_BENIGN_RATIO, DEFAULT_TOTAL_FILES
from shortcut_ablation.data.synthesize_pcaps import run_synthesis
from shortcut_ablation.data.build_graphs import run_build_graphs
from shortcut_ablation.data.build_shifted_graphs import run_build_shifted_graphs
from shortcut_ablation.data.make_splits import run_make_splits
from shortcut_ablation.data.dataset import load_original_graphs, load_shifted_graphs, get_split_graphs, summarize_graphs
from shortcut_ablation.training.trainer import train_shortcut_models
from shortcut_ablation.evaluation.robustness import run_robustness


def main():
    parser = argparse.ArgumentParser(description="Shortcut-ablation experiment utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser(
        "synthesize",
        help="Generate shortcut-exposed PCAPs without dynamic IP remapping.",
    )
    synth.add_argument("--max-files", type=int, default=DEFAULT_TOTAL_FILES)
    synth.add_argument("--benign-ratio", type=float, default=DEFAULT_BENIGN_RATIO)
    synth.add_argument("--seed", type=int, default=2026)
    synth.add_argument("--reset", action="store_true")

    build = subparsers.add_parser(
        "build-graphs",
        help="Build shortcut-exposed graphs from shortcut PCAPs.",
    )
    build.add_argument("--no-reset", action="store_true")

    shifted = subparsers.add_parser(
        "build-shifted-graphs",
        help="Build shifted shortcut graphs with topology-preserving IP reassignment, port reassignment, and feature noise.",
    )
    shifted.add_argument("--seed", type=int, default=2026)
    shifted.add_argument("--noise-std", type=float, default=0.05)
    shifted.add_argument("--no-reset", action="store_true")

    splits = subparsers.add_parser(
        "make-splits",
        help="Create train/test splits for shortcut-ablation graphs.",
    )
    splits.add_argument("--n-splits", type=int, default=5)
    splits.add_argument("--seed", type=int, default=2026)
    splits.add_argument("--no-reset", action="store_true")

    inspect = subparsers.add_parser(
        "inspect-data",
        help="Inspect shortcut-ablation original/shifted graphs and one split.",
    )
    inspect.add_argument("--fold", type=int, default=1)

    train = subparsers.add_parser(
        "train",
        help="Train shortcut-visible models on original shortcut graphs.",
    )
    train.add_argument("--model", default="all")
    train.add_argument("--fold", default="1")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--weight-decay", type=float, default=None)

    robustness = subparsers.add_parser(
        "robustness",
        help="Evaluate shortcut models on original and shifted shortcut graphs.",
    )
    robustness.add_argument("--model", default="all")
    robustness.add_argument("--fold", default="1")
    robustness.add_argument("--batch-size", type=int, default=2)

    args = parser.parse_args()

    if args.command == "synthesize":
        run_synthesis(
            max_files=args.max_files,
            benign_ratio=args.benign_ratio,
            seed=args.seed,
            reset=args.reset,
        )
    elif args.command == "build-graphs":
        run_build_graphs(reset=not args.no_reset)
    elif args.command == "build-shifted-graphs":
        run_build_shifted_graphs(
            seed=args.seed,
            noise_std=args.noise_std,
            reset=not args.no_reset,
        )
    elif args.command == "make-splits":
        run_make_splits(
            n_splits=args.n_splits,
            seed=args.seed,
            reset=not args.no_reset,
        )
    elif args.command == "inspect-data":
        original = load_original_graphs()
        shifted = load_shifted_graphs()
        train_graphs, test_graphs = get_split_graphs(args.fold, domain="original")

        print("[Original]", summarize_graphs(original))
        print("[Shifted]", summarize_graphs(shifted))
        print(f"[Fold {args.fold} train]", summarize_graphs(train_graphs))
        print(f"[Fold {args.fold} test]", summarize_graphs(test_graphs))
    elif args.command == "train":
        train_shortcut_models(
            model=args.model,
            fold=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.command == "robustness":
        run_robustness(
            model=args.model,
            fold=args.fold,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
