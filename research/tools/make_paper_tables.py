import argparse
from pathlib import Path

from stage.analysis.paper_tables import (
    make_all_tables,
    make_table5_detection,
    make_table6_robustness,
    make_table8_ablation,
    make_table9_explainers,
    make_table10_llm,
    set_table_dirs,
)


TABLE_MAKERS = {
    "table5": make_table5_detection,
    "table6": make_table6_robustness,
    "table8": make_table8_ablation,
    "table9": make_table9_explainers,
    "table10": make_table10_llm,
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate paper-ready CSV tables.")
    parser.add_argument(
        "--table",
        type=str,
        default="all",
        choices=["all"] + sorted(TABLE_MAKERS.keys()),
        help="Which table to regenerate.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip table10 when regenerating all tables.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Optional source_metrics directory. Defaults to outputs/metrics/source_metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional paper_tables output directory. Defaults to outputs/metrics/paper_tables.",
    )

    args = parser.parse_args()

    if args.source_dir is not None and not Path(args.source_dir).exists():
        raise FileNotFoundError(f"source-dir not found: {args.source_dir}")

    set_table_dirs(source_dir=args.source_dir, output_dir=args.output_dir)

    if args.table == "all":
        results = make_all_tables(include_llm=not args.skip_llm)
    else:
        results = [TABLE_MAKERS[args.table]()] 

    for path, df in results:
        print(f"saved: {path}")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
