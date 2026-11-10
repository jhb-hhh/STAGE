"""Command-line entry point for STAGE explanation tasks."""

import argparse
import glob
import shutil
from pathlib import Path

import torch

from stage.explanation import benchmark, sensitivity
from stage.explanation.endogenous import EndogenousExplainer
from stage.utils.config import get_path


def parse_k_list(value):
    """Parse comma-separated K values for sensitivity analysis."""
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def run_endogenous(args):
    """Run the endogenous STAGE explainer on selected graph snapshots."""
    graph_dir = get_path("data", "graph_dir")
    chunk_files = sorted(glob.glob(str(Path(graph_dir) / "graphs_chunk_*.pt")))

    if not chunk_files:
        raise FileNotFoundError(f"No graph chunks found in {graph_dir}")

    if args.chunk >= len(chunk_files):
        raise ValueError(f"chunk id {args.chunk} out of range. Total chunks: {len(chunk_files)}")

    graphs = torch.load(chunk_files[args.chunk], weights_only=False)
    graphs = graphs[:args.limit]

    explainer = EndogenousExplainer(load_dataset=False)

    print("Endogenous explanation plan")
    print("chunk:", chunk_files[args.chunk])
    print("num_graphs:", len(graphs))
    print("top_k:", args.top_k)
    print("output_dir:", explainer.output_dir)

    for i, data in enumerate(graphs):
        top_k_indices, scores, valid_len, bias_ratio = explainer.explain(data)
        top_k_indices = top_k_indices[:args.top_k]

        filename = getattr(data, "filename", f"graph_{i}")
        print("=" * 80)
        print("index:", i)
        print("filename:", filename)
        print("valid_len:", valid_len)
        print("top_k_indices:", top_k_indices.tolist() if hasattr(top_k_indices, "tolist") else top_k_indices)
        print("score_min:", float(scores.min()))
        print("score_max:", float(scores.max()))
        print("bias_ratio:", float(bias_ratio))

        if args.save_report:
            confidence = None
            try:
                report = explainer.generate_report(data, top_k_indices, scores, confidence, bias_ratio)
                if isinstance(report, dict):
                    print("report generated:", report.get("meta", {}).get("filename", filename))
                else:
                    print("report generated:", type(report))
            except Exception as e:
                print("report generation failed:", e)


def run_benchmark(args):
    if args.results_dir is not None:
        benchmark.RESULTS_DIR = args.results_dir
        Path(benchmark.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print("Explanation benchmark plan")
    print("model:", benchmark.EvalConfig.MODEL_PATH)
    print("pgexplainer:", benchmark.EvalConfig.PGEXPLAINER_WEIGHT_PATH)
    print("results_dir:", benchmark.RESULTS_DIR)
    print("num_samples:", args.num_samples or benchmark.EvalConfig.NUM_SAMPLES)
    print("top_k:", args.top_k or benchmark.EvalConfig.TOP_K)
    print("source_metrics_dir:", args.source_metrics_dir)

    if not args.run:
        print("dry-run only; benchmark is not executed.")
        return

    if args.num_samples is not None:
        benchmark.EvalConfig.NUM_SAMPLES = args.num_samples
    if args.top_k is not None:
        benchmark.EvalConfig.TOP_K = args.top_k

    benchmark.main()

    if args.source_metrics_dir is not None:
        src = Path(benchmark.RESULTS_DIR) / "final_benchmark_result_top5.csv"
        dst_dir = Path(args.source_metrics_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "explanation_benchmark.csv"

        if not src.exists():
            raise FileNotFoundError(f"benchmark result not found: {src}")

        shutil.copy2(src, dst)
        print(f"source metric saved: {dst}")

def run_sensitivity(args):
    k_list = parse_k_list(args.k_list)

    if args.results_dir is not None:
        sensitivity.RESULTS_DIR = args.results_dir
        Path(sensitivity.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print("Sensitivity analysis plan")
    print("model:", sensitivity.SenseConfig.MODEL_PATH)
    print("pgexplainer:", sensitivity.SenseConfig.PGEXPLAINER_WEIGHT_PATH)
    print("results_dir:", sensitivity.RESULTS_DIR)
    print("num_samples:", args.num_samples or sensitivity.SenseConfig.NUM_SAMPLES)
    print("k_list:", k_list or sensitivity.SenseConfig.K_LIST)

    if not args.run:
        print("dry-run only; sensitivity analysis is not executed.")
        return

    if args.num_samples is not None:
        sensitivity.SenseConfig.NUM_SAMPLES = args.num_samples
    if k_list is not None:
        sensitivity.SenseConfig.K_LIST = k_list

    sensitivity.main()

def main():
    """Parse command-line arguments and dispatch the selected explanation task."""
    parser = argparse.ArgumentParser(description="Run STAGE explanation tasks.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["endogenous", "benchmark", "sensitivity"],
        help="Explanation task.",
    )

    parser.add_argument("--chunk", type=int, default=0, help="Graph chunk id for endogenous explanation.")
    parser.add_argument("--limit", type=int, default=5, help="Number of graphs for endogenous explanation.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence nodes.")
    parser.add_argument("--save-report", action="store_true", help="Generate JSON reports for endogenous explanation.")

    parser.add_argument("--run", action="store_true", help="Actually run benchmark or sensitivity tasks.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--k-list", type=str, default=None, help="Comma-separated K values for sensitivity analysis.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional result directory for benchmark or sensitivity outputs.",
    )
    parser.add_argument(
        "--source-metrics-dir",
        type=str,
        default=None,
        help="Optional source_metrics directory for exporting explanation_benchmark.csv.",
    )

    args = parser.parse_args()

    if args.task == "endogenous":
        run_endogenous(args)
    elif args.task == "benchmark":
        run_benchmark(args)
    elif args.task == "sensitivity":
        run_sensitivity(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
