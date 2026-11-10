"""Measure prediction-plus-explanation overhead without changing paper artifacts.

The STAGE row reuses signals from the classification forward. Other rows run
the same classification forward before invoking their existing explainer.
Forward counts exclude sample selection, initialization, and quality metrics.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from stage.data.dataset import IDAPTDataset
from stage.explanation import benchmark
from stage.models.stage import StageModel
from stage.training.checkpoint import load_model_checkpoint


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("STAGE", "GNNExplainer", "PGExplainer", "SHAP", "LIME")


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stage_scores(internals):
    """Match EndogenousExplainer.explain's node-score calculation."""
    p = internals["final_attn"].reshape(-1).detach().cpu().numpy()
    spatial = internals["gate_spa"].detach().cpu().numpy().mean(axis=1)
    temporal = internals["gate_tem"].detach().cpu().numpy().mean(axis=1)
    return (spatial + temporal) * np.log1p(p)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/workspace/metrics/benchmarks/computational_overhead.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/reference/checkpoints/stage/best_model_fold_2.pth")
    parser.add_argument("--pg-checkpoint", type=Path, default=ROOT / "outputs/reference/checkpoints/explainers/pgexplainer_weights.pt")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    return args


def main():
    args = parse_args()
    benchmark.set_seed(42)
    label_map = json.loads((ROOT / "data/reference/graphs/label_mapping.json").read_text(encoding="utf-8"))
    device = torch.device(benchmark.DEVICE)
    model = StageModel(num_classes=len(label_map)).to(device).eval()
    load_model_checkpoint(model, args.checkpoint, map_location=device, strict=True)

    explainers = {}
    if "GNNExplainer" in args.methods:
        explainers["GNNExplainer"] = benchmark.GNNExplainerWrapper(model)
    if "PGExplainer" in args.methods:
        if not args.pg_checkpoint.is_file():
            raise FileNotFoundError(args.pg_checkpoint)
        explainers["PGExplainer"] = benchmark.PGExplainerWrapper(model, str(args.pg_checkpoint))
    if "SHAP" in args.methods:
        if benchmark.shap is None:
            raise ImportError("SHAP is not installed")
        explainers["SHAP"] = benchmark.ShapExplainerWrapper(model)
    if "LIME" in args.methods:
        if benchmark.lime_tabular is None:
            raise ImportError("LIME is not installed")
        explainers["LIME"] = benchmark.LimeExplainerWrapper(model)

    dataset = IDAPTDataset(root_dir=ROOT / "data/reference/graphs")
    split = json.loads((ROOT / "data/reference/splits/fold_2.json").read_text(encoding="utf-8"))
    loader = DataLoader(Subset(dataset, split["val"]), batch_size=1, shuffle=True)
    ground_truth = benchmark.GroundTruthManager(ROOT / "data/processed/labels/labels.csv")
    records = []
    selected = 0

    for data in loader:
        if selected >= args.samples:
            break
        data = data.to(device)
        benchmark.clear_pyg_state(model)
        with torch.no_grad():
            predicted = model(data).argmax(dim=1).item()
        if data.y.item() == 0 or predicted != data.y.item():
            continue
        if ground_truth.get_node_ground_truth(data).sum().item() == 0:
            continue
        selected += 1

        for method in args.methods:
            benchmark.clear_pyg_state(model)
            model.zero_grad(set_to_none=True)
            gc.collect()
            synchronize()
            baseline_mb = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            calls = [0]

            def count_forward(_module, _inputs, _output):
                calls[0] += 1

            hook = model.register_forward_hook(count_forward)
            try:
                start = time.perf_counter()
                if method == "STAGE":
                    with torch.no_grad():
                        _, internals = model(data, return_explanation=True)
                        scores = stage_scores(internals)
                    del internals
                else:
                    # Shared prediction cost is included for every method.
                    with torch.no_grad():
                        model(data)
                    scores = explainers[method].explain(data, target=data.y)
                synchronize()
                elapsed = time.perf_counter() - start
                peak_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
            finally:
                hook.remove()
                benchmark.clear_pyg_state(model)
            if len(scores) != data.num_nodes:
                raise ValueError(f"{method}: returned {len(scores)} scores for {data.num_nodes} nodes")
            if calls[0] < 1:
                raise RuntimeError(f"{method}: no model forward was counted")
            records.append({
                "sample": selected,
                "method": method,
                "time_seconds": elapsed,
                "model_evaluations_total": calls[0],
                "additional_model_evaluations": calls[0] - 1,
                "baseline_gpu_memory_mb": baseline_mb,
                "peak_gpu_memory_mb": peak_mb,
                "incremental_peak_gpu_memory_mb": peak_mb - baseline_mb if peak_mb is not None else None,
                "nodes": data.num_nodes,
                "edges": data.edge_index.size(1),
            })
            print(f"sample={selected} method={method} time={elapsed:.4f}s total_forward={calls[0]} extra={calls[0]-1} peak_gpu_mb={peak_mb}", flush=True)
            del scores

    if selected != args.samples:
        raise RuntimeError(f"Only {selected} eligible samples found, requested {args.samples}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"Saved {args.output}")
    print(pd.DataFrame(records).groupby("method")[["time_seconds", "model_evaluations_total", "additional_model_evaluations", "peak_gpu_memory_mb", "incremental_peak_gpu_memory_mb"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
