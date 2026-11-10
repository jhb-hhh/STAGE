"""Build the omitted de-identified ID-APT shifted fold 1 in workspace only.

The preserved generator targeted folds 0,2,3,4. We use its perturbation rule,
but seed the new fold explicitly; no reference artifact is ever modified.
"""

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch

from stage.evaluation.robustness import perturb_graph


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "reference" / "graphs"
DEST = ROOT / "outputs" / "workspace" / "graphs_shifted_generated" / "fold_1"
SPLIT = ROOT / "outputs" / "workspace" / "splits" / "fold_1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if DEST.exists() and list(DEST.glob("*.pt")):
        raise FileExistsError(f"Fold 1 already generated; refusing to overwrite: {DEST}")
    with SPLIT.open(encoding="utf-8") as handle:
        val = set(json.load(handle)["val"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    DEST.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(SOURCE / "graphs_chunk_*.pt")))
    count, buffer = 0, []
    for chunk in files:
        graphs = torch.load(chunk, map_location="cpu", weights_only=False)
        for graph in graphs:
            if count in val:
                buffer.append(perturb_graph(graph.clone()))
            count += 1
        if len(buffer) >= 500:
            out = DEST / f"shifted_chunk_{count}.pt"
            torch.save(buffer, out)
            print("saved", out, "graphs", len(buffer), flush=True)
            buffer.clear()
    if buffer:
        out = DEST / "shifted_chunk_final.pt"
        torch.save(buffer, out)
        print("saved", out, "graphs", len(buffer), flush=True)
    with (DEST / "generation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"fold": 1, "seed": args.seed, "source": str(SOURCE),
                   "original_generator": "src_v2_core/tools/generate_shifted_multifold.py",
                   "graphs_seen": count, "expected_validation_graphs": len(val)},
                  handle, indent=2)
    print("completed fold_1", count, "total graphs; expected validation", len(val))


if __name__ == "__main__":
    main()
