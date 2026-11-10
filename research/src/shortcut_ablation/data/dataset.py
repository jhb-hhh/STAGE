import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from shortcut_ablation.config import (
    GRAPH_OUTPUT_DIR,
    SHIFTED_GRAPH_OUTPUT_DIR,
    SPLIT_OUTPUT_DIR,
)


def load_graphs(graph_dir: Path) -> List:
    graph_dir = Path(graph_dir)
    chunks = sorted(graph_dir.glob("graphs_chunk_*.pt"))

    if not chunks:
        raise FileNotFoundError(f"No graph chunks found in {graph_dir}")

    graphs = []
    for chunk_path in chunks:
        graphs.extend(torch.load(chunk_path, map_location="cpu", weights_only=False))

    return graphs


def load_original_graphs() -> List:
    return load_graphs(GRAPH_OUTPUT_DIR)


def load_shifted_graphs() -> List:
    return load_graphs(SHIFTED_GRAPH_OUTPUT_DIR)


def load_split(fold: int) -> Dict:
    split_path = SPLIT_OUTPUT_DIR / f"fold_{fold}.json"

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_split_graphs(fold: int, domain: str = "original") -> Tuple[List, List]:
    if domain == "original":
        graphs = load_original_graphs()
    elif domain == "shifted":
        graphs = load_shifted_graphs()
    else:
        raise ValueError(f"Unknown domain: {domain}. Use 'original' or 'shifted'.")

    split = load_split(fold)
    train_indices = split["train_indices"]
    test_indices = split["test_indices"]

    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]

    return train_graphs, test_graphs


def get_num_classes(graphs: List) -> int:
    if not graphs:
        raise ValueError("Empty graph list.")

    max_label = max(int(g.y.item()) for g in graphs)
    return max_label + 1


def summarize_graphs(graphs: List) -> Dict:
    labels = [int(g.y.item()) for g in graphs]
    label_counts = {}

    for y in labels:
        label_counts[y] = label_counts.get(y, 0) + 1

    return {
        "num_graphs": len(graphs),
        "num_classes": get_num_classes(graphs),
        "label_distribution": label_counts,
        "has_shortcut_fields": all(
            hasattr(g, "src_ip_idx_global")
            and hasattr(g, "dst_ip_idx_global")
            and hasattr(g, "sport_idx_raw")
            and hasattr(g, "dport_idx_raw")
            for g in graphs
        ),
    }
