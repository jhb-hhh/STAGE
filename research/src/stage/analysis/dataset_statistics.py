"""Dataset statistics and split-integrity checks for STAGE."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.data.dataset import IDAPTDataset
from stage.utils.config import get_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GroundTruthManager:
    """Build node-level ground-truth masks from label metadata."""

    def __init__(self, label_file_path=None):
        self.file_index = {}
        self.label_file_path = Path(label_file_path) if label_file_path else get_path("data", "label_file")

        if not self.label_file_path.exists():
            print(f"[WARN] Label file not found: {self.label_file_path}")
            return

        try:
            self.df = pd.read_csv(self.label_file_path)
            self.file_index = self.df.set_index("filename").to_dict("index")
        except Exception as exc:
            print(f"[WARN] Failed to read label file: {exc}")

    @staticmethod
    def _normalize_filename(filename):
        """Normalize a graph filename field to its basename."""
        if isinstance(filename, list):
            filename = filename[0]

        return os.path.basename(filename) if filename else ""

    @staticmethod
    def _parse_malicious_ips(value):
        """Parse malicious IP metadata into a set of strings."""
        value = str(value)

        if not value or value.lower() in {"nan", "n/a", "none", ""}:
            return set()

        value = value.replace("'", "").replace('"', "").replace("[", "").replace("]", "")
        return {ip.strip() for ip in value.split(",") if ip.strip()}

    def get_node_ground_truth(self, data):
        """Return a binary node mask for malicious source or destination IPs."""
        filename = self._normalize_filename(getattr(data, "filename", None))
        record = self.file_index.get(filename)

        if record is None:
            return torch.zeros(data.num_nodes, device=DEVICE)

        malicious_ips = self._parse_malicious_ips(record.get("malicious_ips", ""))
        gt_mask = torch.zeros(data.num_nodes, device=DEVICE)

        if not malicious_ips:
            return gt_mask

        if hasattr(data, "src_ip_list") and hasattr(data, "dst_ip_list"):
            src_list = data.src_ip_list
            dst_list = data.dst_ip_list

            if src_list and isinstance(src_list[0], list):
                src_list = src_list[0]

            if dst_list and isinstance(dst_list[0], list):
                dst_list = dst_list[0]

            for index in range(min(len(src_list), data.num_nodes)):
                if src_list[index] in malicious_ips or dst_list[index] in malicious_ips:
                    gt_mask[index] = 1.0

        return gt_mask


def get_dir_size_gb(folder_path):
    """Return the total size of PCAP files under a directory in GB."""
    folder_path = Path(folder_path)
    total_size = 0

    if not folder_path.exists():
        return 0.0

    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"} and not path.is_symlink():
            total_size += path.stat().st_size

    return total_size / (1024 ** 3)


def summarize_graph_dataset(include_ground_truth=False, batch_size=1, save=True):
    """Summarize graph counts, class distribution, and optional node-level masks."""
    dataset = IDAPTDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    stats = {
        "total_graphs": 0,
        "global_nodes": 0,
        "global_edges": 0,
        "attack_graphs": 0,
        "valid_attack_graphs": 0,
        "nodes_list": [],
        "edges_list": [],
        "malicious_nodes_list": [],
        "sparsity_list": [],
    }

    class_counter = Counter()
    gt_manager = GroundTruthManager() if include_ground_truth else None

    for data in tqdm(loader, desc="Scanning dataset"):
        data = data.to(DEVICE)

        num_nodes = int(data.num_nodes)
        num_edges = int(data.edge_index.size(1)) if hasattr(data, "edge_index") else 0
        y_label = int(data.y.item())

        stats["total_graphs"] += 1
        stats["global_nodes"] += num_nodes
        stats["global_edges"] += num_edges
        class_counter[y_label] += 1

        if y_label != 0:
            stats["attack_graphs"] += 1

        if include_ground_truth and y_label != 0:
            gt_mask = gt_manager.get_node_ground_truth(data)
            num_malicious = int(gt_mask.sum().item())

            if num_malicious > 0:
                stats["valid_attack_graphs"] += 1
                stats["nodes_list"].append(num_nodes)
                stats["edges_list"].append(num_edges)
                stats["malicious_nodes_list"].append(num_malicious)
                stats["sparsity_list"].append(num_malicious / num_nodes if num_nodes > 0 else 0.0)

    total_graphs = stats["total_graphs"]
    summary = {
        "total_graphs": total_graphs,
        "total_nodes": stats["global_nodes"],
        "total_edges": stats["global_edges"],
        "avg_nodes": stats["global_nodes"] / total_graphs if total_graphs else 0,
        "avg_edges": stats["global_edges"] / total_graphs if total_graphs else 0,
        "attack_graphs": stats["attack_graphs"],
        "class_distribution": dict(sorted(class_counter.items())),
    }

    if include_ground_truth:
        summary.update(
            {
                "valid_attack_graphs_with_gt": stats["valid_attack_graphs"],
                "avg_malicious_nodes": (
                    float(np.mean(stats["malicious_nodes_list"]))
                    if stats["malicious_nodes_list"]
                    else 0.0
                ),
                "avg_malicious_sparsity": (
                    float(np.mean(stats["sparsity_list"]))
                    if stats["sparsity_list"]
                    else 0.0
                ),
            }
        )

    if save:
        out_dir = get_path("outputs", "metrics_dir") / "dataset"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "dataset_summary.json"
        distribution_path = out_dir / "class_distribution.csv"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        pd.DataFrame(
            [
                {
                    "class_id": class_id,
                    "count": count,
                    "percentage": count / total_graphs * 100 if total_graphs else 0,
                }
                for class_id, count in sorted(class_counter.items())
            ]
        ).to_csv(distribution_path, index=False)

        print(f"saved: {summary_path}")
        print(f"saved: {distribution_path}")

    return summary


def check_fold_leakage():
    """Check graph-index and filename overlap across five-fold splits."""
    split_dir = get_path("data", "split_dir")

    index_overlap = {}
    split_cache = {}

    for fold_id in range(5):
        path = split_dir / f"fold_{fold_id}.json"

        with open(path, "r", encoding="utf-8") as file:
            split = json.load(file)

        train_idx = set(split["train"])
        val_idx = set(split["val"])
        split_cache[fold_id] = (train_idx, val_idx)
        index_overlap[f"fold_{fold_id}"] = len(train_idx & val_idx)

    dataset = IDAPTDataset()

    idx2file = {}
    for index in range(len(dataset)):
        data = dataset[index]
        filename = data.filename[0] if isinstance(data.filename, list) else data.filename
        idx2file[index] = filename

    filename_overlap = {}
    fold_files = []

    for fold_id in range(5):
        train_idx, val_idx = split_cache[fold_id]
        train_files = {idx2file[index] for index in train_idx}
        val_files = {idx2file[index] for index in val_idx}

        filename_overlap[f"fold_{fold_id}"] = len(train_files & val_files)
        fold_files.append((train_files, val_files))

    file_to_val_folds = defaultdict(set)

    for fold_id, (_, val_files) in enumerate(fold_files):
        for filename in val_files:
            file_to_val_folds[filename].add(fold_id)

    multi_val_files = [
        filename
        for filename, folds in file_to_val_folds.items()
        if len(folds) > 1
    ]

    return {
        "split_granularity": "graph_snapshot_level",
        "index_overlap_train_val": index_overlap,
        "filename_overlap_train_val": filename_overlap,
        "filenames_appearing_in_multiple_val_folds": len(multi_val_files),
        "multi_val_examples": multi_val_files[:10],
        "interpretation": {
            "index_overlap_train_val": "Exact graph index leakage. Expected value: 0.",
            "filename_overlap_train_val": "Same PCAP may produce multiple graph windows. Non-zero values indicate graph-level rather than file-level splitting.",
            "filenames_appearing_in_multiple_val_folds": "A filename can appear in validation of multiple folds if multiple windows from the same PCAP are split independently.",
        },
    }
