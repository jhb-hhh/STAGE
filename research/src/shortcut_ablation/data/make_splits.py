import json
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import KFold, StratifiedKFold

from shortcut_ablation.config import GRAPH_OUTPUT_DIR, SPLIT_OUTPUT_DIR


def _load_graph_labels():
    chunks = sorted(Path(GRAPH_OUTPUT_DIR).glob("graphs_chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No graph chunks found in {GRAPH_OUTPUT_DIR}")

    labels = []
    records = []
    global_index = 0

    for chunk_id, chunk_path in enumerate(chunks):
        graphs = torch.load(chunk_path, map_location="cpu", weights_only=False)

        for local_index, data in enumerate(graphs):
            y = int(data.y.item())
            labels.append(y)
            records.append(
                {
                    "global_index": global_index,
                    "chunk_id": chunk_id,
                    "chunk_file": chunk_path.name,
                    "local_index": local_index,
                    "label": y,
                    "attack_type_str": getattr(data, "attack_type_str", "Unknown"),
                    "filename": getattr(data, "filename", "Unknown"),
                    "window_index": int(getattr(data, "window_index", -1)),
                }
            )
            global_index += 1

    return labels, records


def _write_json(path, obj):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_make_splits(n_splits=5, seed=2026, reset=True):
    labels, records = _load_graph_labels()
    n_samples = len(labels)

    if n_samples < 2:
        raise RuntimeError("Need at least 2 graph samples to create splits.")

    if reset and SPLIT_OUTPUT_DIR.exists():
        shutil.rmtree(SPLIT_OUTPUT_DIR)
    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_counts = Counter(labels)
    min_class_count = min(label_counts.values())

    requested_n_splits = int(n_splits)

    if min_class_count >= requested_n_splits:
        splitter = StratifiedKFold(
            n_splits=requested_n_splits,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(list(range(n_samples)), labels)
        actual_n_splits = requested_n_splits
        strategy = "StratifiedKFold"
    else:
        # 小样本功能测试时，某些类别数量不足，不能做严格 StratifiedKFold。
        # 正式实验数据量足够时会自动使用 StratifiedKFold。
        actual_n_splits = max(2, min(requested_n_splits, n_samples))
        splitter = KFold(
            n_splits=actual_n_splits,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(list(range(n_samples)))
        strategy = "KFold_fallback_for_small_test_set"

    fold_summaries = []

    for fold_id, split in enumerate(split_iter, start=1):
        if strategy == "StratifiedKFold":
            train_idx, test_idx = split
        else:
            train_idx, test_idx = split

        train_idx = [int(i) for i in train_idx]
        test_idx = [int(i) for i in test_idx]

        fold_obj = {
            "fold_id": fold_id,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "train_label_distribution": dict(Counter(labels[i] for i in train_idx)),
            "test_label_distribution": dict(Counter(labels[i] for i in test_idx)),
        }

        fold_path = SPLIT_OUTPUT_DIR / f"fold_{fold_id}.json"
        _write_json(fold_path, fold_obj)

        fold_summaries.append(
            {
                "fold_id": fold_id,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "fold_file": fold_path.name,
                "train_label_distribution": fold_obj["train_label_distribution"],
                "test_label_distribution": fold_obj["test_label_distribution"],
            }
        )

    metadata = {
        "graph_dir": str(GRAPH_OUTPUT_DIR),
        "split_dir": str(SPLIT_OUTPUT_DIR),
        "num_graphs": n_samples,
        "requested_n_splits": requested_n_splits,
        "actual_n_splits": actual_n_splits,
        "strategy": strategy,
        "seed": seed,
        "label_distribution": dict(label_counts),
        "folds": fold_summaries,
    }

    _write_json(SPLIT_OUTPUT_DIR / "split_metadata.json", metadata)
    _write_json(SPLIT_OUTPUT_DIR / "graph_index.json", records)

    print(f"[OK] Splits saved to: {SPLIT_OUTPUT_DIR}")
    print(f"[INFO] graphs: {n_samples}")
    print(f"[INFO] requested_n_splits: {requested_n_splits}")
    print(f"[INFO] actual_n_splits: {actual_n_splits}")
    print(f"[INFO] strategy: {strategy}")
    print(f"[INFO] label_distribution: {dict(label_counts)}")
