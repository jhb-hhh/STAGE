"""Generate five-fold stratified splits for workspace graph artifacts.

This script is intentionally simple and opinionated:
- it only serves the rebuild workspace;
- it reads graphs from outputs/workspace/graphs;
- it writes splits to outputs/workspace/splits;
- it never falls back to reference graphs or reference splits.

Run:
    python tools/make_splits.py
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stage.data.dataset import IDAPTDataset  # noqa: E402


GRAPH_DIR = PROJECT_ROOT / "outputs" / "workspace" / "graphs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "workspace" / "splits"

NUM_FOLDS = 5
SEED = 42
SHUFFLE = True

GRAPH_CHUNK_PATTERN = "graphs_chunk_*.pt"


def _relative_or_absolute(path):
    """Return a project-relative path string when possible."""
    path = Path(path).resolve()

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _count_labels(labels):
    """Return a stable JSON-serializable label distribution."""
    counter = Counter(int(label) for label in labels)
    return {
        str(label): int(counter[label])
        for label in sorted(counter)
    }


def _check_workspace_graphs():
    """Ensure workspace graph chunks exist before loading the dataset."""
    if not GRAPH_DIR.exists():
        raise FileNotFoundError(
            f"Workspace graph directory does not exist: {GRAPH_DIR}\n"
            "Please run first:\n"
            "  python tools/run.py rebuild graphs"
        )

    chunk_files = sorted(GRAPH_DIR.glob(GRAPH_CHUNK_PATTERN))

    if not chunk_files:
        raise FileNotFoundError(
            f"No {GRAPH_CHUNK_PATTERN} files found in {GRAPH_DIR}\n"
            "Please run first:\n"
            "  python tools/run.py rebuild graphs"
        )

    return chunk_files


def _load_labels():
    """Load workspace graphs and extract graph-level labels."""
    print(f"Loading workspace graphs from: {GRAPH_DIR}")
    dataset = IDAPTDataset(root_dir=GRAPH_DIR)

    labels = []

    for index in range(len(dataset)):
        data = dataset[index]

        if not hasattr(data, "y"):
            raise AttributeError(f"Graph at index {index} does not contain attribute 'y'.")

        labels.append(int(data.y.item()))

    if not labels:
        raise ValueError(f"No graphs loaded from {GRAPH_DIR}")

    return np.asarray(labels, dtype=np.int64)


def _validate_stratification(labels):
    """Fail early if any class has too few samples for five-fold splitting."""
    distribution = Counter(int(label) for label in labels)
    too_small = {
        label: count
        for label, count in distribution.items()
        if count < NUM_FOLDS
    }

    if too_small:
        raise ValueError(
            "StratifiedKFold cannot create "
            f"{NUM_FOLDS} folds because some classes have fewer samples: "
            f"{too_small}"
        )


def _write_json(path, data):
    """Write pretty JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    """Generate fold_0.json to fold_4.json and split_metadata.json."""
    chunk_files = _check_workspace_graphs()
    labels = _load_labels()
    _validate_stratification(labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=SHUFFLE,
        random_state=SEED,
    )

    fold_label_distribution = {}
    graph_indices = np.arange(len(labels))

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(graph_indices, labels)):
        split = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        }

        split_path = OUTPUT_DIR / f"fold_{fold_id}.json"
        _write_json(split_path, split)

        fold_label_distribution[f"fold_{fold_id}"] = {
            "train": _count_labels(labels[train_idx]),
            "val": _count_labels(labels[val_idx]),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
        }

        print(
            f"[WRITE] {split_path} "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )

    metadata = {
        "num_graphs": int(len(labels)),
        "num_folds": NUM_FOLDS,
        "seed": SEED,
        "shuffle": SHUFFLE,
        "label_distribution": _count_labels(labels),
        "fold_label_distribution": fold_label_distribution,
        "graph_dir": _relative_or_absolute(GRAPH_DIR),
        "output_dir": _relative_or_absolute(OUTPUT_DIR),
        "graph_chunk_pattern": GRAPH_CHUNK_PATTERN,
        "num_graph_chunks": int(len(chunk_files)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = OUTPUT_DIR / "split_metadata.json"
    _write_json(metadata_path, metadata)
    print(f"[WRITE] {metadata_path}")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, AttributeError) as error:
        print()
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("[ERROR] Interrupted by user.", file=sys.stderr)
        sys.exit(130)
