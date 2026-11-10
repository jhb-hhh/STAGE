"""Dataset adapters shared by the advanced baseline reproductions.

The adapter never rewrites source graphs.  It attaches a graph-level target in
memory and stores newly generated external-dataset folds below
``outputs/workspace/splits``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from stage.data.dataset import IDAPTDataset
from stage.utils.config import get_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BaselineDatasetSpec:
    name: str
    graph_dir: Path
    split_dir: Path
    label_map: Path | None


DATASET_ALIASES = {
    "id-apt": "ID-APT",
    "idapt": "ID-APT",
    "ID-APT": "ID-APT",
    "dapt2020": "DAPT2020",
    "dapt-2020": "DAPT2020",
    "DAPT2020": "DAPT2020",
    "cicids2018-sandworm": "CIC-IDS-2018+Sandworm",
    "cic-ids-2018+sandworm": "CIC-IDS-2018+Sandworm",
    "CIC-IDS-2018+Sandworm": "CIC-IDS-2018+Sandworm",
}


def normalize_dataset_name(name):
    value = str(name).strip()
    if value in DATASET_ALIASES:
        return DATASET_ALIASES[value]
    lower = value.lower()
    if lower in DATASET_ALIASES:
        return DATASET_ALIASES[lower]
    valid = ", ".join(("ID-APT", "DAPT2020", "CIC-IDS-2018+Sandworm"))
    raise ValueError(f"Unknown dataset: {name}. Valid datasets: {valid}")


def resolve_dataset_spec(name):
    canonical = normalize_dataset_name(name)
    workspace_splits = PROJECT_ROOT / "outputs" / "workspace" / "splits"

    if canonical == "ID-APT":
        return BaselineDatasetSpec(
            name=canonical,
            graph_dir=Path(get_path("data", "graph_dir")),
            split_dir=Path(get_path("data", "split_dir")),
            label_map=Path(get_path("data", "label_map")),
        )
    if canonical == "DAPT2020":
        return BaselineDatasetSpec(
            name=canonical,
            graph_dir=PROJECT_ROOT / "data" / "external" / "dapt2020" / "graphs_temporal",
            split_dir=workspace_splits / "dapt2020",
            label_map=None,
        )
    return BaselineDatasetSpec(
        name=canonical,
        graph_dir=PROJECT_ROOT / "data" / "external" / "cicids2018_sandworm" / "graphs",
        split_dir=workspace_splits / "cicids2018_sandworm",
        label_map=PROJECT_ROOT
        / "data"
        / "external"
        / "cicids2018_sandworm"
        / "graphs"
        / "label_mapping.json",
    )


def source_graph_target(data):
    """Return the native graph target, collapsing edge labels when necessary."""
    labels = data.y.view(-1).long()
    if labels.numel() == 1:
        return int(labels.item())
    # External converters store one binary label per flow edge.  A window is an
    # attack window when it contains at least one malicious flow.
    return int(torch.any(labels != 0).item())


def binary_graph_target(data):
    return int(source_graph_target(data) != 0)


class GraphTargetDataset(Dataset):
    """Attach a collatable graph_y without modifying serialized artifacts."""

    def __init__(self, dataset, task="native", indices=None):
        self.dataset = dataset
        self.task = task
        self.indices = list(range(len(dataset))) if indices is None else [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        data = self.dataset[self.indices[item]].clone()
        target_fn = binary_graph_target if self.task == "binary" else source_graph_target
        data.graph_y = torch.tensor([target_fn(data)], dtype=torch.long)
        return data


def load_baseline_dataset(name):
    spec = resolve_dataset_spec(name)
    if not spec.graph_dir.exists() or not list(spec.graph_dir.glob("graphs_chunk_*.pt")):
        if spec.name == "CIC-IDS-2018+Sandworm":
            hint = (
                "Provide converted graph chunks at "
                "data/external/cicids2018_sandworm/graphs before training."
            )
        elif spec.name == "DAPT2020":
            hint = "Run tools/convert_dapt2020.py to build temporal graph windows first."
        else:
            hint = "Run python tools/run.py rebuild graphs first."
        raise FileNotFoundError(f"Missing graph chunks: {spec.graph_dir}\n{hint}")
    return spec, IDAPTDataset(root_dir=spec.graph_dir)


def load_or_create_splits(spec, dataset, num_folds=5, seed=42):
    split_paths = [spec.split_dir / f"fold_{fold}.json" for fold in range(num_folds)]
    if all(path.is_file() for path in split_paths):
        return split_paths

    # ID-APT folds are an experimental artifact and must never be silently
    # replaced. External folds are derived using the same stratified protocol.
    if spec.name == "ID-APT":
        raise FileNotFoundError(
            f"Missing ID-APT five-fold splits in {spec.split_dir}. "
            "Run python tools/run.py rebuild splits first."
        )

    labels = np.asarray([binary_graph_target(dataset[i]) for i in range(len(dataset))])
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < num_folds:
        raise ValueError(
            f"{spec.name} cannot form {num_folds} stratified folds; "
            f"binary label counts are {dict(zip(unique.tolist(), counts.tolist()))}."
        )

    spec.split_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(labels))
    for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, labels)):
        payload = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "dataset": spec.name,
            "task": "binary_graph_detection",
            "seed": seed,
        }
        with open(spec.split_dir / f"fold_{fold}.json", "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
    return split_paths


def load_baseline_split(spec, dataset, fold_id):
    paths = load_or_create_splits(spec, dataset)
    with open(paths[int(fold_id)], "r", encoding="utf-8") as file:
        return json.load(file)


def infer_dimensions(dataset):
    sample = dataset[0]
    return int(sample.x.size(-1)), int(sample.edge_attr.size(-1))


def native_num_classes(spec, dataset):
    if spec.label_map is not None and spec.label_map.is_file():
        with open(spec.label_map, "r", encoding="utf-8") as file:
            return len(json.load(file))
    return max(source_graph_target(dataset[i]) for i in range(len(dataset))) + 1
