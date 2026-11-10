"""Shared helpers for STAGE evaluation modules."""

import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from stage.utils.config import get_path


def resolve_device(device=None):
    """Resolve the evaluation device."""
    if device is not None:
        return device

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path):
    """Load a JSON file."""
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_label_map():
    """Load the dataset label mapping."""
    return load_json(get_path("data", "label_map"))


def load_split_indices(fold_id):
    """Load train and validation indices for one fold."""
    split_path = get_path("data", "split_dir") / f"fold_{fold_id}.json"
    return load_json(split_path)


def make_data_loader(dataset, batch_size, shuffle=False, num_workers=0):
    """Create a PyG data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
