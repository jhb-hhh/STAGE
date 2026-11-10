"""Checkpoint loading utilities for STAGE models."""

import torch


def adapt_stage_checkpoint_keys(state_dict):
    """Map legacy checkpoint parameter names to current module names."""
    new_state_dict = {}

    for key, value in state_dict.items():
        key = key.replace("module.", "")

        if key.startswith("gate_fc1."):
            new_key = key.replace("gate_fc1.", "gate_module.fc1.")
        elif key.startswith("gate_gnn_delta."):
            new_key = key.replace("gate_gnn_delta.", "gate_module.gnn_delta.")
        elif key.startswith("gate_trans_delta."):
            new_key = key.replace("gate_trans_delta.", "gate_module.trans_delta.")
        elif key.startswith("gate_spatial_delta."):
            new_key = key.replace("gate_spatial_delta.", "gate_module.gnn_delta.")
        elif key.startswith("gate_temporal_delta."):
            new_key = key.replace("gate_temporal_delta.", "gate_module.trans_delta.")
        else:
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict


def load_state_dict(path, map_location="cpu", adapt_keys=True):
    """Load a checkpoint state dictionary from disk."""
    checkpoint = torch.load(path, map_location=map_location)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if adapt_keys:
        checkpoint = adapt_stage_checkpoint_keys(checkpoint)

    return checkpoint


def load_model_checkpoint(model, path, map_location="cpu", strict=False, adapt_keys=True):
    """Load a checkpoint into a model instance."""
    state_dict = load_state_dict(path, map_location=map_location, adapt_keys=adapt_keys)
    return model.load_state_dict(state_dict, strict=strict)
