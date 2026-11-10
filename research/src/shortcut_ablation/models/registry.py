import json
from pathlib import Path

from shortcut_ablation.config import OUTPUT_ROOT
from shortcut_ablation.models.stage_shortcut import StageShortcutModel
from shortcut_ablation.models.baseline_shortcut import (
    TSIDSShortcut,
    FlowTransformerShortcut,
    ETGraphSAGEShortcut,
)


SHORTCUT_MODEL_REGISTRY = {
    "STAGE": StageShortcutModel,
    "TS-IDS": TSIDSShortcut,
    "FlowTransformer": FlowTransformerShortcut,
    "E-T-GraphSAGE": ETGraphSAGEShortcut,
}

MODEL_ALIASES = {
    "stage": "STAGE",
    "STAGE": "STAGE",
    "ts_ids": "TS-IDS",
    "TS-IDS": "TS-IDS",
    "flow_transformer": "FlowTransformer",
    "FlowTransformer": "FlowTransformer",
    "et_graphsage": "E-T-GraphSAGE",
    "E-T-GraphSAGE": "E-T-GraphSAGE",
}


def normalize_model_name(name):
    if name not in MODEL_ALIASES:
        valid = ", ".join(sorted(MODEL_ALIASES.keys()))
        raise ValueError(f"Unknown shortcut model: {name}. Valid: {valid}")
    return MODEL_ALIASES[name]


def load_ip_vocab_size():
    ip_vocab_path = OUTPUT_ROOT / "meta" / "ip_vocab.json"
    if not ip_vocab_path.exists():
        raise FileNotFoundError(f"Missing IP vocab: {ip_vocab_path}")

    with ip_vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    return max(int(v) for v in vocab.values()) + 1


def build_shortcut_model(name, num_classes):
    canonical = normalize_model_name(name)
    cls = SHORTCUT_MODEL_REGISTRY[canonical]
    ip_vocab_size = load_ip_vocab_size()

    return cls(
        num_classes=num_classes,
        ip_vocab_size=ip_vocab_size,
    )


def list_shortcut_models():
    return list(SHORTCUT_MODEL_REGISTRY.keys())
