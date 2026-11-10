"""Baseline model registry."""

from stage.baselines.et_graphsage.model import ETGraphSAGE
from stage.baselines.flow_transformer.model import FlowTransformer
from stage.baselines.ts_ids.model import TS_IDS

CANONICAL_BASELINE_NAMES = [
    "TS-IDS",
    "FlowTransformer",
    "E-T-GraphSAGE",
]

BASELINE_ALIASES = {
    "TS-IDS": "TS-IDS",
    "ts_ids": "TS-IDS",
    "FlowTransformer": "FlowTransformer",
    "flow_transformer": "FlowTransformer",
    "E-T-GraphSAGE": "E-T-GraphSAGE",
    "et_graphsage": "E-T-GraphSAGE",
}

BASELINE_REGISTRY = {
    "TS-IDS": TS_IDS,
    "FlowTransformer": FlowTransformer,
    "E-T-GraphSAGE": ETGraphSAGE,
}


def normalize_baseline_name(name):
    """Normalize a baseline alias to its canonical name."""
    if name not in BASELINE_ALIASES:
        available = ", ".join(sorted(BASELINE_ALIASES.keys()))
        raise ValueError(f"Unknown baseline: {name}. Available: {available}")

    return BASELINE_ALIASES[name]


def get_baseline_class(name):
    """Return a baseline class by name or alias."""
    canonical_name = normalize_baseline_name(name)
    return BASELINE_REGISTRY[canonical_name]


def build_baseline(name, num_classes, in_channels=8):
    """Build a baseline model by name or alias."""
    canonical_name = normalize_baseline_name(name)

    if canonical_name == "E-T-GraphSAGE":
        return ETGraphSAGE(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=128,
            num_gnn_layers=2,
            num_trans_layers=2,
            nhead=4,
            dropout=0.3,
        )

    cls = BASELINE_REGISTRY[canonical_name]
    return cls(in_channels=in_channels, num_classes=num_classes)
