"""Baseline model registry."""

from stage.baselines.et_graphsage.model import ETGraphSAGE
from stage.baselines.flow_transformer.model import FlowTransformer
from stage.baselines.dids_mfl.model import DIDSMFL
from stage.baselines.graphids.model import GraphIDS
from stage.baselines.ts_ids.model import TS_IDS

LEGACY_BASELINE_NAMES = [
    "TS-IDS",
    "FlowTransformer",
    "E-T-GraphSAGE",
]

CANONICAL_BASELINE_NAMES = [
    *LEGACY_BASELINE_NAMES,
    "DIDS-MFL",
    "GraphIDS",
]

BASELINE_ALIASES = {
    "TS-IDS": "TS-IDS",
    "ts_ids": "TS-IDS",
    "FlowTransformer": "FlowTransformer",
    "flow_transformer": "FlowTransformer",
    "E-T-GraphSAGE": "E-T-GraphSAGE",
    "et_graphsage": "E-T-GraphSAGE",
    "DIDS-MFL": "DIDS-MFL",
    "dids-mfl": "DIDS-MFL",
    "dids_mfl": "DIDS-MFL",
    "GraphIDS": "GraphIDS",
    "graphids": "GraphIDS",
}

BASELINE_REGISTRY = {
    "TS-IDS": TS_IDS,
    "FlowTransformer": FlowTransformer,
    "E-T-GraphSAGE": ETGraphSAGE,
    "DIDS-MFL": DIDSMFL,
    "GraphIDS": GraphIDS,
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


def build_baseline(name, num_classes, in_channels=8, edge_dim=2):
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

    if canonical_name == "DIDS-MFL":
        return DIDSMFL(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=128,
            diffusion_alpha=0.02,
            dropout_implicit=0.5,
            dropout_explicit=0.0,
        )

    if canonical_name == "GraphIDS":
        return GraphIDS(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_dim=64,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.5,
            ae_dropout=0.2,
            mask_ratio=0.15,
        )

    cls = BASELINE_REGISTRY[canonical_name]
    return cls(in_channels=in_channels, num_classes=num_classes)
