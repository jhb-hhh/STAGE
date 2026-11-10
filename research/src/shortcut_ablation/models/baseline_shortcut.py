from copy import copy

import torch.nn as nn

from stage.baselines.ts_ids.model import TS_IDS
from stage.baselines.flow_transformer.model import FlowTransformer
from stage.baselines.et_graphsage.model import ETGraphSAGE

from shortcut_ablation.models.identity_encoder import ShortcutFeatureAdapter


class _ShortcutWrapper(nn.Module):
    def __init__(self, base_model, ip_vocab_size):
        super().__init__()
        self.feature_adapter = ShortcutFeatureAdapter(ip_vocab_size=ip_vocab_size)
        self.base_model = base_model

    def forward(self, data):
        adapted = copy(data)
        adapted.x = self.feature_adapter(data)
        return self.base_model(adapted)


class TSIDSShortcut(_ShortcutWrapper):
    def __init__(self, num_classes, ip_vocab_size):
        adapter = ShortcutFeatureAdapter(ip_vocab_size=ip_vocab_size)
        base = TS_IDS(
            in_channels=adapter.output_dim,
            num_classes=num_classes,
        )
        super().__init__(base, ip_vocab_size=ip_vocab_size)


class FlowTransformerShortcut(_ShortcutWrapper):
    def __init__(self, num_classes, ip_vocab_size):
        adapter = ShortcutFeatureAdapter(ip_vocab_size=ip_vocab_size)
        base = FlowTransformer(
            in_channels=adapter.output_dim,
            num_classes=num_classes,
        )
        super().__init__(base, ip_vocab_size=ip_vocab_size)


class ETGraphSAGEShortcut(_ShortcutWrapper):
    def __init__(self, num_classes, ip_vocab_size):
        adapter = ShortcutFeatureAdapter(ip_vocab_size=ip_vocab_size)
        base = ETGraphSAGE(
            in_channels=adapter.output_dim,
            num_classes=num_classes,
            hidden_dim=128,
            num_gnn_layers=2,
            num_trans_layers=2,
            nhead=4,
            dropout=0.3,
        )
        super().__init__(base, ip_vocab_size=ip_vocab_size)
