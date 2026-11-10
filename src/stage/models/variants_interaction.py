"""Interaction-structure ablation variants."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from stage.models.ablation_base import StageAblationBaseModel


class Serial_GCN_LSTM_Model(StageAblationBaseModel):
    """Serial GCN-to-LSTM variant without RVG fusion."""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        del self.gate_module
        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch):
        feat_spatial = self.spatial_encoder(batch)
        x_dense, mask = to_dense_batch(feat_spatial, batch.batch)

        feat_temporal_dense, _ = self.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]

        out_max = global_max_pool(feat_temporal, batch.batch)
        out_mean = global_mean_pool(feat_temporal, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))
