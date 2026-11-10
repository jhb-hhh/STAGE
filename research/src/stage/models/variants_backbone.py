"""Backbone replacement ablation variants."""

import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from stage.models.ablation_base import BiLSTMEncoder, StageAblationBaseModel, TRANS_DROPOUT
from stage.models.legacy.fusion_net import StandardTransformer
from stage.models.legacy.gat_transformer_rvg import LegacyGATTransformerRVGModel

TRANS_NUM_HEADS = 4


class GCN_Trans_RVG_Model(StageAblationBaseModel):
    """Variant using GCN, Transformer, and RVG fusion."""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.pos_encoder = self._make_pos_encoding(self.model_dim)
        self.temporal_encoder = StandardTransformer(
            d_model=self.model_dim,
            nhead=TRANS_NUM_HEADS,
            num_layers=2,
            dim_feedforward=self.model_dim * 4,
            dropout=TRANS_DROPOUT,
        )

    @staticmethod
    def _make_pos_encoding(d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, batch):
        feat_spatial = self.spatial_encoder(batch)

        x_dense, mask = to_dense_batch(feat_spatial, batch.batch)
        x_dense = x_dense + self.pos_encoder[:, : x_dense.size(1), :].to(x_dense.device)

        feat_temporal_dense, _ = self.temporal_encoder(x_dense, mask=~mask)
        feat_temporal = feat_temporal_dense[mask]

        feat_fused, _, _ = self.gate_module(feat_spatial, feat_temporal)

        out_max = global_max_pool(feat_fused, batch.batch)
        out_mean = global_mean_pool(feat_fused, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))


class GAT_LSTM_RVG_Model(LegacyGATTransformerRVGModel):
    """Variant using GAT, Bi-LSTM, and RVG fusion."""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.temporal_encoder = BiLSTMEncoder(
            d_model=self.model_dim,
            num_layers=2,
            dropout=TRANS_DROPOUT,
        )

    def forward(self, batch, return_explanation=False):
        gat_output = self.gat_encoder(batch)
        feat_spatial = gat_output[0] if isinstance(gat_output, tuple) else gat_output

        x_dense, mask = to_dense_batch(feat_spatial, batch.batch)
        feat_temporal_dense, _ = self.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]

        feat_fused, _, _ = self.gate_module(feat_spatial, feat_temporal)

        out_max = global_max_pool(feat_fused, batch.batch)
        out_mean = global_mean_pool(feat_fused, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))
