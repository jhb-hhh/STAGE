"""Fusion ablation variants for STAGE."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from stage.models.ablation_base import StageAblationBaseModel, TRANS_DROPOUT


class GCN_LSTM_Add(StageAblationBaseModel):
    """Additive fusion between GCN and Bi-LSTM features."""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        del self.gate_module

    def forward(self, batch):
        feat_spatial = self.spatial_encoder(batch)
        x_dense, mask = to_dense_batch(feat_spatial, batch.batch)
        feat_temporal_dense, _ = self.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]

        feat_fused = feat_spatial + feat_temporal
        out_max = global_max_pool(feat_fused, batch.batch)
        out_mean = global_mean_pool(feat_fused, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))


class GCN_LSTM_Concat(StageAblationBaseModel):
    """Concatenation fusion between GCN and Bi-LSTM features."""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        del self.gate_module
        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch):
        feat_spatial = self.spatial_encoder(batch)
        x_dense, mask = to_dense_batch(feat_spatial, batch.batch)
        feat_temporal_dense, _ = self.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]

        feat_fused = torch.cat([feat_spatial, feat_temporal], dim=1)
        out_max = global_max_pool(feat_fused, batch.batch)
        out_mean = global_mean_pool(feat_fused, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))


class CrossAttentionGate(nn.Module):
    """Cross-attention fusion from temporal features to spatial features."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_spatial, feat_temporal):
        query = feat_spatial.unsqueeze(1)
        key = feat_temporal.unsqueeze(1)
        value = feat_temporal.unsqueeze(1)
        attn_out, _ = self.mha(query, key, value)

        return self.ln(feat_spatial + self.dropout(attn_out.squeeze(1)))


class GCN_LSTM_CrossAttn(StageAblationBaseModel):
    """Cross-attention fusion between GCN and Bi-LSTM features."""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        del self.gate_module
        self.ca_gate = CrossAttentionGate(self.model_dim, num_heads=4, dropout=TRANS_DROPOUT)

    def forward(self, batch):
        feat_spatial = self.spatial_encoder(batch)
        x_dense, mask = to_dense_batch(feat_spatial, batch.batch)
        feat_temporal_dense, _ = self.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]

        feat_fused = self.ca_gate(feat_spatial, feat_temporal)
        out_max = global_max_pool(feat_fused, batch.batch)
        out_mean = global_mean_pool(feat_fused, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))
