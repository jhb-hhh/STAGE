"""Single-branch STAGE ablation variants."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from stage.models.ablation_base import (
    BiLSTMEncoder,
    GCNEncoder,
    NUM_NODE_FEATURES,
    TRANS_DROPOUT,
    TRANS_HIDDEN_DIM,
    VOCAB_SIZE_FLAGS,
    VOCAB_SIZE_PORT,
    VOCAB_SIZE_PROTO,
)


class GCN_Only_Model(nn.Module):
    """Ablation variant using only the spatial GCN branch."""

    def __init__(self, num_classes):
        super().__init__()
        self.model_dim = TRANS_HIDDEN_DIM
        self.spatial_encoder = GCNEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch):
        feat_spatial = self.spatial_encoder(batch)
        out_max = global_max_pool(feat_spatial, batch.batch)
        out_mean = global_mean_pool(feat_spatial, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))


class LSTM_Only_Model(nn.Module):
    """Ablation variant using only the temporal Bi-LSTM branch."""

    def __init__(self, num_classes):
        super().__init__()
        self.model_dim = TRANS_HIDDEN_DIM

        self.proto_emb = nn.Embedding(VOCAB_SIZE_PROTO, 8)
        self.port_emb = nn.Embedding(VOCAB_SIZE_PORT, 16)
        self.flag_emb = nn.Embedding(VOCAB_SIZE_FLAGS, 8)

        input_dim = NUM_NODE_FEATURES + 8 + 16 + 16 + 8
        self.input_proj = nn.Linear(input_dim, self.model_dim)
        self.temporal_encoder = BiLSTMEncoder(
            d_model=self.model_dim,
            num_layers=2,
            dropout=TRANS_DROPOUT,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch):
        x = torch.cat(
            [
                batch.x,
                self.proto_emb(batch.proto_idx),
                self.port_emb(batch.sport_idx),
                self.port_emb(batch.dport_idx),
                self.flag_emb(batch.flags_idx),
            ],
            dim=1,
        )

        x = self.input_proj(x)
        x_dense, mask = to_dense_batch(x, batch.batch)
        feat_temporal_dense, _ = self.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]

        out_max = global_max_pool(feat_temporal, batch.batch)
        out_mean = global_mean_pool(feat_temporal, batch.batch)

        return self.classifier(torch.cat([out_max, out_mean], dim=1))
