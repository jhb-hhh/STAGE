"""STAGE model definition."""

import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

TRANS_HIDDEN_DIM = 64
TRANS_DROPOUT = 0.1
NUM_NODE_FEATURES = 8
GAT_HIDDEN_DIM = 64
GAT_HEADS = 4
VOCAB_SIZE_PROTO = 256
VOCAB_SIZE_PORT = 1026
VOCAB_SIZE_FLAGS = 64


class GCNEncoder(nn.Module):
    """GCN-based spatial encoder for flow-level graph nodes."""

    def __init__(self):
        super().__init__()
        self.proto_emb = nn.Embedding(VOCAB_SIZE_PROTO, 8)
        self.port_emb = nn.Embedding(VOCAB_SIZE_PORT, 16)
        self.flag_emb = nn.Embedding(VOCAB_SIZE_FLAGS, 8)

        self.input_dim = NUM_NODE_FEATURES + 8 + 16 + 16 + 8
        hidden_aligned = GAT_HIDDEN_DIM * GAT_HEADS

        self.gcn1 = GCNConv(self.input_dim, hidden_aligned)
        self.ln1 = LayerNorm(hidden_aligned)
        self.gcn2 = GCNConv(hidden_aligned, TRANS_HIDDEN_DIM)
        self.ln2 = LayerNorm(TRANS_HIDDEN_DIM)
        self.act = nn.ELU()

    def forward(self, data):
        x = torch.cat(
            [
                data.x,
                self.proto_emb(data.proto_idx),
                self.port_emb(data.sport_idx),
                self.port_emb(data.dport_idx),
                self.flag_emb(data.flags_idx),
            ],
            dim=1,
        )

        x = self.act(self.ln1(self.gcn1(x, data.edge_index)))
        x = self.act(self.ln2(self.gcn2(x, data.edge_index)))

        return x


class BiLSTMEncoder(nn.Module):
    """Bi-LSTM temporal encoder over dense graph batches."""

    def __init__(self, d_model, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        out, (_, cell_state) = self.lstm(x)
        out = self.ln(out)

        return out, cell_state


class ResidualVectorGate(nn.Module):
    """Residual vector gate for spatial-temporal feature fusion."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim * 2)
        self.relu = nn.ReLU()
        self.gnn_delta = nn.Linear(dim * 2, dim)
        self.trans_delta = nn.Linear(dim * 2, dim)

    def forward(self, feat_spatial, feat_temporal):
        combined = torch.cat([feat_spatial, feat_temporal], dim=1)
        gate_hidden = self.relu(self.fc1(combined))

        delta_spatial = torch.tanh(self.gnn_delta(gate_hidden))
        delta_temporal = torch.tanh(self.trans_delta(gate_hidden))

        w_spatial = 1.0 + delta_spatial
        w_temporal = 1.0 + delta_temporal
        feat_fused = feat_spatial * w_spatial + feat_temporal * w_temporal

        return feat_fused, w_spatial, w_temporal


class StageModel(nn.Module):
    """Final STAGE model with endogenous node gating."""

    def __init__(self, num_classes):
        super().__init__()
        self.model_dim = TRANS_HIDDEN_DIM

        self.spatial_encoder = GCNEncoder()
        self.pos_encoder = self._make_pos_encoding(self.model_dim)
        self.temporal_encoder = BiLSTMEncoder(
            d_model=self.model_dim,
            num_layers=2,
            dropout=TRANS_DROPOUT,
        )
        self.gate_module = ResidualVectorGate(self.model_dim)

        self.node_importance_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.Tanh(),
            nn.Linear(self.model_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _make_pos_encoding(d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, batch, return_explanation=False):
        feat_spatial = self.spatial_encoder(batch)

        x_dense, mask = to_dense_batch(feat_spatial, batch.batch)
        x_dense = x_dense + self.pos_encoder[:, : x_dense.size(1), :].to(x_dense.device)

        feat_temporal_dense, _ = self.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]

        feat_fused, w_spatial, w_temporal = self.gate_module(feat_spatial, feat_temporal)
        importance_mask = self.node_importance_net(feat_fused)
        feat_weighted = feat_fused * importance_mask

        out_max = global_max_pool(feat_weighted, batch.batch)
        out_mean = global_mean_pool(feat_weighted, batch.batch)
        logits = self.classifier(torch.cat([out_max, out_mean], dim=1))

        if return_explanation:
            return logits, {
                "final_attn": importance_mask,
                "gate_spa": w_spatial,
                "gate_tem": w_temporal,
                "mask": mask,
            }

        return logits
