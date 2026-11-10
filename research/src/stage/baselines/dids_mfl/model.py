"""DIDS-MFL adapted for STAGE's flow-node graph snapshots.

The official implementation operates on a single TemporalData event stream.
STAGE stores one chronologically ordered flow window per PyG graph.  This
adapter preserves the method's temporal memory, dynamic graph diffusion,
implicit/explicit feature disentanglement, and multi-scale fusion, then pools
each window for graph-level classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch


class DynamicGraphDiffusion(MessagePassing):
    """Residual dynamic diffusion corresponding to the official MGD block."""

    def __init__(self, channels=128, alpha=0.02, dropout_implicit=0.5, dropout_explicit=0.0):
        super().__init__(aggr="mean")
        self.alpha = float(alpha)
        self.implicit = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh(),
            nn.Dropout(dropout_implicit),
        )
        self.explicit = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ELU(),
            nn.Dropout(dropout_explicit),
        )
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x, edge_index):
        implicit = self.implicit(x)
        propagated = self.propagate(edge_index, x=self.explicit(x))
        return self.out_norm(x + self.alpha * propagated + (1.0 - self.alpha) * implicit)

    def message(self, x_j):
        return x_j


class MultiScaleFusion(nn.Module):
    """Learn short-, medium-, and long-range flow patterns before pooling."""

    def __init__(self, channels=128, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2) for k in (1, 3, 5)]
        )
        self.scale_gate = nn.Linear(channels * 3, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dense, mask):
        conv_input = dense.transpose(1, 2)
        scales = [F.elu(conv(conv_input)).transpose(1, 2) for conv in self.convs]
        summaries = []
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        for scale in scales:
            summaries.append((scale * mask.unsqueeze(-1)).sum(dim=1) / denom)
        joined = torch.cat(summaries, dim=-1)
        weights = torch.softmax(self.scale_gate(joined), dim=-1)
        fused = sum(weights[:, i : i + 1] * summaries[i] for i in range(3))
        return self.dropout(fused)


class DIDSMFL(nn.Module):
    """Graph-window DIDS-MFL with an auxiliary disentanglement objective."""

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_dim=128,
        diffusion_alpha=0.02,
        dropout_implicit=0.5,
        dropout_explicit=0.0,
    ):
        super().__init__()
        self.input_projection = nn.Linear(in_channels, hidden_dim)
        self.temporal_memory = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.diffusion = DynamicGraphDiffusion(
            hidden_dim,
            alpha=diffusion_alpha,
            dropout_implicit=dropout_implicit,
            dropout_explicit=dropout_explicit,
        )
        self.invariant_head = nn.Linear(hidden_dim, hidden_dim)
        self.specific_head = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = MultiScaleFusion(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data, return_aux=False):
        x = torch.nan_to_num(data.x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = F.elu(self.input_projection(x))
        spatial = self.diffusion(x, data.edge_index)

        dense, mask = to_dense_batch(spatial, data.batch)
        temporal, _ = self.temporal_memory(dense)
        temporal_flat = temporal[mask]

        invariant = torch.tanh(self.invariant_head(spatial))
        specific = F.elu(self.specific_head(temporal_flat))
        node_repr = invariant + specific

        node_dense, node_mask = to_dense_batch(node_repr, data.batch)
        multi_scale = self.fusion(node_dense, node_mask)
        graph_mean = global_mean_pool(node_repr, data.batch)
        logits = self.classifier(torch.cat([graph_mean, multi_scale], dim=-1))

        if not return_aux:
            return logits

        inv_norm = F.normalize(invariant, dim=-1)
        spec_norm = F.normalize(specific, dim=-1)
        orthogonality = (inv_norm * spec_norm).sum(dim=-1).pow(2).mean()
        centered = invariant - invariant.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(1, centered.size(0) - 1)
        decorrelation = (covariance - torch.diag(torch.diag(covariance))).pow(2).mean()
        smoothness = (node_repr[data.edge_index[0]] - node_repr[data.edge_index[1]]).pow(2).mean()
        return logits, {
            "orthogonality": orthogonality,
            "decorrelation": decorrelation,
            "smoothness": smoothness,
        }
