"""E-T-GraphSAGE baseline model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch


class EGraphSAGEConv(MessagePassing):
    """Edge-aware GraphSAGE convolution with batch normalization."""

    def __init__(self, in_channels, out_channels, edge_dim, dropout=0.0):
        super(EGraphSAGEConv, self).__init__(aggr="mean", flow="source_to_target")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.lin = nn.Linear(in_channels * 2 + edge_dim, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """Run edge-aware message passing."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """Construct messages from neighbor features and edge attributes."""
        return torch.cat([x_j, edge_attr], dim=-1)

    def update(self, aggr_out, x):
        """Update node representations after aggregation."""
        combined = torch.cat([x, aggr_out], dim=1)
        x = self.lin(combined)
        return self.norm(x)


class ETGraphSAGE(nn.Module):
    """E-T-GraphSAGE baseline for graph-level intrusion detection."""

    def __init__(
        self,
        in_channels,
        num_classes,
        edge_dim=2,
        hidden_dim=128,
        num_gnn_layers=2,
        num_trans_layers=1,
        nhead=4,
        dropout=0.3,
    ):
        super(ETGraphSAGE, self).__init__()

        self.dropout_ratio = dropout
        self.node_embedding = nn.Linear(in_channels, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                EGraphSAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, data):
        """Run graph-level classification."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if edge_attr is not None:
            edge_emb = self.edge_embedding(edge_attr.float())
        else:
            edge_emb = torch.zeros((edge_index.size(1), self.node_embedding.out_features), device=x.device)

        x = self.node_embedding(x)
        x = F.relu(x)

        for conv in self.gnn_layers:
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x_dense, mask = to_dense_batch(x, batch)
        x_trans = self.transformer(x_dense, src_key_padding_mask=~mask)
        x_out = x_trans[mask]

        out = global_mean_pool(x_out, batch)
        return self.classifier(out)
