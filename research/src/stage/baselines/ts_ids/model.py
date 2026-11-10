"""TS-IDS baseline model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool


class TS_IDS(nn.Module):
    """TS-IDS baseline adapted for graph-level intrusion detection."""

    def __init__(self, in_channels, num_classes, hidden_dim=128, num_layers=2, nhead=4, dropout=0.2):
        super(TS_IDS, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_dim // nhead, heads=nhead, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // nhead, heads=nhead, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.ssl_discriminator = nn.Linear(hidden_dim, 2)
        self.dropout_ratio = dropout

    def forward(self, data):
        """Run graph-level classification."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_out = torch.cat([x_max, x_mean], dim=1)

        return self.classifier(x_out)
