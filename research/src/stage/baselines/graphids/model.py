"""PyG GraphIDS core adapted from the authors' Apache-2.0 implementation.

The official model embeds flows as graph edges.  STAGE's graph constructor has
already converted flows into nodes and links flows sharing endpoints.  The
adapter therefore applies the same mean-aggregation E-GraphSAGE principle to
flow nodes, followed by the official masked Transformer reconstruction task.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch


class FlowSAGELayer(MessagePassing):
    def __init__(self, in_channels, edge_dim, out_channels, dropout=0.5):
        super().__init__(aggr="mean")
        self.message_projection = nn.Linear(in_channels + edge_dim, out_channels)
        self.update_projection = nn.Linear(in_channels + out_channels, out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.activation(self.message_projection(torch.cat([x_j, edge_attr], dim=-1)))

    def update(self, aggr_out, x):
        return self.dropout(self.update_projection(torch.cat([x, aggr_out], dim=-1)))


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=64,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.2,
        mask_ratio=0.15,
        feedforward_dim=256,
    ):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embed_dim, input_dim)

    def forward(self, src, valid_mask):
        projected = self.input_projection(src)
        padding_mask = ~valid_mask
        attention_mask = None
        if self.training and self.mask_ratio > 0:
            length = src.size(1)
            upper = torch.triu(
                torch.ones(length, length, device=src.device, dtype=torch.bool), diagonal=1
            )
            sampled = torch.rand(length, length, device=src.device) < self.mask_ratio
            attention_mask = (upper & sampled)
            attention_mask = attention_mask | attention_mask.T

        memory = self.encoder(
            projected,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )
        decoded = self.decoder(
            projected,
            memory,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        return self.output_projection(decoded)


class GraphIDS(nn.Module):
    """Self-supervised GraphIDS returning per-graph reconstruction errors."""

    def __init__(
        self,
        in_channels,
        edge_dim=2,
        hidden_dim=64,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.5,
        ae_dropout=0.2,
        mask_ratio=0.15,
        window_size=512,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.encoder = FlowSAGELayer(in_channels, edge_dim, hidden_dim, dropout=dropout)
        self.transformer = TransformerAutoencoder(
            input_dim=hidden_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=ae_dropout,
            mask_ratio=mask_ratio,
        )

    def encode_dense(self, data):
        x = torch.nan_to_num(data.x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(data.edge_attr.float(), nan=0.0, posinf=0.0, neginf=0.0)
        embeddings = self.encoder(x, data.edge_index, edge_attr)
        dense, valid_mask = to_dense_batch(embeddings, data.batch)
        return dense, valid_mask

    def _graph_node_errors(self, sequence):
        errors = []
        for start in range(0, sequence.size(0), self.window_size):
            target = sequence[start : start + self.window_size].unsqueeze(0)
            valid = torch.ones(
                (1, target.size(1)), dtype=torch.bool, device=target.device
            )
            reconstruction = self.transformer(target, valid)
            errors.append((reconstruction - target).pow(2).mean(dim=-1).squeeze(0))
        return torch.cat(errors)

    def reconstruction_loss(self, data):
        dense, valid_mask = self.encode_dense(data)
        errors = [
            self._graph_node_errors(dense[i][valid_mask[i]])
            for i in range(dense.size(0))
        ]
        return torch.cat(errors).mean()

    @torch.no_grad()
    def anomaly_score(self, data):
        dense, valid_mask = self.encode_dense(data)
        return torch.stack(
            [
                self._graph_node_errors(dense[i][valid_mask[i]]).mean()
                for i in range(dense.size(0))
            ]
        )
