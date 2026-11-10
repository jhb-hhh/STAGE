"""FlowTransformer baseline model."""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class FlowTransformer(nn.Module):
    """Transformer baseline for graph-level flow-sequence classification."""

    def __init__(self, in_channels, num_classes, hidden_dim=128, nhead=2, num_layers=2, dropout=0.1):
        super(FlowTransformer, self).__init__()

        self.embedding = nn.Linear(in_channels, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        """Run sequence-based graph classification."""
        x, batch = data.x, data.batch

        x_dense, mask = to_dense_batch(x, batch)
        x_emb = self.embedding(x_dense)

        out = self.transformer_decoder(
            tgt=x_emb,
            memory=x_emb,
            tgt_key_padding_mask=~mask,
            memory_key_padding_mask=~mask,
        )

        lengths = mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(x_dense.size(0), device=x.device)
        last_token_feat = out[batch_indices, lengths, :]

        return self.classifier(last_token_feat)
