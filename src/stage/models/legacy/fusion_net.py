import torch
import torch.nn as nn
import torch.nn.functional as F
import math


                                                      
class StandardTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(StandardTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None, return_attn=False):
        
                                                          
        src2, attn_weights = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask,
            average_attn_weights=False
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if return_attn:
            return src, attn_weights
        return src, None


class StandardTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(StandardTransformer, self).__init__()
        self.layers = nn.ModuleList([
            StandardTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, return_attn=False):
        
        all_attentions = []
        for layer in self.layers:
                                                                     
            x, attn = layer(x, src_key_padding_mask=mask, return_attn=return_attn)
            if return_attn:
                all_attentions.append(attn)

                                                                          
        final_attn = all_attentions[-1] if all_attentions else None

        if return_attn:
            return x, final_attn
        return x, None
