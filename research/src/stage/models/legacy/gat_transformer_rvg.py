import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool
import math

                                                              
                                                                        
TRANS_HIDDEN_DIM = 64
TRANS_NUM_HEADS = 4
TRANS_DROPOUT = 0.1

from stage.models.legacy.gat_encoder import GATEncoder
from stage.models.legacy.fusion_net import StandardTransformer


                                                                 
class ResidualVectorGate(nn.Module):
    def __init__(self, dim):
        super(ResidualVectorGate, self).__init__()
        self.fc1 = nn.Linear(dim * 2, dim * 2)
        self.relu = nn.ReLU()
        self.gnn_delta = nn.Linear(dim * 2, dim)
        self.trans_delta = nn.Linear(dim * 2, dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        for m in [self.gnn_delta, self.trans_delta]:
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)

    def forward(self, feat_gnn, feat_trans):
                                      
        combined = torch.cat([feat_gnn, feat_trans], dim=1)
                             
        x_gate = self.relu(self.fc1(combined))
                                  
        delta_gnn = torch.tanh(self.gnn_delta(x_gate))
        delta_trans = torch.tanh(self.trans_delta(x_gate))
                                                   
        weight_gnn = 1.0 + delta_gnn
        weight_trans = 1.0 + delta_trans
                               
        feat_fused = (feat_gnn * weight_gnn) + (feat_trans * weight_trans)

        return feat_fused, weight_gnn, weight_trans


class LegacyGATTransformerRVGModel(nn.Module):
    """
    Legacy GAT-Transformer-RVG baseline model.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.model_dim = TRANS_HIDDEN_DIM

                             
        self.gat_encoder = GATEncoder()
        self.pos_encoder = self._make_pos_encoding(self.model_dim)

        self.transformer = StandardTransformer(
            d_model=self.model_dim,
            nhead=TRANS_NUM_HEADS,
            num_layers=2,
            dim_feedforward=self.model_dim * 4,
            dropout=TRANS_DROPOUT
        )

                                         
        self.gate_module = ResidualVectorGate(self.model_dim)

                               
        input_dim = self.model_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def _make_pos_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, batch, return_explanation=False):
        """
        Forward pass.
        :param return_explanation: If true, return intermediate attention and gate weights.
        """
                         
                                                                  
        
        if return_explanation:
            feat_gnn, gat_attn_data = self.gat_encoder(batch, return_attn=True)
        else:
            feat_gnn, _ = self.gat_encoder(batch)

                                 
        x_dense, mask = to_dense_batch(feat_gnn, batch.batch)
        x_dense = x_dense + self.pos_encoder[:, :x_dense.size(1), :].to(feat_gnn.device)
        padding_mask = ~mask

                                                                          
        if return_explanation:
            feat_trans_dense, trans_attn_maps = self.transformer(x_dense, mask=padding_mask, return_attn=True)
        else:
            feat_trans_dense, _ = self.transformer(x_dense, mask=padding_mask)

        feat_trans = feat_trans_dense[mask]

                               
                                                        
        feat_fused, weight_gnn, weight_trans = self.gate_module(feat_gnn, feat_trans)

                                     
        out_max = global_max_pool(feat_fused, batch.batch)
        out_mean = global_mean_pool(feat_fused, batch.batch)
        out = torch.cat([out_max, out_mean], dim=1)

        logits = self.classifier(out)

                              
        if return_explanation:
            return logits, {
                'gate_gnn': weight_gnn,                                        
                'gate_trans': weight_trans,                                         
                'attn_gnn': gat_attn_data,                               
                'attn_trans': trans_attn_maps                                                        
            }

        return logits
