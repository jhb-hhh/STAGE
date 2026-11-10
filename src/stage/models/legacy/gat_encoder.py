import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm

                                                              
                                                                        
NUM_NODE_FEATURES = 8
GAT_HIDDEN_DIM = 64
GAT_HEADS = 4
GAT_DROPOUT = 0.3
PROTO_MAP = {6: 0, 17: 1}
FLAG_MAP = {
    'NONE': 0, 'S': 1, 'SA': 2, 'A': 3, 'F': 4,
    'FA': 5, 'FSA': 6, 'R': 7, 'PA': 8, 'FPA': 9
}


class GATEncoder(nn.Module):
    """
    Micro-spatial encoder based on GATv2.
    """

    def __init__(self):
        super(GATEncoder, self).__init__()

                           
        self.proto_emb = nn.Embedding(num_embeddings=len(PROTO_MAP) + 1, embedding_dim=8)
        self.port_emb = nn.Embedding(num_embeddings=1025, embedding_dim=16)
        self.flag_emb = nn.Embedding(num_embeddings=len(FLAG_MAP) + 1, embedding_dim=8)

                                                      
        self.edge_rel_emb = nn.Embedding(num_embeddings=3, embedding_dim=8)

                                                  
                                                                    
        self.input_dim = NUM_NODE_FEATURES + 8 + 16 + 16 + 8

                       
        self.gat1 = GATv2Conv(
            in_channels=self.input_dim,
            out_channels=GAT_HIDDEN_DIM,
            heads=GAT_HEADS,
            dropout=GAT_DROPOUT,
            edge_dim=9,       
            concat=True
        )

        self.ln1 = LayerNorm(GAT_HIDDEN_DIM * GAT_HEADS)

        self.gat2 = GATv2Conv(
            in_channels=GAT_HIDDEN_DIM * GAT_HEADS,
            out_channels=GAT_HIDDEN_DIM,
            heads=1,
            dropout=GAT_DROPOUT,
            edge_dim=9,
            concat=False
        )

        self.ln2 = LayerNorm(GAT_HIDDEN_DIM)
        self.act = nn.ELU()

    def forward(self, data, return_attn=False,only_embedding=False):
        """
        Run the GAT encoder and optionally return attention weights.
        The return structure is kept stable for both training and explanation.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

                               
        proto_vec = self.proto_emb(data.proto_idx)
        src_port_vec = self.port_emb(data.sport_idx)
        dst_port_vec = self.port_emb(data.dport_idx)
        flag_vec = self.flag_emb(data.flags_idx)

        x = torch.cat([x, proto_vec, src_port_vec, dst_port_vec, flag_vec], dim=1)

        if only_embedding:
            return x
                                    
        rel_types = edge_attr[:, 0].long()
        time_diffs = edge_attr[:, 1].unsqueeze(1)

        rel_vec = self.edge_rel_emb(rel_types)
        edge_embeddings = torch.cat([rel_vec, time_diffs], dim=1)

                                      
                 
        x = self.gat1(x, edge_index, edge_attr=edge_embeddings)
        x = self.ln1(x)
        x = self.act(x)
        x = F.dropout(x, p=GAT_DROPOUT, training=self.training)

                 
                                                                         
        
        
        x, (return_edge_index, att_weights) = self.gat2(
            x,
            edge_index,
            edge_attr=edge_embeddings,
            return_attention_weights=True
        )

        x = self.ln2(x)

                                                       
                                                                             
        return x, (return_edge_index, att_weights)
