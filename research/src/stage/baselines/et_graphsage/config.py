"""Configuration defaults for the E-T-GraphSAGE baseline."""


class Config:
    hidden_dim = 128
    num_gnn_layers = 2
    num_trans_layers = 2
    nhead = 4
    dropout = 0.3
    learning_rate = 0.0001
    epochs = 30
