"""GraphIDS settings based on the official CIC-IDS-2018-v2 configuration."""


class Config:
    learning_rate = 7.423525164428174e-5
    epochs = 100
    hidden_dim = 64
    ae_embedding_dim = 32
    num_heads = 4
    num_layers = 1
    mask_ratio = 0.15
    patience = 20
    dropout = 0.5
    ae_dropout = 0.2
    encoder_weight_decay = 0.6
    ae_weight_decay = 0.011263928281120002
    threshold_steps = 500
