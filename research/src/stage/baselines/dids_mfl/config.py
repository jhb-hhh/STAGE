"""Published/default DIDS-MFL settings used by the STAGE adapter."""


class Config:
    learning_rate = 1e-4
    epochs = 50
    hidden_dim = 128
    diffusion_layers = 1
    diffusion_alpha = 0.02
    dropout_implicit = 0.5
    dropout_explicit = 0.0
    disentangle_beta = 1.0
    disentangle_gamma = 0.1
    disentangle_eta = 0.1
    disentangle_alpha = 0.1
