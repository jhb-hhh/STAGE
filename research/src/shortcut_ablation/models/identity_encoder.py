import torch
import torch.nn as nn


class ShortcutIdentityEncoder(nn.Module):
    """
    Encode shortcut-visible identity features:
    proto, flags, raw source/destination ports, global source/destination IP ids.
    """

    def __init__(
        self,
        ip_vocab_size,
        proto_vocab_size=256,
        port_vocab_size=65536,
        flag_vocab_size=64,
        proto_dim=8,
        port_dim=16,
        flag_dim=8,
        ip_dim=16,
    ):
        super().__init__()

        self.proto_emb = nn.Embedding(proto_vocab_size, proto_dim)
        self.port_emb = nn.Embedding(port_vocab_size, port_dim)
        self.flag_emb = nn.Embedding(flag_vocab_size, flag_dim)
        self.ip_emb = nn.Embedding(ip_vocab_size, ip_dim)

        self.output_dim = proto_dim + port_dim * 2 + flag_dim + ip_dim * 2

    def forward(self, data):
        src_port = data.sport_idx_raw.clamp(0, 65535).long()
        dst_port = data.dport_idx_raw.clamp(0, 65535).long()

        src_ip = data.src_ip_idx_global.clamp(min=0).long()
        dst_ip = data.dst_ip_idx_global.clamp(min=0).long()

        return torch.cat(
            [
                self.proto_emb(data.proto_idx.long()),
                self.port_emb(src_port),
                self.port_emb(dst_port),
                self.flag_emb(data.flags_idx.long()),
                self.ip_emb(src_ip),
                self.ip_emb(dst_ip),
            ],
            dim=1,
        )


class ShortcutFeatureAdapter(nn.Module):
    """
    Build model-visible shortcut feature matrix:
    [continuous x || shortcut identity embeddings]
    """

    def __init__(self, ip_vocab_size, base_dim=8):
        super().__init__()
        self.identity_encoder = ShortcutIdentityEncoder(ip_vocab_size=ip_vocab_size)
        self.output_dim = base_dim + self.identity_encoder.output_dim

    def forward(self, data):
        identity_feat = self.identity_encoder(data)
        x = torch.nan_to_num(data.x.float(), nan=0.0)
        return torch.cat([x, identity_feat], dim=1)
