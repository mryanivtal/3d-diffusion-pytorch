import torch


class FiLM(torch.nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, features: int, emb_ch=1024):
        super().__init__()
        self.features = features
        self.dense = torch.nn.Linear(emb_ch, 2 * features)

    def forward(self, h, emb):
        emb = self.dense(torch.nn.functional.silu(emb.transpose(-1, -3))).transpose(-1, -3)
        scale, shift = torch.split(emb, self.features, dim=-3)
        return h * (1. + scale) + shift
