import torch


class GroupNorm(torch.nn.Module):
    """Group normalization, applied over frames."""

    def __init__(self, num_groups=32, num_channels=64):
        super().__init__()
        self.gn = torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, h):
        B, _, C, H, W = h.shape
        h = self.gn(h.reshape(B * 2, C, H, W))
        return h.reshape(B, 2, C, H, W)

