import numpy as np
import torch
from einops import rearrange

from model.film import FiLM
from model.group_norm import GroupNorm


def nearest_neighbor_upsample(h):
    B, F, C, H, W = h.shape
    h = torch.nn.functional.interpolate(h,scale_factor=(1, 2, 2), mode='nearest')
    return h.view(B, F, C, 2*H, 2*W)


def avgpool_downsample(h, k=2):
    B, F, C, H, W = h.shape
    h = h.view(B * F, C, H, W)
    h = torch.nn.functional.avg_pool2d(h, kernel_size=k, stride=k)
    return h.view(B, F, C, H//2, W//2)
#   return nn.avg_pool(h, (1, k, k), (1, k, k))
    # raise NotImplementedError


class ResnetBlock(torch.nn.Module):
    """BigGAN-style residual block, applied over frames."""

    def __init__(self,
                 in_features, out_features: int = None,
                 dropout: float = 0,
                 resample: str = None):
        super().__init__()
        self.in_features = in_features
        self.features = out_features
        self.dropout = dropout
        self.resample = resample

        if resample is not None:
            self.updown = {
                'up': nearest_neighbor_upsample,
                'down': avgpool_downsample,
            }[resample]
        else:
            self.updown = torch.nn.Identity()

        self.groupnorm0 = GroupNorm(num_channels=in_features)
        self.groupnorm1 = GroupNorm(num_channels=self.features)
        self.conv1 = torch.nn.Conv2d(in_channels=in_features,
                                     out_channels=self.features,
                                     kernel_size=3,
                                     stride=1,
                                     padding='same')
        self.film = FiLM(self.features)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(in_channels=self.features,
                                     out_channels=self.features,
                                     kernel_size=3,
                                     padding='same',
                                     stride=1)

        if in_features != out_features:
            self.dense = torch.nn.Conv2d(in_features, out_features, kernel_size=1)

        torch.nn.init.zeros_(self.conv2.weight)

    def forward(self, h_in, emb):
        B, F, C, H, W= h_in.shape
        assert C == self.in_features

        h = torch.nn.functional.silu(self.groupnorm0(h_in))
        h = self.conv1(h.view(B * F, C, H, W))
        h = h.view(B, F, self.features, H, W)
        h = self.groupnorm1(h)
        h = self.film(h, emb)
        h = self.dropout(h)
        h = self.conv2(h.view(B * F, self.features, H, W)).view(B, F, self.features, H, W)

        if C != self.features:
            h_in = self.dense(rearrange(h_in, 'b f c h w -> (b f) c h w'))
            h_in = rearrange(h_in, '(b f) c h w -> b f c h w', b=B)

        return self.updown((h + h_in) / np.sqrt(2))
