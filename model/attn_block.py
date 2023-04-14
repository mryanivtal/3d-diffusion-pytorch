import numpy as np
import torch
from einops import rearrange

from model.group_norm import GroupNorm


class AttnLayer(torch.nn.Module):
    def __init__(self, attn_heads=4, in_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.attn_heads = attn_heads
        self.attn = torch.nn.MultiheadAttention(in_channels, attn_heads, batch_first=True)
        # hidden_dim = attn_heads * in_channels
        # self.q = torch.nn.Conv1d(in_channels, hidden_dim, 1)
        # self.kv = torch.nn.Conv1d(in_channels, hidden_dim*2, 1)

    def forward(self, q, kv):
        assert len(q.shape) == 3, "make sure the size if [current_batch_size, C, H*W]"
        assert len(kv.shape) == 3, "make sure the size if [current_batch_size, C, H*W]"
        assert q.shape[1] == self.in_channels

        q = rearrange(q, "b c l -> b l c")
        kv = rearrange(kv, "b c l -> b l c")
        out = self.attn(q, kv, kv)[0]
        return rearrange(out, "b l c -> b c l")


class AttnBlock(torch.nn.Module):
    def __init__(self, attn_type, attn_heads=4, in_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.attn_type = attn_type
        self.attn_heads = attn_heads

        self.groupnorm = GroupNorm(num_channels=in_channels)
        self.attn_layer = AttnLayer(attn_heads=attn_heads, in_channels=in_channels)
        # self.attn_layer1 = AttnLayer(attn_heads=attn_heads, in_channels=in_channels)
        self.linear = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, h_in):
        B, F, C, H, W = h_in.shape
        assert self.in_channels == C, f"{self.in_channels} {C}"

        h = self.groupnorm(h_in)
        h0 = h[:, 0].reshape(B, C, H * W)
        h1 = h[:, 1].reshape(B, C, H * W)

        if self.attn_type == 'self':
            h0 = self.attn_layer(q=h0, kv=h0)
            h1 = self.attn_layer(q=h1, kv=h1)
        elif self.attn_type == 'cross':
            h_0 = self.attn_layer(q=h0, kv=h1)
            h_1 = self.attn_layer(q=h1, kv=h0)

            h0 = h_0
            h1 = h_1
        else:
            raise NotImplementedError(self.attn_type)

        h = torch.stack([h0, h1], axis=1)
        h = h.view(B * F, C, H, W)
        h = self.linear(h)
        h = h.view(B, F, C, H, W)

        return (h + h_in) / np.sqrt(2)