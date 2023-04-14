import torch

from model.attn_block import AttnBlock
from model.resnet_block import ResnetBlock


class XUNetBlock(torch.nn.Module):
    def __init__(self, in_channels, features, use_attn=False, attn_heads=4, dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.features = features
        self.use_attn = use_attn
        self.attn_heads = attn_heads
        self.dropout = dropout

        self.resnetblock = ResnetBlock(in_features=in_channels,
                                       out_features=features,
                                       dropout=dropout)

        if use_attn:
            self.attnblock_self = AttnBlock(attn_type="self",
                                            attn_heads=attn_heads,
                                            in_channels=features)
            self.attnblock_cross = AttnBlock(attn_type="cross",
                                             attn_heads=attn_heads,
                                             in_channels=features)

    def forward(self, x, emb):
        assert x.shape[-3] == self.in_channels, f"check if channel size is correct, {x.shape[-3]}!={self.in_channels}"
        h = self.resnetblock(x, emb)
        if self.use_attn:
            h = self.attnblock_self(h)
            h = self.attnblock_cross(h)

        return h