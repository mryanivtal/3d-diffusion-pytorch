import numpy as np
import torch
from einops import rearrange
import etils
from etils.enp import numpy_utils

from model.conditioning_processor import ConditioningProcessor
from model.group_norm import GroupNorm
from model.resnet_block import ResnetBlock
from model.xunet_block import XUNetBlock


lazy = numpy_utils.lazy
etils.enp.linalg._tf_or_xnp = lambda x: lazy.get_xnp(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def out_init_scale():
    raise NotImplementedError


class XUNet(torch.nn.Module):
    H: int = 128
    W: int = 128
    ch: int = 256
    ch_mult: tuple[int] = (1, 2, 2, 4)
    emb_ch: int = 1024
    num_res_blocks: int = 3
    attn_resolutions: tuple[int] = (2,3,4) # actually, level of depth, from 0 th N
    attn_heads: int = 4
    dropout: float = 0.1
    use_pos_emb: bool = True
    use_ref_pose_emb: bool = True

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()
        
        assert self.H % (2**(len(self.ch_mult)-1)) == 0, f"Size of the image must me multiple of {2**(len(self.ch_mult)-1)}"
        assert self.W % (2**(len(self.ch_mult)-1)) == 0, f"Size of the image must me multiple of {2**(len(self.ch_mult)-1)}"
        
        self.num_resolutions = len(self.ch_mult)
        self.conditioningprocessor = ConditioningProcessor(
            emb_ch=self.emb_ch,
            num_resolutions=self.num_resolutions,
            use_pos_emb=self.use_pos_emb,
            use_ref_pose_emb=self.use_ref_pose_emb,
            H = self.H,
            W = self.W
        )
        
        self.conv = torch.nn.Conv2d(3, self.ch, kernel_size=3, stride=1, padding='same')
        
        # channel size
        self.dim_in = [self.ch] + (self.ch * np.array(self.ch_mult)[:-1]).tolist()
        self.dim_out = (self.ch * np.array(self.ch_mult)).tolist()

        # Downsampling
        self.xunetblocks = torch.nn.ModuleList([])
        for i_level in range(self.num_resolutions):
            single_level = torch.nn.ModuleList([])
            for i_block in range(self.num_res_blocks):
                use_attn = i_level in self.attn_resolutions
                
                single_level.append(
                    XUNetBlock(
                        in_channels = self.dim_in[i_level] if i_block ==0 else self.dim_out[i_level],
                        features=self.dim_out[i_level],
                        dropout=self.dropout,
                        attn_heads=self.attn_heads,
                        use_attn=use_attn,
                    )
                )
                
            if i_level != self.num_resolutions - 1:
                # todo: the above looks lik a bug.
                #  based on the image in the paper - downsampling layer 3 should have a residual block as well.
                single_level.append(ResnetBlock(in_features=self.dim_out[i_level], 
                                                out_features=self.dim_out[i_level], 
                                                dropout=self.dropout, 
                                                resample='down'))
            self.xunetblocks.append(single_level)
            
        #middle
        self.middle = XUNetBlock(
            in_channels = self.dim_out[-1],
            features=self.dim_out[-1],
            dropout=self.dropout,
            attn_heads=self.attn_heads,
            use_attn= self.num_resolutions in self.attn_resolutions)
        
        #upsample
        self.upsample = torch.nn.ModuleDict()
        for i_level in reversed(range(self.num_resolutions)):
            single_level = torch.nn.ModuleList([])
            use_attn = i_level in self.attn_resolutions
            
            for i_block in range(self.num_res_blocks + 1):
                if i_block == 0:
                    # then the input size is same as output of previous level
                    prev_h_channels = self.dim_out[i_level+1] if (i_level+1 < len(self.dim_out)) else self.dim_out[i_level]
                    prev_emb_channels = self.dim_out[i_level]
                elif i_block == self.num_res_blocks:
                    prev_h_channels = self.dim_out[i_level]
                    prev_emb_channels = self.dim_in[i_level]
                else:
                    prev_h_channels = self.dim_out[i_level]
                    prev_emb_channels = self.dim_out[i_level]
                    
                    # self.dim_out[i_level]*2 if i_block != self.num_res_blocks else (self.dim_out[i_level] + self.dim_in[i_level])
                
                in_channels = prev_h_channels + prev_emb_channels 
                
                single_level.append(
                    XUNetBlock(
                        in_channels = in_channels,
                        features = self.dim_out[i_level],
                        dropout=self.dropout,
                        attn_heads=self.attn_heads,
                        use_attn=use_attn)
                )

            if i_level != 0:
                single_level.append(
                    ResnetBlock(in_features=self.dim_out[i_level], 
                                out_features=self.dim_out[i_level],
                                dropout=self.dropout, resample='up')
                )
            self.upsample[str(i_level)] = single_level

        self.lastgn = GroupNorm(num_channels=self.ch)
        self.lastconv = torch.nn.Conv2d(in_channels=self.ch, out_channels=3, kernel_size=3, stride=1,padding='same')
        torch.nn.init.zeros_(self.lastconv.weight)
        
    
    def forward(self, batch, *, cond_mask):
        B, C, H, W = batch['x'].shape
        for key, temp in batch.items():
            assert temp.shape[0] == B, f"{key} should have batch size of {B}, not {temp.shape[0]}"
        assert B == cond_mask.shape[0]
        assert (H, W) == (self.H, self.W), ((H, W), (self.H, self.W))

        # Conditional processor = create time and pose embedding
        logsnr_emb, pose_embs = self.conditioningprocessor(batch, cond_mask)
        del cond_mask

        # initial convolution - 2 FC layers
        h = torch.stack([batch['x'], batch['z']], dim=1)
        h = self.conv(rearrange(h, 'b f c h w -> (b f) c h w'))
        h = rearrange(h, '(b f) c h w -> b f c h w', b=B, f=2)

        # downsampling
        hs = [h]
        for i_level in range(self.num_resolutions):
            emb = logsnr_emb[..., None, None] + pose_embs[i_level]
            for i_block in range(self.num_res_blocks):
                h = self.xunetblocks[i_level][i_block](h, emb)
                hs.append(h)
                
            if i_level != self.num_resolutions - 1:
                h = self.xunetblocks[i_level][-1](
                    h, emb)
                hs.append(h)
                
        # middle, 1x block
        emb = logsnr_emb[..., None, None] + pose_embs[-1]
        h = self.middle(h, emb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            emb = logsnr_emb[..., None, None] + pose_embs[i_level]
            for i_block in range(self.num_res_blocks + 1):
                h = torch.concat([h, hs.pop()], dim=-3)
                
                orishape = h.shape
                h = self.upsample[str(i_level)][i_block](h, emb)
                
            if i_level != 0:
                h = self.upsample[str(i_level)][-1](h, emb)
                
        assert not hs # check hs is empty
                
        h = torch.nn.functional.silu(self.lastgn(h)) # [current_batch_size, F, self.ch, 128, 128]
        return rearrange(self.lastconv(rearrange(h, 'b f c h w -> (b f) c h w')), '(b f) c h w -> b f c h w', b=B)[:, 1]


if __name__ == "__main__":
    h,w = 56, 56
    b = 8
    a = torch.nn.DataParallel(XUNet(H=h, W=w, ch=128)).to(device)

    batch = {
    'x': torch.zeros(b,3, h, w).to(device),
    'z': torch.zeros(b,3, h, w).to(device),
    'logsnr': torch.tensor([10]*(2*b)).reshape(b,2),
    'R': torch.tensor([[[  # Define a rigid rotation
       [-1/3, -(1/3)**.5, (1/3)**.5],
       [1/3, -(1/3)**.5, -(1/3)**.5],
       [-2/3, 0, -(1/3)**.5]],
    ]]).repeat(b, 2,1,1).to(device),
    't': torch.tensor([[[2, 2, 2]]]).repeat(b,2,1).to(device),
    'K':torch.tensor([[  # Define a rigid rotation
       [1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
    ]]).repeat(b,1,1).to(device),
    }

    print(a(batch, cond_mask=torch.tensor([True]*b).to(device)).shape)