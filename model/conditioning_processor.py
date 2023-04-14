import numpy as np
import torch
from einops import rearrange
import visu3d as v3d


class ConditioningProcessor(torch.nn.Module):
    def __init__(self, emb_ch, H, W,
                 num_resolutions,
                 use_pos_emb=True,
                 use_ref_pose_emb=True, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.emb_ch = emb_ch
        self.num_resolutions = num_resolutions
        self.use_pos_emb = use_pos_emb
        self.use_ref_pose_emb = use_ref_pose_emb

        self.logsnr_emb_emb = torch.nn.Sequential(
            torch.nn.Linear(emb_ch, emb_ch),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_ch, emb_ch)
        )

        D = 144
        if use_pos_emb:
            self.pos_emb = torch.nn.Parameter(torch.zeros(D, H, W), requires_grad=True)
            torch.nn.init.normal_(self.pos_emb, std=(1 / np.sqrt(D)))

        if use_ref_pose_emb:
            self.first_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
            torch.nn.init.normal_(self.first_emb, std=(1 / np.sqrt(D)))

            self.other_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
            torch.nn.init.normal_(self.other_emb, std=(1 / np.sqrt(D)))

        convs = []
        for i_level in range(self.num_resolutions):
            convs.append(torch.nn.Conv2d(in_channels=D,
                                         out_channels=self.emb_ch,
                                         kernel_size=3,
                                         stride=2 ** i_level, padding=1))

        self.convs = torch.nn.ModuleList(convs)


    def forward(self, batch, cond_mask):
        B, C, H, W = batch['x'].shape

        logsnr = torch.clip(batch['logsnr'], -20, 20)
        logsnr = 2 * torch.arctan(torch.exp(
            -logsnr / 2)) / torch.pi  # fixed, was "lossnr" instead of logsnr.   converts form [-20, 20] to [~1, ~0.00002]
        logsnr_emb = self.posenc_ddpm(logsnr, emb_ch=self.emb_ch, max_time=1.)
        logsnr_emb = self.logsnr_emb_emb(logsnr_emb)

        world_from_cam = v3d.Transform(R=batch['R'].cpu().numpy(), t=batch['t'].cpu().numpy())
        cam_spec = v3d.PinholeCamera(resolution=(H, W), K=batch['K'].unsqueeze(1).cpu().numpy())
        rays = v3d.Camera(
            spec=cam_spec, world_from_cam=world_from_cam).rays()

        pose_emb_pos = self.posenc_nerf(torch.tensor(rays.pos).float().to(self.device), min_deg=0, max_deg=15)
        pose_emb_dir = self.posenc_nerf(torch.tensor(rays.dir).float().to(self.device), min_deg=0, max_deg=8)
        pose_emb = torch.concat([pose_emb_pos, pose_emb_dir], dim=-1)  # [batch, 2, h, w, 144]

        assert cond_mask.shape == (B,), (cond_mask.shape, B)
        cond_mask = cond_mask[:, None, None, None, None]
        pose_emb = torch.where(cond_mask, pose_emb, torch.zeros_like(pose_emb))  # [current_batch_size, F, H, W, 144]
        pose_emb = rearrange(pose_emb, "b f h w c -> b f c h w")
        # pose_emb = torch.tensor(pose_emb).float().to(device)

        # checkpoint_path [current_batch_size, 1, C=144, H, W]
        if self.use_pos_emb:
            pose_emb += self.pos_emb[None, None]
        if self.use_ref_pose_emb:
            pose_emb = torch.concat([self.first_emb, self.other_emb], axis=1) + pose_emb
            # checkpoint_path [current_batch_size, 2, C=144, H, W]

        pose_embs = []
        for i_level in range(self.num_resolutions):
            B, F = pose_emb.shape[:2]
            pose_embs.append(
                rearrange(self.convs[i_level](
                    rearrange(pose_emb, 'b f c h w -> (b f) c h w')
                ),
                    '(b f) c h w -> b f c h w', b=B, f=F
                )
            )
        return logsnr_emb, pose_embs


    def posenc_nerf(self, x, min_deg=0, max_deg=15):
        """Concatenate x and its positional encodings, following NeRF."""
        if min_deg == max_deg:
            return x
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)]).float().to(self.device)
        xb = rearrange(
            (x[..., None, :] * scales[:, None]), "b f h w c d -> b f h w (c d)")
        emb = torch.sin(torch.concat([xb, xb + torch.pi / 2.], dim=-1))

        return torch.concat([x, emb], dim=-1)


    def posenc_ddpm(self, timesteps, emb_ch: int, max_time=1000.):
        """Positional encodings for noise levels, following DDPM."""
        # 1000 is the magic number from DDPM. With different timesteps, we
        # normalize by the number of steps but still multiply by 1000.
        timesteps = timesteps.float()
        timesteps *= (1000. / max_time)
        half_dim = emb_ch // 2
        # 10000 is the magic number from transformers.
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(self.device)
        emb = emb.reshape(*([1] * (timesteps.ndim - 1)), emb.shape[-1])
        emb = timesteps[..., None] * emb
        emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=-1).float()

        return emb