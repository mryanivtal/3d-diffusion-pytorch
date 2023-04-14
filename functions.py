import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    """
    Gets a tensor t of numbers in [0, 1] (one scalar per sample in the batch)
    returns their cosine logsnr tensor of same size, scalars in [-1, 20]
    """
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b

    return -2. * torch.log(torch.tan(a * t + b))


def wrap_batch_in_dict(x, logsnr, z, R, T, K):
    return {
        'x': x.to(device),
        'z': z.to(device),
        'logsnr': torch.stack([logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1).to(device),
        'R': R.to(device),
        't': T.to(device),
        'K': K.to(device),
    }


def q_sample(z, logsnr, noise):
    """
    Add noise to z based on logsnr
    @param z: Tensor [batch, channels, h, w] - images
    @param logsnr: Tensor [batch, ] - scalars representing logsnr values, one per sample
    @param noise: random noise
    @return: Tensor [batch, channels, h, w] - noise-added images
    """
    alpha = logsnr.sigmoid().sqrt()
    sigma = (-logsnr).sigmoid().sqrt()

    alpha = alpha[:, None, None, None]
    sigma = sigma[:, None, None, None]

    return alpha * z + sigma * noise


def p_losses(denoise_model, img, R, T, K, logsnr, noise=None, loss_type="l2", cond_prob=0.1):
    """

    @param denoise_model: torch model
    @param img:Tensor of dims [Batch, 2, channels, H, W], where img[:, 0] = x (ref image), img[:, 1] = z (current image in generation)
    @param R: Camera rotations, Tensor of [Batch, 2, 3, 3] where R[:, 0] = x rotation, R[:, 1] = z rotation
    @param T: Camera positions, Tensor of [Batch, 2, 3] where T[:, 0] = x position, R[:, 1] = z position
    @param K: Camera intrinsics, Tensor of [Batch, 3, 3] - shared by all samples in batch
    @param logsnr: logsnr (reflects noise schedule for current diffusion step)
    @param noise: normal / other noise
    @param loss_type: l1, l2 etc
    @param cond_prob: ratio of number of images in the batch to replace by gaussian noise
    @return:
    """
    batch_size = img.shape[0]
    x = img[:, 0]
    z = img[:, 1]
    if noise is None:
        noise = torch.randn_like(x)

    # add noise to z images
    z_noisy = q_sample(z=z, logsnr=logsnr, noise=noise)

    # replace some images with random noise based on ratio
    cond_mask = (torch.rand((batch_size,)) > cond_prob).to(device)
    x_condition = torch.where(cond_mask[:, None, None, None], x, torch.randn_like(x))

    # wrap in dictinary, also adds logsnr=0 values to ref images (x)
    batch = wrap_batch_in_dict(x=x_condition, logsnr=logsnr, z=z_noisy, R=R, T=T, K=K)

    predicted_noise = denoise_model(batch, cond_mask=cond_mask.to(device))

    if loss_type == 'l1':
        loss = F.l1_loss(noise.to(device), predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise.to(device), predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise.to(device), predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def sample(model, img, R, T, K, w, timesteps=256):
    x = img[:, 0]
    img = torch.randn_like(x)
    imgs = []

    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[1:])

    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts)):  # [1, ..., 0] = size is 257
        img = p_sample(model, x=x, z=img, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def p_sample(model, x, z, R, T, K, logsnr, logsnr_next, w):
    model_mean, model_variance = p_mean_variance(model, x=x, z=z, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
    if logsnr_next == 0:
        return model_mean

    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def p_mean_variance(model, x, z, R, T, K, logsnr, logsnr_next, w=2.0):
    strt = time.time()
    b = x.shape[0]
    w = w[:, None, None, None]
    c = - torch.special.expm1(logsnr - logsnr_next)
    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
    alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    batch = wrap_batch_in_dict(x, logsnr.repeat(b), z, R, T, K)

    strt = time.time()
    pred_noise = model(batch, cond_mask=torch.tensor([True] * b)).detach().cpu()
    batch['x'] = torch.randn_like(x).to(device)
    pred_noise_unconditioned = model(batch, cond_mask=torch.tensor([False] * b)).detach().cpu()
    pred_noise_final = (1 + w) * pred_noise - w * pred_noise_unconditioned
    z = z.detach().cpu()
    z_start = (z - sigma * pred_noise_final) / alpha
    z_start.clamp_(-1., 1.)
    model_mean = alpha_next * (z * (1 - c) / alpha + c * z_start)
    posterior_variance = squared_sigma_next * c

    return model_mean, posterior_variance


def warmup(optimizer, step, last_step, last_lr):
    if step < last_step:
        optimizer.param_groups[0]['lr'] = step / last_step * last_lr
    else:
        optimizer.param_groups[0]['lr'] = last_lr


def evaluate_model(loader_val: DataLoader, model, tb_writer, step: int):
    print('Evaluating model...')
    model.eval()
    with torch.no_grad():
        for original_img, R, T, K in loader_val:
            current_batch_size = original_img.shape[0]
            w = torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7] * (current_batch_size // 8) + list(range(current_batch_size % 8)))
            image_samples = sample(model, img=original_img, R=R, T=T, K=K, w=w)

            # todo: need to fix the below - currently hard coded to batch_size=128. currently will update the batch size back to 128
            image_samples = rearrange(((image_samples[-1].clip(-1, 1) + 1) * 127.5).astype(np.uint8),
                                      "(b a) c h w -> a c h (b w)", a=8, b=16)
            gt = rearrange(((original_img[:, 1] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8),
                           "(b a) c h w -> a c h (b w)", a=8, b=16)
            cd = rearrange(((original_img[:, 0] + 1) * 127.5).detach().cpu().numpy().astype(np.uint8),
                           "(b a) c h w -> a c h (b w)", a=8, b=16)
            fi = np.concatenate([cd, gt, image_samples], axis=2)
            for i, ww in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
                tb_writer.add_image(f"train/{ww}", fi[i], step)
            break
    print('image sampled!')
    tb_writer.flush()
    model.train()