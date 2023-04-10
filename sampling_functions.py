import random
import numpy as np
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b
    return -2. * torch.log(torch.tan(a * t + b))


def xt2batch(x, logsnr, z, R, T, K):
    b = x.shape[0]
    batch = {
        'x': x.to(device),
        'z': z.to(device),
        'logsnr': torch.stack([logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1).to(device),
        'R': R.to(device),
        't': T.to(device),
        'K': K[None].repeat(b, 1, 1).to(device),
    }
    return batch


@torch.no_grad()
def p_mean_variance(model, x, z, R, T, K, logsnr, logsnr_next, w):
    w = w[:, None, None, None]
    b = w.shape[0]
    c = - torch.special.expm1(logsnr - logsnr_next)
    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()

    alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))

    batch = xt2batch(x, logsnr.repeat(b), z, R, T, K)
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


@torch.no_grad()
def p_sample(model, z, x, R, T, K, logsnr, logsnr_next, w):
    model_mean, model_variance = p_mean_variance(model, x=x, z=z, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
    if logsnr_next == 0:
        return model_mean

    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def sample(model, record, target_R, target_T, K, w, timesteps=256):
    b = w.shape[0]
    img = torch.randn_like(torch.tensor(record[0][0]))

    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[1:])

    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts), total=len(logsnrs), desc='diffusion loop', position=1, leave=False):  # [1, ..., 0] = size is 257
        condition_img, condition_R, condition_T = random.choice(record)
        condition_img = torch.tensor(condition_img)
        condition_R = torch.tensor(condition_R)
        condition_T = torch.tensor(condition_T)
        R = torch.stack([condition_R, target_R], 0)[None].repeat(b, 1, 1, 1)
        T = torch.stack([condition_T, target_T], 0)[None].repeat(b, 1, 1)
        condition_img = condition_img
        img = p_sample(model, z=img, x=condition_img, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)

    return img.cpu().numpy()
