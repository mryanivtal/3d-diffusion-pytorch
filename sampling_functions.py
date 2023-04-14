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
    """
    Samples batch of images from the model, within a single diffusion iteration
    @param model: model
    @param z: batch of prev. interation noisy images
    @param x: batch of ref. photos
    @param R: cam rotations - batch of tuples (Condition_R, Target_R)
    @param T: cam positions - batch of tuples (condition_T, target_T)
    @param K: cam intrinsics
    @param logsnr: scalar - current z log_SnR level
    @param logsnr_next: scalar - next z log_SnR level
    @param w: list: [0,..., B] B = batch size to sample
    @return: model_mean: Tensor[B, C, H, W], posterior_variance: Tensor[]
    """

    # todo: connect this code to equations / theory somehow
    w = w[:, None, None, None]
    b = w.shape[0]
    c = - torch.special.expm1(logsnr - logsnr_next)   # -1 * (exp(logsnr - logsnr_next) - 1)
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
    """
    Samples batch of images from the model, within a single diffusion iteration
    @param model: model
    @param z: batch of prev. interation noisy images
    @param x: batch of ref. photos
    @param R: cam rotations - batch of tuples (Condition_R, Target_R)
    @param T: cam positions - batch of tuples (condition_T, target_T)
    @param K: cam intrinsics
    @param logsnr: scalar - current z log_SnR level
    @param logsnr_next: scalar - next z log_SnR level
    @param w: list: [0,..., B] B = batch size to sample
    @return: batch of denoised images based on model.   if not final diffusion step - adds gaussian noise before return
    """
    model_mean, model_variance = p_mean_variance(model, x=x, z=z, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
    if logsnr_next == 0:
        return model_mean

    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def sample(model, record, target_R, target_T, K, w, timesteps=256):
    """
    Runs the entire diffusion flow and samples a batch of images form the model.
    @param model: torch model
    @param record: list of lists: [[ref_image: Tensor[B, C, H, W], ref_R: Tensor[3, 3],  ref_T: Tensor[3,]]]
    @param target_R: Target cam rotation
    @param target_T: target cam location
    @param K: cam intrinsics
    @param w: list: [0,..., B] B = batch size to sample
    @param timesteps: diffusion timesteps
    @return: final diffusion sample image batch
    """
    b = w.shape[0]
    # sample a batch of gaussian noise images from gaussian - initial input for backward process
    img = torch.randn_like(torch.tensor(record[0][0]))      #gaussian noise like image batch

    # create two tensors with shape [timesteps, ] of log(snr) values ordered desc, with phase 1 between them:
    # logsnrs:       logsnr_schedule_cosine(1.,...., almost 0)
    # logsnr_nexts:  logsnr_schedule_cosine(almost 1.,...., 0)
    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[1:])

    # diffusion loop, 256 steps
    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts), total=len(logsnrs), desc='diffusion loop', position=1, leave=False):  # [1, ..., 0] = size is 257
        # Choose one  image from the list in record for use as condition reference for this iteration
        condition_img, condition_R, condition_T = random.choice(record)
        condition_img = torch.tensor(condition_img)
        condition_R = torch.tensor(condition_R)
        condition_T = torch.tensor(condition_T)

        # Prepare data with useful format, we are using the same ref image for the entire batch so need to duplicate as batch
        R = torch.stack([condition_R, target_R], 0)[None].repeat(b, 1, 1, 1)
        T = torch.stack([condition_T, target_T], 0)[None].repeat(b, 1, 1)

        # Sample image based on model, this is the denoised output of this iteration
        img = p_sample(model, z=img, x=condition_img, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)

    return img.cpu().numpy()
