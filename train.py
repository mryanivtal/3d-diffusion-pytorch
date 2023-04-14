import argparse
from functions import logsnr_schedule_cosine, p_losses, warmup, evaluate_model
from model.xunet import XUNet

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm
import time
from pathlib import Path

from SRNdataset import dataset, MultiEpochsDataLoader
from tensorboardX import SummaryWriter

# ===== Parse command line arguments =====
argparser = argparse.ArgumentParser()
argparser.add_argument('--outdir', type=str, default='./output', help='output folder')
argparser.add_argument('--datadir', type=str, default='../datasets/srn_cars/cars_train', help='dataset folder')
argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# argparser.add_argument('--timesteps', type=int, default=300, help='model number of timesteps (T)')
argparser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
argparser.add_argument('--batchsize', type=int, default=32, help='train batch size')
# argparser.add_argument('--randomseed', type=int, default=123, help='initial random seed')
argparser.add_argument('--checkpointdir', type=str, default=None, help='start from saved model')
# argparser.add_argument('--betastart', type=float, default=1e-4, help='diffusion model noise scheduler beta start')
# argparser.add_argument('--betaend', type=float, default=2e-2, help='diffusion model noise scheduler beta end')
argparser.add_argument('--checkpointevery', type=int, default=20, help='save checkpoint every N epochs, 0 for disable')
# argparser.add_argument('--inferonly', type=int, default=0, help='0 - train. 1 - Only sample from model, no training')
argparser.add_argument('--warmupsteps', type=int, default=None, help='amount of steps fpr warmup')
argparser.add_argument('--dlworkers', type=int, default=2, help='Number of dataloader workers')
argparser.add_argument('--onebatchperepoch', type=int, default=0, help='For debug purposes')
argparser.add_argument('--ideoverride', type=int, default=0, help='For debug purposes')
argparser.add_argument('--reportlossevery', type=int, default=10, help='Report loss every n batches')
argparser.add_argument('--evaluateevery', type=int, default=20, help='evaluate model every n epochs')

args = argparser.parse_args()
ONE_BATCH_PER_EPOCH = args.onebatchperepoch
CHECKPOINT_DIR = args.checkpointdir
OUTPUT_DIR = args.outdir
DATASET_DIR = args.datadir
# TIMESTEPS = args.timesteps
LEARNING_RATE = args.lr
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
# RANDOM_SEED = args.randomseed
DL_WORKERS = args.dlworkers
# BETA_START = args.betastart
# BETA_END = args.betaend
CHECKPOINT_EVERY = args.checkpointevery
# INFER_ONLY = args.inferonly
WARMUP_STEPS = args.warmupsteps
REPORT_LOSS_EVERY = args.reportlossevery
EVALUATE_EVERY = args.evaluateevery
IDE_OVERRIDE = args.ideoverride

if WARMUP_STEPS is None:
    WARMUP_STEPS = 10000000/BATCH_SIZE

# IDE Debug settings
IDE_OVERRIDE = 1            # todo: remove

if IDE_OVERRIDE == 1:
    DL_WORKERS=0
    ONE_BATCH_PER_EPOCH = 1
    BATCH_SIZE = 5
    CHECKPOINT_EVERY = 1
    EVALUATE_EVERY = 1
    REPORT_LOSS_EVERY = 1

# ===== CPU / GPU selection =====
# Note - works only on GPU at the moment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# ===== Data =====
image_size = 64
batch_size = BATCH_SIZE

d = dataset('train', path=Path(DATASET_DIR), imgsize=image_size)
d_val = dataset('val', path=Path(DATASET_DIR), imgsize=image_size)

loader = MultiEpochsDataLoader(d, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=DL_WORKERS)
loader_val = DataLoader(d_val, batch_size=128, shuffle=True, drop_last=True, num_workers=DL_WORKERS)   # todo: fix batch size after the image viewer size is made dynamic

# ===== Model and Optimizer =====
model = XUNet(H=image_size, W=image_size, ch=128)
model = torch.nn.DataParallel(model)
model.to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))

# ===== Load Model and Optimizer from checkpoint =====
if CHECKPOINT_DIR is None:
    checkpoint_path = Path(OUTPUT_DIR) / Path(str(int(time.time())))
    writer = SummaryWriter(checkpoint_path)
    step = 0
    print(f'Checkpoints path: {checkpoint_path}')
else:
    checkpoint_path = Path(CHECKPOINT_DIR)
    print('Loading model checkpoint from: ', checkpoint_path)
    ckpt = torch.load(checkpoint_path / Path('latest.pt'))
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    writer = SummaryWriter(checkpoint_path)
    step = ckpt['step']


# ===== Train loop - Epoch =====
for epoch in range(NUM_EPOCHS):
    print(f'starting epoch {epoch}')

    # ===== Train loop - Batch =====
    for image_samples, R, T, K in tqdm(loader):
        current_batch_size = image_samples.shape[0]
        warmup(optimizer, step, WARMUP_STEPS, LEARNING_RATE)
        optimizer.zero_grad()

        logsnr = logsnr_schedule_cosine(torch.rand((current_batch_size,)))
        loss = p_losses(model, img=image_samples.to(device), R=R.to(device), T=T.to(device), K=K.to(device), logsnr=logsnr.to(device), loss_type="l2", cond_prob=0.1)
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), global_step=step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step=step)

        # Report every N batches
        if (step + 1) % REPORT_LOSS_EVERY == 0:
            print("Loss:", loss.item())

        if step == int(WARMUP_STEPS):
            filename = checkpoint_path / Path('after_warmup.pt')
            print(f'Saving checkpoint after warmup to: {filename}')
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step}, filename)
        
        step += 1

        if ONE_BATCH_PER_EPOCH != 0:        # debug parameter: stop after one batch
            break

    # Evaluate model every N epochs
    if (epoch + 1) % EVALUATE_EVERY == 0:
        # todo: model evaluations not saved most of the times!
        evaluate_model(loader_val, model, writer, step)

    if CHECKPOINT_EVERY is not None:
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            filename = checkpoint_path / Path('latest.pt')
            print(f'Saving checkpoint to {filename}')
            torch.save({'optim':optimizer.state_dict(), 'model':model.state_dict(), 'step':step, 'epoch':epoch}, filename)