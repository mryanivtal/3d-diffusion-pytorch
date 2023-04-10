from sampling_functions import sample
from xunet import XUNet
import torch
import numpy as np
from tqdm import tqdm
import os
import glob
from PIL import Image
from pathlib import Path
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="./latest.pt")
parser.add_argument('--imagesdir', type=str, default="../datasets/srn_cars/cars_train/a4d535e1b1d3c153ff23af07d9064736")
parser.add_argument('--outdir', type=str, default="./samples")

args = parser.parse_args()
OUTPUT_DIR = args.outdir
TRAINED_MODEL = args.model
IMAGES_FOLDER = args.images_folder


# === Folders etc. ====
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
assert Path(OUTPUT_DIR).exists(), f'Output path {OUTPUT_DIR} does not exist'
assert Path(IMAGES_FOLDER).exists(), f'Reference Data path {IMAGES_FOLDER} does not exist'
assert Path(TRAINED_MODEL).exists(), f'Model path {TRAINED_MODEL} does not exist'


# === Prep data ===
imgsize = 64
data_imgs = []
data_Rs = []
data_Ts = []

for img_filename in sorted(glob.glob(IMAGES_FOLDER + "/rgb/*.png")):
    img = Image.open(img_filename)
    img = img.resize((imgsize, imgsize))
    img = np.array(img) / 255 * 2 - 1
    img = img.transpose(2, 0, 1)[:3].astype(np.float32)
    data_imgs.append(img)
    pose_filename = os.path.join(IMAGES_FOLDER, 'pose', os.path.basename(img_filename)[:-4] + ".txt")
    pose = np.array(open(pose_filename).read().strip().split()).astype(float).reshape((4, 4))
    data_Rs.append(pose[:3, :3])
    data_Ts.append(pose[:3, 3])

data_K = np.array(open(os.path.join(IMAGES_FOLDER, 'intrinsics', os.path.basename(img_filename)[:-4] + ".txt")).read().strip().split()).astype(float).reshape((3, 3))
data_K = torch.tensor(data_K)


# === Prep model ===
model = XUNet(H=imgsize, W=imgsize, ch=128)
model = torch.nn.DataParallel(model)
model.to(device)

# todo: uncomment below to load checkpoint
# ckpt = torch.load(args.model)
# model.load_state_dict(ckpt['model'])


# === Create the list of available reference views of the 3d element,start with a single given one (initial reference) ===
w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
b = w.shape[0]
record = [[data_imgs[0][None].repeat(b, axis=0),
           data_Rs[0],
           data_Ts[0]]]

sample_path = Path(OUTPUT_DIR) / Path('0')
sample_path.mkdir(exist_ok=True)
os.makedirs(f'sampling/0', exist_ok=True)
Image.fromarray(((data_imgs[0].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save(sample_path / Path('gt.png'))


# === Do the diffusion loop (from noise to image etc.) ===
with torch.no_grad():
    step = 1
    for gt, R, T in tqdm(zip(data_imgs[1:], data_Rs[1:], data_Ts[1:]), total=len(data_imgs[1:]), desc='view loop',
                         position=0):
        R = torch.tensor(R)
        T = torch.tensor(T)
        img = sample(model, record=record, target_R=R, target_T=T, K=data_K, w=w)
        record.append([img, R.cpu().numpy(), T.cpu().numpy()])

        sample_path = Path(OUTPUT_DIR) / Path('step')
        sample_path.mkdir(exist_ok=True)
        Image.fromarray(((gt.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save(sample_path / Path('gt.png'))
        for i in w:
            Image.fromarray(((img[i].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save(sample_path / Path(f'{i}.png'))

        step += 1
