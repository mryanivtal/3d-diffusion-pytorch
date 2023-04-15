import numpy as np
from PIL import Image


def show_image_from_tensor(img_tensor):
    assert len(img_tensor.shape) == 3, f'img_tensor shape should be [C, H, W], got {img_tensor.shape}'
    if img_tensor.type() == 'torch.FloatTensor':
        img_tensor = img_tensor.numpy()
    Image.fromarray(((img_tensor.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).show()