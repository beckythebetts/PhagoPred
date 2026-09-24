import time
import h5py
import numpy as np
import torch
from cellpose import models

from PhagoPred import SETTINGS

H5 = '/home/ubuntu/PhagoPred/PhagoPred/Datasets/B.h5'

with h5py.File(H5, 'r') as f:
    image = f['Images']['Phase'][0]
print('image', image.shape, image.dtype)

model = models.CellposeModel(
    gpu=True,
    pretrained_model=str(SETTINGS.CELLPOSE_MODEL / 'Models' / 'model'))


def run(label, img, autocast=False, **kw):
    torch.cuda.synchronize()
    t = time.time()
    if autocast:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            m, _, _ = model.eval(img[:, :, np.newaxis], **kw)
    else:
        m, _, _ = model.eval(img[:, :, np.newaxis], **kw)
    torch.cuda.synchronize()
    n = len(np.unique(m)) - 1
    print(f'{label:42s} {time.time() - t:6.2f}s   ncells={n}')


full = image
half = image[::2, ::2]
print('half', half.shape)

run('warmup', full)
run('full res (baseline)', full)
run('full res, tile_overlap=0', full, tile_overlap=0)
run('full res, fp16 autocast', full, autocast=True)
run('half res (2x downsample)', half)
run('half res, fp16 autocast', half, autocast=True)
