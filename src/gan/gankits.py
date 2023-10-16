"""
  @Time : 2022/7/18 14:12 
  @Author : Ziqi Wang
  @File : gankits.py
"""

import torch
import numpy as np
from src.smb.level import MarioLevel
from src.gan.gans import nz
from src.utils.filesys import getpath


def sample_latvec(n=1, distribuion='uniform', tensor=True, device='cpu'):
    if distribuion == 'normal':
        if tensor:
            return torch.randn(n, nz, 1, 1, device=device)
        else:
            return np.random.randn(n, nz).astype(np.float32)
    elif distribuion == 'uniform':
        if tensor:
            return torch.rand(n, nz, 1, 1, device=device) * 2 - 1
        else:
            return (np.random.rand(n, nz) * 2 - 1).astype(np.float32)
    else:
        raise TypeError(f'unknow noise distribution: {distribuion}')

def process_onehot(raw_tensor_onehot):
    H, W = MarioLevel.height, MarioLevel.seg_width
    res = []
    for single in raw_tensor_onehot:
        data = single[:, :H, :W].detach().cpu().numpy()
        lvl = MarioLevel.from_one_hot_arr(data)
        res.append(lvl)
    return res

def get_generator(path, device='cpu'):
    generator = torch.load(getpath(path), map_location=device)
    generator.requires_grad_(False)
    generator.eval()
    return generator
    pass


