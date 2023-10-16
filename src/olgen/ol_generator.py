"""
  @Time : 2022/4/19 13:22 
  @Author : Ziqi Wang
  @File : ol_generator.py 
"""
from typing import List

import numpy as np
import torch
from torch import nn

from src.gan.gans import nz
from src.gan.gankits import process_onehot
from src.gan.gankits import sample_latvec
from src.smb.level import MarioLevel
from src.utils.datastruct import RingQueue


class PackedGenerator:
    # TODO: Pack desigenr and generator together
    def __init__(self, device):
        self.device = device
        pass

    def from_file(self, dpath, gpath):
        pass

    def step(self, observation):
        pass

    def forward(self, start_obs, n):
        pass

    def __pad_obs(self):
        pass


class OnlineGenerator:
    def __init__(self, designer, generator, g_device='cuda:0'):
        self.designer = designer
        self.generator = generator
        self.generator.to(g_device)
        self.g_device = g_device
        self.obs_buffer = RingQueue(designer.n)

    def re_init(self, condition=None):
        if condition is None:
            for _ in range(self.designer.n):
                latvec = sample_latvec(1, tensor=False).astype(np.float32).squeeze()
                self.obs_buffer.push(latvec)
        else:
            if len(condition) < self.designer.n:
                raise ValueError(f'Intial condition need {self.designer.n} items but {len(condition)} are given')
            for item in condition:
                self.obs_buffer.push(item)

    def step(self):
        obs = np.concatenate(self.obs_buffer.to_list())
        latvec = self.designer.step(obs)
        self.obs_buffer.push(latvec)
        z = torch.tensor(latvec, device=self.g_device).view(-1, nz, 1, 1)
        seg = process_onehot(self.generator(z))[0]
        seg_str = ('-' * seg.w + '\n') * max(0, 16 - seg.h)  + str(seg)
        return seg_str

    def forward(self, n) -> List[MarioLevel]:
        return [MarioLevel(self.step()) for _ in range(n)]

