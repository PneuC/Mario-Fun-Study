"""
  @Time : 2022/1/4 10:15 
  @Author : Ziqi Wang
  @File : use_designer.py 
"""
import json
import torch

from src.gan.gankits import sample_latvec
from src.utils.filesys import getpath


class Designer:
    def __init__(self, path, device='cpu'):
        self.model = torch.load(getpath(f'{path}/actor.pth'), map_location=device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = device
        with open(getpath(f'{path}/N.json'), 'r') as f:
            self.n = json.load(f)

    def step(self, obs):
        model_in = torch.tensor(obs, device=self.device)
        if len(obs.shape) == 1:
            model_in = model_in.unsqueeze(0)
        model_output, _ = self.model(model_in)
        return model_output.squeeze().cpu().numpy()


class RandDesigner:
    def __init__(self):
        self.n = 1

    def step(self, obs):
        if len(obs.shape) == 1:
            return sample_latvec(1, tensor=False).squeeze()
        else:
            n = obs.shape[0]
            return sample_latvec(n, tensor=False)
