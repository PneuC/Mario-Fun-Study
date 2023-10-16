"""
  @Time : 2022/1/24 14:40 
  @Author : Ziqi Wang
  @File : gans.py
"""

import torch
from torch import nn
from src.smb.level import MarioLevel


nz = 20

# TODO: Try put self-attention layer after onehot directly

###### Borrowed from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py ######
class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
        # return out, attention
#######################################################################################################


class SAGenerator(nn.Module):
    def __init__(self, base_channels=32):
        super(SAGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(nz, base_channels * 4, 4)),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(),
            nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(),
            SelfAttn(base_channels * 2),
            nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)),
            nn.BatchNorm2d(base_channels), nn.ReLU(),
            SelfAttn(base_channels),
            nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels, MarioLevel.n_types, 3, 1, 1)),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)


class SADiscriminator(nn.Module):
    def __init__(self, base_channels=32):
        super(SADiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(MarioLevel.n_types, base_channels, 3, 1, 1)),
            nn.BatchNorm2d(base_channels), nn.LeakyReLU(0.1),
            SelfAttn(base_channels),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.BatchNorm2d(base_channels * 2), nn.LeakyReLU(0.1),
            SelfAttn(base_channels * 2),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.BatchNorm2d(base_channels * 4), nn.LeakyReLU(0.1),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 4, 1, 4)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.main(x)


# class SAGenerator(nn.Module):
#     def __init__(self, base_channels=32):
#         super(SAGenerator, self).__init__()
#         self.main = nn.Sequential(
#             nn.utils.spectral_norm(nn.ConvTranspose2d(nz, base_channels * 4, 4)),
#             nn.BatchNorm2d(base_channels * 4), nn.ReLU(),
#             nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)),
#             nn.BatchNorm2d(base_channels * 2), nn.ReLU(),
#             SelfAttn(base_channels * 2),
#             nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)),
#             nn.BatchNorm2d(base_channels), nn.ReLU(),
#             SelfAttn(base_channels),
#             nn.utils.spectral_norm(nn.ConvTranspose2d(base_channels, MarioLevel.n_types, 3, 1, 1)),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         return self.main(x)
#
#
# class SADiscriminator(nn.Module):
#     def __init__(self, base_channels=32):
#         super(SADiscriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.utils.spectral_norm(nn.Conv2d(MarioLevel.n_types, base_channels, 3, 1, 1)),
#             nn.BatchNorm2d(base_channels), nn.ELU(),
#             SelfAttn(base_channels),
#             nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
#             nn.BatchNorm2d(base_channels * 2), nn.ELU(),
#             SelfAttn(base_channels * 2),
#             nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
#             nn.BatchNorm2d(base_channels * 4), nn.ELU(),
#             nn.utils.spectral_norm(nn.Conv2d(base_channels * 4, 1, 4)),
#             nn.Flatten()
#         )
#
#     def forward(self, x):
#         return self.main(x)
#

if __name__ == '__main__':
    noise = torch.rand(2, nz, 1, 1) * 2 - 1
    netG = SAGenerator()
    netD = SADiscriminator()
    # print(netG)
    X = netG(noise)
    Y = netD(X)
    print(X.shape, Y.shape)
    pass

