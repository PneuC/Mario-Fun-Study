"""
  @Time : 2022/3/10 11:15 
  @Author : Ziqi Wang
  @File : math.py 
"""

import math
import numpy as np
from scipy.stats import entropy, norm


def a_clip(v, g, r=1.0, s=0):
    if s > 0:
        return min(r, 1 - (v - g) / g)
    elif s < 0:
        return min(r, 1 - (g - v) / g)
    else:
        return min(r, 1 - abs(v - g) / g)

def jsdiv(p, q):
    return (entropy(p, p + q, base=2) + entropy(q, p + q, base=2)) / 2

def grid_cnt(data, ranges, n_grids=10, normalize=True):
    eps = 1e-10
    d = data.shape[1]
    res = np.zeros([n_grids] * d)
    itvs = (ranges[:, 1] - ranges[:, 0]) * ((1 + eps) / n_grids)

    for item in data:
        indexes = tuple((item // itvs))
        res[indexes] = res[indexes] + 1
    if normalize:
        res /= res.size
    return res

def crowdivs(distmat):
    N = len(distmat)
    vmax = np.max(distmat)
    return N ** -0.5 * np.min(distmat + np.identity(N) * vmax, axis=0).sum()

def lpdist_mat(X, Y, p=2):
    diff = np.abs(np.expand_dims(X, axis=1) - np.expand_dims(Y, axis=0))
    distance_matrix = np.sum(diff ** p, axis=-1) ** (1 / p)
    return distance_matrix

def linfdist_mat(X, Y):
    diff = np.abs(np.expand_dims(X, axis=1) - np.expand_dims(Y, axis=0))
    distance_matrix = np.max(diff, axis=-1)
    return distance_matrix

def p_val_of_tau(tau, n):
    z = 3 * abs(tau) * math.sqrt(n * (n-1) / (4*n + 10))
    return 2 * (1 - norm.cdf(z))
    pass

if __name__ == '__main__':
    # x = [[1, 0], [0, 1], [3, -1]]
    # print(lpdist_mat(x, x, 1))
    # print(lpdist_mat(x, x, 2))
    # print(linfdist_mat(x, x))

    # print(p_val_of_tau(0.125, 48), p_val_of_tau(0.000, 42), p_val_of_tau(0.067, 90))
    # print(p_val_of_tau(0.292, 48), p_val_of_tau(0.190, 42), p_val_of_tau(0.244, 90))
    # print(p_val_of_tau(0.083, 48), p_val_of_tau(0.070, 42), p_val_of_tau(0.011, 90))
    # print(p_val_of_tau(0.333, 48), p_val_of_tau(0.238, 42), p_val_of_tau(0.289, 90))
    # print(p_val_of_tau(0.083, 48), p_val_of_tau(0.095, 42), p_val_of_tau(0.000, 90))

    # print(p_val_of_tau(0.082, 71), p_val_of_tau(0.246, 71), p_val_of_tau(0.049, 71), p_val_of_tau(0.279, 71), p_val_of_tau(0.049, 71))
    print(p_val_of_tau(0.034, 29), p_val_of_tau(0.241, 29), p_val_of_tau(0.103, 29), p_val_of_tau(0.310, 29), p_val_of_tau(0.103, 71))
    # print(p_val_of_tau(0.067, 90), p_val_of_tau(0.244, 90), p_val_of_tau(0.000, 90), p_val_of_tau(0.289, 90), p_val_of_tau(0.000, 90))
    pass

