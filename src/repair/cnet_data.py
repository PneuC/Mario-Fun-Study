"""
  @Time : 2022/1/5 14:41 
  @Author : Ziqi Wang
  @File : generate_data.py 
"""

import glob
import json
import numpy as np
from root import PRJROOT
from itertools import product
from src.smb.level import MarioLevel, traverse_level_files
from src.utils.filesys import getpath

shifts = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

def get_combinations(lvl):
    # res = []
    h, w = lvl.h, lvl.w
    padded = MarioLevel.n_types * np.ones((h + 2, w + 2), dtype=np.int32)
    padded[1:-1, 1:-1] = lvl.to_num_arr()

    centers, surroundings = [], []
    for i, j in product(range(1, h + 1), range(1, w + 1)):
        indexes = np.array([i, j]) + shifts
        srrd = padded[indexes[:, 0], indexes[:, 1]]
        # if padded[i, j] in ppset or any(x in ppset for x in srrd):  # only record combinations near pipes
        centers.append(padded[i, j])
        surroundings.append(srrd)
            # res.append((padded[i, j], surroundings))
    return centers, surroundings

def generate_cnet_data():
    # lvl_files = glob.glob(getpath('levels/train/*.smblv'))
    levels = [lvl for lvl, _ in traverse_level_files('smb/levels')] #[MarioLevel.from_file(lvl_file) for lvl_file, _ in lvl_files]
    centers, surroundings = [], []
    for lvl in levels:
        c, s = get_combinations(lvl)
        centers += c
        surroundings += s

    counter = {}
    for center, srrd in zip(centers, surroundings):
        key = ','.join(map(str, srrd))
        if key in counter.keys():
            counter[key].add(center)
        else:
            counter[key] = {center}
    data = []

    for key in counter.keys():
        surrounding = list(map(int, key.split(',')))
        valids = np.zeros(MarioLevel.n_types)
        for i in counter[key]:
            valids[i] = 1

        data.append((surrounding, valids.tolist()))
    with open(getpath('smb/cnet_data.json'), 'w') as f:
        json.dump(data, f)
    pass


if __name__ == '__main__':
    generate_cnet_data()
