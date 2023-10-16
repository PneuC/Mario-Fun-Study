import json
import os
import glob
from itertools import combinations

import numpy as np
# from analysis.survey.test_metrics import test_part2
import pandas as pds
import pygame
from matplotlib import pyplot as plt

from analysis.algo.analyse_designer import test_metrics
from analysis.survey.test_metrics import compute_normalisation_ratio
from misc.make_rand_lvls import make_lvls
from src.smb.level import MarioLevel
from src.utils.filesys import getpath, load_dict_json
from src.utils.img import make_img_sheet
from src.smb.level import tile_pattern_js_div


def test_part2_repaired(parallel=5):
    def test_persona(personame):
        lvls = [MarioLevel.from_file(path) for path in glob.glob(getpath('lvls/part2-repaired', f'{personame}*.lvl'))]
        test_metrics(lvls, 'lvls/part2-repaired', f'{personame}-metricvals.json', parallel)
        print(personame, 'finsihed')

    test_persona('Runner')
    test_persona('Collector')
    test_persona('Killer')

def test_part2_fin(parallel=5):
    def test_persona(personame):
        lvls = [MarioLevel.from_file(path) for path in glob.glob(getpath('lvls/part2-fin', f'{personame}*.lvl'))]
        test_metrics(lvls, 'lvls/part2-fin', f'{personame}-metricvals.json', parallel, n=1)
        print(personame, 'finsihed')

    test_persona('Runner')
    test_persona('Collector')
    test_persona('Killer')


def viz_with_trace(lvlpath, tarpath, lw=3):
    def __foo(jsonpath):
        with open(jsonpath, 'r') as f:
            simlt_res = json.load(f)
        full_trace = []
        b = 0
        for item in simlt_res:
            seg_trace = item['trace']
            appendee = [[x + b, y] for x, y in seg_trace]
            full_trace += appendee
            b += 256
        return full_trace

    lvl = MarioLevel.from_file(lvlpath)
    name_no_ext = os.path.split(lvlpath)[-1]
    name_no_ext = name_no_ext[:name_no_ext.rindex('.')]
    r_trace = __foo(f'exp_data/agent_simlt_res/Runner/{name_no_ext}.json')
    k_trace = __foo(f'exp_data/agent_simlt_res/Killer/{name_no_ext}.json')
    c_trace = __foo(f'exp_data/agent_simlt_res/Collector/{name_no_ext}.json')

    img = lvl.to_img(None)
    # p = 0
    # while p < len(trace) and trace[p][0] < self.w * MarioLevel.tex_size:
    #     p += 1
    pygame.draw.lines(img, 'black', False, [(x, y-8) for x, y in r_trace], lw)
    pygame.draw.lines(img, 'red', False, [(x, y-8) for x, y in k_trace], lw)
    pygame.draw.lines(img, 'white', False, [(x, y-8) for x, y in c_trace], lw)
    # if save_path is not None:
    pygame.image.save(img, getpath(tarpath))

    return img


if __name__ == '__main__':
    clr_agree = '#FF4F8D' # '#764900'
    clr_disagree = '#0077FF' # '#452BE1'
    clr_equal = '#009474' # '#009763'
    clr_neither = '#8B51F8' # '#CA0AC2'

    plt.figure(figsize=(5, 4), dpi=256)
    plt.plot([0,0], [1, 1], c=clr_agree, lw=10, label='Agreement')
    plt.plot([0,0], [1, 1], c=clr_equal, lw=10, label='Equally Fun')
    plt.plot([0,0], [1, 1], c=clr_disagree, lw=10, label='Disagreement')
    plt.plot([0,0], [1, 1], c=clr_neither, lw=10, label='Neither was Fun')
    # plt.legend(framealpha=0, ncol=2, columnspacing=0.5)
    # plt.show()
    # plt.figure(figsize=(4, 4), dpi=256)
    plt.legend(framealpha=0, ncol=2, columnspacing=1.2)
    plt.show()
    pass

