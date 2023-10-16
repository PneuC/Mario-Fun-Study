"""
  @Time : 2022/3/24 16:09
  @Author : Ziqi Wang
  @File : analyze_fun.py
"""
import sys
sys.path.append('../')

import json
import time
import numpy as np
from src.smb.level import *
from matplotlib import pyplot as plt
from src.utils.parallel import MyAsyncPool
from src.env.rfunc import LevelSACN, GameplaySACN


def collect_orilvls(path, l):
    W = MarioLevel.seg_width
    ts = LevelRender.tex_size
    lvls, simlt_res = [], []
    for orilvl, name in traverse_level_files(path):
        with open(getpath(f'{path}/{name}_simlt_res.json'), 'r') as f:
            full_trace = json.load(f)['full_trace']

        for s in range(0, orilvl.w - W * l):
            segs = [orilvl[:, s+i*W:s+(i+1)*W] for i in range(l)]
            lvls.append(segs)

            simlt_res.append([])
            i, p = 0, 0
            while full_trace[p][0] < s * ts:
                p += 1
            while i < l:
                p0 = p
                while p < len(full_trace) and full_trace[p][0] < ((i + 1) * W + s) * ts:
                    p += 1
                simlt_res[-1].append(
                    {'trace': [[x-full_trace[p0][0], y] for x, y in full_trace[p0:p]]}
                )
                i += 1
    return lvls, simlt_res

def collect_batched_lvls(path):
    # W = MarioLevel.seg_width
    lvls, simlt_res = [], []
    for batch, name in traverse_batched_level_files(path):
        lvls += [lvl.to_segs() for lvl in batch]
        with open(getpath(f'{path}/{name}_simlt_res.json'), 'r') as f:
            data = json.load(f)
        simlt_res += [item['seg_infos'] for item in data]
    return lvls, simlt_res

def compute_fg_vals(g):
    ori_lvls, ori_simlt = collect_orilvls('smb/levels', 10)
    rand_lvls, rand_simlt = collect_batched_lvls('analysis/fun_tuning/randgen_lvls')
    print(len(ori_lvls), len(rand_lvls))
    fg_func = GameplaySACN(g=g)
    ori_fun_b = [
        np.mean(fg_func.compute_rewards(segs=segs, simlt_res=item))
        for segs, item in zip(ori_lvls, ori_simlt)
    ]
    rand_fun_b = [
        np.mean(fg_func.compute_rewards(segs=segs, simlt_res=item))
        for segs, item in zip(rand_lvls, rand_simlt)
    ]
    return {'g': g, 'ori': ori_fun_b, 'rand': rand_fun_b}

def compute_vals():
    parallel = 5
    gls = np.linspace(0.1, 0.16, 7)
    ggs = np.linspace(0.24, 0.36, 7)
    computing_pool = MyAsyncPool(parallel)
    fg_res = []
    for ggval in ggs:
        computing_pool.push(compute_fg_vals, (ggval,))
        while computing_pool.get_num_waiting() > 2 * parallel:
            fg_res += computing_pool.collect()
            time.sleep(1)
    fg_res += computing_pool.wait_and_get()
    with open(getpath('./fg_statistics.json'), 'w') as f:
        json.dump(fg_res, f)

    ori_lvls, _ = collect_orilvls('smb/levels', 10)
    rand_lvls, _ = collect_batched_lvls('analysis/fun_tuning/randgen_lvls')
    fl_res = []
    for glval in gls:
        fl_func = LevelSACN(g=glval)
        fl_oris = [np.mean(fl_func.compute_rewards(segs=segs)) for segs in ori_lvls]
        fl_rands = [np.mean(fl_func.compute_rewards(segs=segs)) for segs in rand_lvls]
        fl_res.append({'g': glval, 'ori': fl_oris, 'rand': fl_rands})
    with open(getpath('./fl_statistics.json'), 'w') as f:
        json.dump(fl_res, f)

def plot_fun_scatter(fl_infos, fg_infos):
    plt.figure(figsize=(4, 4), dpi=256)
    gl_val, gg_val = fl_infos['g'], fg_infos['g']
    x1, y1 = fl_infos['rand'], fg_infos['rand']
    x2, y2 = fl_infos['ori'], fg_infos['ori']
    plt.scatter(x1, y1, color='red', alpha=0.4, linewidths=0, s=15)
    plt.scatter(x2, y2, color='blue', alpha=0.4, linewidths=0, s=15)
    plt.xlabel('$f_L$')
    plt.ylabel('$f_G$', rotation=0.)
    # plt.title(f'gc={gl_val:.3f}, gb={gg_val:.3f}')
    plt.grid()
    # if n_init <= 4:
    plt.xlim((-1.6, 1.1))
    plt.ylim((-1.6, 1.1))
    # else:
    # plt.xlim((-3, 1.1))
    # plt.ylim((-3, 1.1))
    ticks = [-1.5, -1., -0.5, 0., 0.5, 1.]
    plt.xticks(ticks, map(lambda v: '%.1f' % v, ticks))
    plt.yticks(ticks, map(lambda v: '%.1f' % v, ticks))
    plt.tight_layout()
    plt.savefig(f'./fun_plots.png')
    pass

def find_best():
    with open(getpath('./fl_statistics.json'), 'r') as f:
        fl_data = json.load(f)
    with open(getpath('./fg_statistics.json'), 'r') as f:
        fg_data = json.load(f)
    best_ratio, best_gl, best_gg = 1, None, None
    fin_fl_infos, fin_fg_infos = None, None
    for fl_infos, fg_infos in product(fl_data, fg_data):
        mean_ori = 2 - (np.mean(fl_infos['ori']) + np.mean(fg_infos['ori']))
        mean_rands = 2 - (np.mean(fl_infos['rand']) + np.mean(fg_infos['rand']))
        # mean_ori = 1 - np.mean(fl_infos['ori'])
        # mean_rands = 1 - np.mean(fg_infos['rand'])
        ratio = mean_ori / mean_rands
        if ratio < best_ratio:
            best_ratio = ratio
            best_gl, best_gg = fl_infos['g'], fg_infos['g']
            fin_fl_infos = fl_infos
            fin_fg_infos = fg_infos
    print(best_gl, best_gg)
    plot_fun_scatter(fin_fl_infos, fin_fg_infos)
    pass

# def find_best(infos):
#     best_ratio, best_value = 1, None
#     for infos in infos:
#         # mean_ori = 2 - (np.mean(fl_infos['ori']) + np.mean(fg_infos['ori']))
#         # mean_rands = 2 - (np.mean(fl_infos['rand']) + np.mean(fg_infos['rand']))
#         mean_ori = 1 - np.mean(infos['ori'])
#         mean_rands = 1 - np.mean(infos['rand'])
#         ratio = mean_ori / mean_rands
#         if ratio < best_ratio:
#             best_ratio = ratio
#             best_value = infos['g']
#     print(best_value)

if __name__ == '__main__':
    compute_vals()
    # with open(getpath('./fl_statistics.json'), 'r') as f:
    #     find_best(json.load(f))
    # with open(getpath('./fg_statistics.json'), 'r') as f:
    #     find_best(json.load(f))
    # find_best()

    # with open(getpath('./fl_statistics.json'), 'r') as f:
    #     fl_data = json.load(f)
    # with open(getpath('./fg_statistics.json'), 'r') as f:
    #     fg_data = json.load(f)
    # plot_fun_scatter(fl_data[13], fg_data[8])
    pass
