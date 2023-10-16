"""
  @Time : 2022/3/24 16:09 
  @Author : Ziqi Wang
  @File : analyze_fun.py 
"""
import sys
sys.path.append('../')

import json
import time
from src.smb.level import *
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

def compute_fun_vals(gc, gb, n):
    ori_lvls, ori_simlt = collect_orilvls('smb/levels', 10)
    rand_lvls, rand_simlt = collect_batched_lvls('analysis/fun_tuning/randgen_lvls')
    print(len(ori_lvls), len(rand_lvls))
    fun_c_func = LevelSACN(g=gc, n=n)
    fun_b_func = GameplaySACN(g=gb, n=n)
    ori_fun_c = [
        np.mean(fun_c_func.compute_rewards(segs=segs))
        for segs in ori_lvls
    ]
    ori_fun_b = [
        np.mean(fun_b_func.compute_rewards(segs=segs, simlt_res=item))
        for segs, item in zip(ori_lvls, ori_simlt)
    ]
    rand_fun_c = [
        np.mean(fun_c_func.compute_rewards(segs=segs))
        for segs in rand_lvls
    ]
    rand_fun_b = [
        np.mean(fun_b_func.compute_rewards(segs=segs, simlt_res=item))
        for segs, item in zip(rand_lvls, rand_simlt)
    ]
    return {
        'gc': gc, 'gb': gb,
        'ori_fun_c': ori_fun_c,
        'ori_fun_b': ori_fun_b,
        'rand_fun_c': rand_fun_c,
        'rand_fun_b': rand_fun_b
    }


if __name__ == '__main__':
    # lvls, simlt = collect_batched_lvls('exp_data/main/behaviour/for_scatter')
    # print(len(lvls), len(lvls))
    # fun_c_func = FunContent()
    # fun_b_func = FunBehaviour()
    # fun_c = [
    #     np.array(fun_c_func.compute_rewards(segs=segs)[4:]).mean()
    #     for segs in lvls
    # ]
    # fun_b = [
    #     np.array(fun_b_func.compute_rewards(segs=segs, simlt_res=item)[4:]).mean()
    #     for segs, item in zip(lvls, simlt)
    # ]
    # with open(get_path('exp_data/main/behaviour/for_scatter/fun_vals.json'), 'w') as f:
    #     json.dump({'fun_c': fun_c, 'fun_b': fun_b}, f)
    parallel = 5
    gcs = np.linspace(0.1, 0.16, 4)
    gbs = np.linspace(0.1, 0.3, 5)
    computing_pool = MyAsyncPool(parallel)
    # gls = [0.2]
    # ggs = [0.35]
    # ns = [3, 4, 5]
    # for nval in ns:
    res = []
    start_time = time.time()
    for gcval, gbval in product(gcs, gbs):
        computing_pool.push(compute_fun_vals, (gcval, gbval, 5))
        while computing_pool.get_num_waiting() > 2 * parallel:
            res += computing_pool.collect()
            time.sleep(1)
    res += computing_pool.wait_and_get()
    with open(getpath('./fun_statistics.json'), 'w') as f:
        json.dump(res, f)

