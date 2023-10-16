"""
  @Time : 2022/7/27 17:35 
  @Author : Ziqi Wang
  @File : analyse_designer.py 
"""

import os
import json
import time
import pandas as pds
from src.env.rfunc import *
from src.gan.gankits import *
from itertools import combinations
from src.utils.mymath import crowdivs
from src.olgen.ol_generator import OnlineGenerator
from src.env.environment import make_vec_offrew_env
from src.smb.level import lvlhcat, lvl_manhhatan_div
from src.smb.proxy import MarioProxy, MarioJavaAgents
from src.olgen.designer import Designer, RandDesigner
from src.utils.parallel import MyAsyncPool


def generate_levels(designer, dest_path='', n=1, l=25, parallel=2):
    env = make_vec_offrew_env(parallel, N=designer.n, eplen=l, return_lvl=True, init_one=True)
    levels = []
    obs = env.reset()
    while len(levels) < n:
        actions = designer.step(obs)
        next_obs, _, dones, infos = env.step(actions)
        del obs
        obs = next_obs
        for done, info in zip(dones, infos):
            if done:
                level = MarioLevel(info['LevelStr'])
                levels.append(level)
                # print(f'{len(levels)}/{n} generated')
    if dest_path:
        os.makedirs(getpath(dest_path), exist_ok=True)
        for i in range(n):
            level = levels[i]
            level.save(f'{dest_path}/sample{i}.lvl')
            level.to_img(f'{dest_path}/sample{i}.png')
    return levels[:n]

def generate_from_same(designers, names, playable=True, trials=1, L=50, path='lvls'):
    for i in range(trials):
        n = max(d.n for d in designers)
        generator = get_generator('models/generator.pth')
        init_condition = sample_latvec(n)
        segs = process_onehot(generator(init_condition))
        if playable:
            proxy = MarioProxy()
            while True:
                level = lvlhcat(segs)
                res = proxy.simulate_game(level)
                if res['status'] == 'WIN':
                    break
                init_condition = sample_latvec(n)
                segs = process_onehot(generator(init_condition))
        for designer, name in zip(designers, names):
            gen_sys = OnlineGenerator(designer, generator)
            gen_sys.re_init(init_condition.squeeze().numpy())
            generated = gen_sys.forward(L)
            level = lvlhcat(segs + generated)
            level.save(f'{path}/{name}-{i}')
            level.to_img(f'{path}/{name}-{i}.png')
    pass

def test_diversity(designer, n=200, l=50, trails=30, gen_parallel=1):
    res = 0
    start_time = time.time()
    for t in range(trails):
        levels = generate_levels(designer, n=n, l=l, parallel=gen_parallel)
        dist_mat = np.zeros([n, n])
        for i, j in combinations(range(n), 2):
            dist_mat[i, j] = lvl_manhhatan_div(levels[i], levels[j])
        res += crowdivs(dist_mat)
        print('%.5f, %.1fs' % (res / (t+1), time.time() - start_time))
    return res / trails

def test_rewards(designer, rfunc, path, name='', n=30, l=50, parallel=6, env_play_style='Runner'):
    env = make_vec_offrew_env(parallel, rfunc, N=designer.n, eplen=l, play_style=env_play_style)
    keys = [term.__class__.__name__ for term in rfunc.terms]
    keys.append('EpLength')
    results = []
    obs = env.reset()
    while len(results) < n:
        if designer is None:
            actions = sample_latvec()
        else:
            actions = designer.step(obs)
        next_obs, _, dones, infos = env.step(actions)
        del obs
        obs = next_obs
        for done, info in zip(dones, infos):
            if done:
                results.append([info[key] for key in keys])

    fname = name + '.csv' if name else 'test_rewards.csv'
    pds.DataFrame(results[:n], columns=keys).to_csv(getpath(f'{path}/{fname}'))

def test_level(str_level, n=1):
    level = MarioLevel(str_level)
    proxy = MarioProxy()
    fl_func, fg_func, p_func = LevelSACN(), GameplaySACN(), Playability()
    runner_simlt = MarioProxy.get_seg_infos(proxy.simulate_complete(level, MarioJavaAgents.Runner, segTimeK=120))
    killer_simlt = MarioProxy.get_seg_infos(proxy.simulate_complete(level, MarioJavaAgents.Killer, segTimeK=480))
    collector_simlt = MarioProxy.get_seg_infos(proxy.simulate_complete(level, MarioJavaAgents.Collector, segTimeK=480))
    segs = level.to_segs()
    return {
        'fL': np.mean(fl_func.compute_rewards(segs=segs, simlt_res=runner_simlt)[n-1:]),
        'fG-R': np.mean(fg_func.compute_rewards(segs=segs, simlt_res=runner_simlt)[n-1:]),
        'fG-K': np.mean(fg_func.compute_rewards(segs=segs, simlt_res=killer_simlt)[n-1:]),
        'fG-C': np.mean(fg_func.compute_rewards(segs=segs, simlt_res=collector_simlt)[n-1:]),
        'P': np.mean(p_func.compute_rewards(segs=segs, simlt_res=runner_simlt)[n-1:])
    }

def test_metrics(levels, dest_path, name='', parallel=5, n=5):
    # levels = generate_levels(designer, n=n, l=l)
    # proxy = MarioProxy()
    res = [] # {'fL': [], 'fG-R': [], 'fG-K': [], 'fG-C': [], 'P': []}
    # fl_func, fg_func, p_func = LevelSACN(), GameplaySACN(), Playability()
    computing_pool = MyAsyncPool(parallel)
    for level in levels:
        computing_pool.push(test_level, (str(level), n))
        while computing_pool.get_num_waiting() > 2 * parallel:
            res += computing_pool.collect()
            time.sleep(0.1)
    res += computing_pool.wait_and_get()
    if name:
        with open(getpath(dest_path, name), 'w') as f:
            json.dump(res, f)
    else:
        return res

def test_metrics4designers(designer, dest_path, name='', n=30, l=50, parallel=5):
    levels = generate_levels(designer, n=n, l=l)
    test_metrics(levels, dest_path, name, parallel, 5)

dpaths = ('exp_data/main/LGP_R', 'exp_data/main/LGP_C', 'exp_data/main/LGP_K')
playstyles = ('Runner', 'Killer', 'Collector')

def run_fun_metric_tests():
    for dpath, playstyle in zip(dpaths, playstyles):
        designer = Designer(dpath)
        test_metrics4designers(designer, dpath, 'metric_vals.json', 100)
        pass
    pass

def run_reward_tests():
    rfunc = RewardFunc(LevelSACN(), GameplaySACN(), Playability())
    n, p = 100, 5
    for dpath, playstyle in zip(dpaths, playstyles):
        if dpath == '{random}':
            designer = RandDesigner()
            test_rewards(designer, rfunc, 'exp_data/main', 'random_rewards', n=n, parallel=p)
        else:
            designer = Designer(dpath)
            test_rewards(designer, rfunc, dpath, env_play_style=playstyle, n=n, parallel=p)

def run_diversity_tests():
    for dpath, playstyle in zip(dpaths, playstyles):
        if dpath == '{random}':
            designer = RandDesigner()
            diversity = test_diversity(designer, gen_parallel=5)
            print('random', diversity)
            with open(getpath('exp_data/main/rand_div.json'), 'w') as f:
                json.dump(diversity, f)
        else:
            designer = Designer(dpath)
            diversity = test_diversity(designer, gen_parallel=5)
            with open(getpath(f'{dpath}/div.json'), 'w') as f:
                json.dump(diversity, f)
            print(dpath, diversity)


if __name__ == '__main__':
    d1 = Designer('exp_data/main/LGP_R')
    d2 = Designer('exp_data/main/LGP_K')
    d3 = Designer('exp_data/main/LGP_C')
    # generate_from_same((d1, d2, d2), ('Runner', 'Killer', 'Collector'), L=11, trials=100, path='lvls/double-check/part2-raw')
    for i, lvl in enumerate(generate_levels(d1, n=100, l=11)): lvl.save(f'lvls/no_same_start/Runner-{i}.lvl')
    for i, lvl in enumerate(generate_levels(d2, n=100, l=11)): lvl.save(f'lvls/no_same_start/Killer-{i}.lvl')
    for i, lvl in enumerate(generate_levels(d3, n=100, l=11)): lvl.save(f'lvls/no_same_start/Collector-{i}.lvl')

    # d1 = Designer('exp_data/main/LGP_C')
    # test_metrics(d1, 'exp_data/main/LGP_C', 'metric_vals.json', 30)
    # run_reward_tests()

    # run_fun_metric_tests()

    # test_metrics()
    pass

