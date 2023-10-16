"""
  @Time : 2022/3/23 22:51 
  @Author : Ziqi Wang
  @File : randgen_playable.py 
"""
import os
import json
import time
import argparse
from src.smb.level import lvlhcat, save_batch
from src.smb.proxy import MarioProxy
from src.gan.gankits import *
from src.utils.parallel import MyAsyncPool
from src.utils.datastruct import batched_iter


# repairer = DivideConquerRepairer()
def simulate(lvlstr):
    # repaired = repairer.repair(MarioLevel(lvlstr))
    simulator = MarioProxy()
    simlt_res = simulator.simulate_complete(MarioLevel(lvlstr))
    if not len(simlt_res['restarts']):
        simlt = {'full_trace': simlt_res['full_trace']}
        simlt['seg_infos'] = MarioProxy.get_seg_infos(simlt_res)
        return lvlstr, simlt
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str, default='models/generator.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--l', type=int, default=11)
    parser.add_argument('--file_batch', type=int, default=100)
    # parser.add_argument('--space_size', type=float, default=1.)
    parser.add_argument('--parallel', type=int, default=32)
    parser.add_argument('--max_trials', type=int, default=10000)
    parser.add_argument('--res_path', type=str, default='analysis/fun_tuning/randgen_lvls')

    args = parser.parse_args()
    generator = get_generator(args.generator_path, args.device)

    simulate_pool = MyAsyncPool(args.parallel)

    n = 0
    results = []
    while len(results) < args.n and n < args.max_trials:
        latvecs = [sample_latvec(args.l, device=args.device) for _ in range(args.parallel)]
        lvls = [lvlhcat(process_onehot(generator(item))) for item in latvecs]
        # for _ in range(1, args.l):
        #     segs = process_levels(generator(sample_latvec(args.parallel, args.space_size * 2, args.device)), True)
        #     for i, seg in enumerate(segs):
        #         lvls[i] = lvls[i] + seg
        tmp = simulate_pool.collect()
        for lvl, latvec in zip(lvls, latvecs):
            simulate_pool.push(simulate, (str(lvl),))
        while simulate_pool.get_num_waiting() > 2 * args.parallel:
            time.sleep(1)
            tmp += simulate_pool.collect()
        for trial_res in tmp:
            if trial_res is not None:
                results.append(trial_res)
        n += len(tmp)
        n_digitals = len(str(args.max_trials))
        fmt = f'%0{n_digitals}d/%0{n_digitals}d-%0{n_digitals}d/{args.max_trials}'
        print(fmt % (len(results), args.n, n))

    simulate_pool.terminate()

    os.makedirs(getpath(args.res_path), exist_ok=True)
    for i, (batch, _) in enumerate(batched_iter(results[:args.n], args.file_batch)):
        batch_lvls = [item[0] for item in batch]
        batch_simlt_res = [item[1] for item in batch]
        # batch_latvec = [item[2] for item in batch]
        # file_content = ';\n'.join(batch_lvls)
        # with open(getpath(f'{args.res_path}/batch{i}.smblvs'), 'w') as f:
        #     f.write(file_content)
        save_batch(batch_lvls, f'{args.res_path}/batch{i}.smblvs')
        with open(getpath(f'{args.res_path}/batch{i}_simlt_res.json'), 'w') as f:
            json.dump(batch_simlt_res, f)
        # with open(getpath(f'{args.res_path}/batch{i}_latvecs.json'), 'w') as f:
        #     json.dump(batch_latvec, f)
