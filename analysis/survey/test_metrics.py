import csv
import glob
import json
import os
import shutil
import time
from math import ceil
import numpy as np
import pandas as pds
from analysis.algo.analyse_designer import test_metrics
from src.smb.proxy import MarioProxy
from src.smb.level import MarioLevel, trace_div
from src.utils.filesys import getpath, load_dict_json
from analysis.survey.data_root import data_rt
from src.env.rfunc import LevelSACN, GameplaySACN
from itertools import combinations


qdf = pds.read_csv(getpath(data_rt, 'questionare.csv'), index_col='ID')
data = pds.read_csv(getpath(data_rt, 'annotation.csv'))
# proxy = MarioProxy()
fun_level = LevelSACN()
fun_gameplay = GameplaySACN()
entries = []
for _, row in data.iterrows():
    ID, lvlA, lvlB = row['ID'], row['A'], row['B']
    entries.append((ID, lvlA))
    entries.append((ID, lvlB))


def get_metric_vals(key, lname):
    lvl = MarioLevel.from_file(getpath(data_rt, 'levels', f'{lname}.lvl'))
    cplt_rate, trace = load_dict_json(
        getpath(data_rt, 'res', f'{key}_{lname}.json'),
        'completing-ratio', 'trace'
    )
    # trace_ns = load_dict_json(getpath(data_rt, 'res-nostop', f'{key}_{lname}.json'), 'trace')
    segs = lvl.to_segs()
    revision = (16 * 12 - 1) / (16 * 12)
    k = ceil(cplt_rate * revision * len(segs))
    segs = segs[:k]
    if k < 2:
        return None
    # simlt_res = MarioProxy.get_seg_infos(trace)
    # traces =
    simlt_res = MarioProxy.get_seg_infos({'full_trace': trace, 'restarts': []})
    # simlt_res_ns = MarioProxy.get_seg_infos({'full_trace': trace_ns, 'restarts': []})
    fl = np.mean(fun_level.compute_rewards(segs=segs))
    fg = np.mean(fun_gameplay.compute_rewards(segs=segs, simlt_res=simlt_res))
    # fg_ns = np.mean(fun_gameplay.compute_rewards(segs=segs, simlt_res=simlt_res_ns))
    return fl, fg

    # fg_norm = np.mean(fun_gameplay.compute_rewards(segs=segs, simlt_res=simlt_res, trace_size_norm=True))
    # fg_norm_ns = np.mean(fun_gameplay.compute_rewards(segs=segs, simlt_res=simlt_res_ns, trace_size_norm=True))
    # return fg_norm, fg_norm_ns

    pass


def get_agent_fg(token, lname, agent, **metric_args):
    fG_metric = GameplaySACN(**metric_args)
    if agent == 'true':
        true_trace = load_dict_json(getpath(data_rt, 'res', f'{token}_{lname}.json'), 'trace')
        simlt_res = MarioProxy.get_seg_infos({'full_trace': true_trace, 'restarts': []})
    else:
        with open(getpath(f'exp_data/agent_simlt_res/{agent}', f'{lname}.json'), 'r') as fp:
            simlt_res = json.load(fp)
    cplt_rate = load_dict_json(getpath(data_rt, 'res', f'{token}_{lname}.json'), 'completing-ratio')
    lvl = MarioLevel.from_file(getpath(data_rt, 'levels', f'{lname}.lvl'))
    segs = lvl.to_segs()
    revision = (16 * 12 - 1) / (16 * 12)
    k = ceil(cplt_rate * revision * len(segs))
    segs = segs[:k]
    simlt_res = simlt_res[:k]
    return np.mean(fG_metric.compute_rewards(segs=segs, simlt_res=simlt_res))


def add_agent_fg_vals():
    # fG-R, fG-K, fG-C, fG-A
    annt_persona_mapping = {'A': 'R', 'B': 'K', 'C': 'C', 'D': 'N/A'}
    mvals = pds.read_csv(getpath(data_rt, 'metric_vals-partial.csv'), index_col='ID')
    f = open(getpath(data_rt, 'metric_vals.csv'), 'w', newline='')
    wrtr = csv.writer(f)
    wrtr.writerow(['ID', 'fL', 'fG-T', 'fG-R', 'fG-K', 'fG-C', 'fG-A', ''])
    for ID, row in mvals.iterrows():
        old_vals = row['fL'], row['fG']
        key, lname = ID.split('_')
        if lname.find('lvl') < 0:
            continue
        try:
            persona = qdf['Q3'][key]
        except KeyError:
            continue
        fG_R, fG_K, fG_C = get_agent_fg(key, lname, 'Runner'), get_agent_fg(key, lname, 'Killer'), get_agent_fg(key, lname, 'Collector')
        if persona == 'A':
            fG_A = fG_R
        elif persona == 'B':
            fG_A = fG_K
        elif persona == 'C':
            fG_A = fG_C
        else:
            fG_A = fG_R
        wrtr.writerow([ID, *old_vals, fG_R, fG_K, fG_C, fG_A, ''])
        f.flush()
        # print(ID, *vals)
    pass

def compute_metric_vals2file():
    # mvals = pds.read_csv(getpath(data_rt, 'metric_vals-old.csv'), index_col='ID')
    f = open(getpath(data_rt, 'metric_vals-partial.csv'), 'w', newline='')
    wrtr = csv.writer(f)
    wrtr.writerow(['ID', 'fL', 'fG', ''])
    for item in glob.glob(getpath(data_rt, 'res', '*.json')):
        _, fname = os.path.split(item)
        gpid = fname[:-5]
        print(fname)
        # old_vals = row['fL'], row['fG'], row['fG-ns']
        key, lname = gpid.split('_')
        if lname.find('lvl') < 0:
            continue
        vals = get_metric_vals(key, lname)
        if vals is None:
            continue
        wrtr.writerow([gpid, *vals, ''])
        f.flush()
        print(gpid, *vals)
    f.close()

def test_part2_raw(parallel=5):
    def test_persona(personame):
        lvls = [MarioLevel.from_file(path) for path in glob.glob(getpath('lvls/part2-raw', f'{personame}*.lvl'))]
        test_metrics(lvls, 'lvls/part2-raw', f'{personame}-metricvals.json', parallel)

    test_persona('Runner')
    test_persona('Collector')
    test_persona('Killer')

def test_part2(parallel=5):
    def test_persona(personame):
        lvls = [MarioLevel.from_file(path) for path in glob.glob(getpath('lvls/part2-buffered', f'{personame}*.lvl'))]
        test_metrics(lvls, 'lvls/part2', f'{personame}-metricvals.json', parallel)
    test_persona('Runner')
    test_persona('Collector')
    test_persona('Killer')

def every_goal_div_fl(gvals):
    for g in gvals:
        if os.path.exists(getpath(data_rt, 'vary-goal-div-old', f'fL-{g:.3f}.csv')):
            shutil.copyfile(
                getpath(data_rt, 'vary-goal-div-old', f'fL-{g:.3f}.csv'),
                getpath(data_rt, 'vary-goal-div', f'fL-{g:.3f}.csv')
            )
            continue
        f = open(getpath(data_rt, 'vary-goal-div', f'fL-{g:.3f}.csv'), 'w', newline='')
        wrtr = csv.writer(f)
        wrtr.writerow(['ID', 'fval', ''])
        metric = LevelSACN(g=g)

        for pid, lid in entries:
            lvl = MarioLevel.from_file(getpath(data_rt, 'levels', f'lvl{lid}.lvl'))
            cplt_rate = load_dict_json(getpath(data_rt, 'res', f'{pid}_lvl{lid}.json'), 'completing-ratio')
            segs = lvl.to_segs()
            revision = (16 * 12 - 1) / (16 * 12)
            k = ceil(cplt_rate * revision * len(segs))
            segs = segs[:k]

            fl = np.mean(metric.compute_rewards(segs=segs))
            wrtr.writerow([f'{pid}_lvl{lid}', fl, ''])
        f.close()
    pass

def every_goal_div_fg(gvals):
    for g in gvals:
        # fg_tab = pds.read_csv(getpath(data_rt, 'vary-goal-div-old', f'fG-{g:.3f}.csv'), index_col='ID')
        # if os.path.exists(getpath(data_rt, 'vary-goal-div', f'fG-norm-{g:.3f}.csv')):
        #     shutil.copyfile(
        #         getpath(data_rt, 'vary-goal-div-old', f'fG-norm-{g:.3f}.csv'),
        #         getpath(data_rt, 'vary-goal-div', f'fG-norm-{g:.3f}.csv')
        #     )
        #     continue
        f = open(getpath(data_rt, 'vary-goal-div', f'fG-{g:.3f}.csv'), 'w', newline='')
        wrtr = csv.writer(f)
        wrtr.writerow(['ID', 'fG-T', 'fG-R', 'fG-K', 'fG-C', 'fG-A'])

        for token, lid in entries:
            lvl = MarioLevel.from_file(getpath(data_rt, 'levels', f'lvl{lid}.lvl'))
            cplt_rate, trace = load_dict_json(
                getpath(data_rt, 'res', f'{token}_lvl{lid}.json'),
                'completing-ratio', 'trace'
            )
            simlt_res = MarioProxy.get_seg_infos({'full_trace': trace, 'restarts': []})

            segs = lvl.to_segs()
            revision = (16 * 12 - 1) / (16 * 12)
            k = ceil(cplt_rate * revision * len(segs))
            segs = segs[:k]

            # fG_T = fg_tab['fG-T'][f'{token}_lvl{lid}']
            fG_T = get_agent_fg(token, f'lvl{lid}', 'true', g=g)
            fG_R = get_agent_fg(token, f'lvl{lid}', 'Runner', g=g)
            fG_K = get_agent_fg(token, f'lvl{lid}', 'Killer', g=g)
            fG_C = get_agent_fg(token, f'lvl{lid}', 'Collector', g=g)
            try:
                persona = qdf['Q3'][token]
            except KeyError:
                continue
            if persona == 'A':
                fG_A = fG_R
            elif persona == 'B':
                fG_A = fG_K
            elif persona == 'C':
                fG_A = fG_C
            else:
                fG_A = fG_R
            wrtr.writerow([f'{token}_lvl{lid}', fG_T, fG_R, fG_K, fG_C, fG_A])
            f.flush()
        f.close()

def compute_normalisation_ratio():
    df = pds.read_csv(getpath(data_rt, 'metric_vals.csv'), index_col='ID')
    agent_divs = []
    human_divs = []
    start = time.time()
    for ID, row in df.iterrows():
        print(time.time() - start)
        _, lname = ID.split('_')
        with open(getpath(f'exp_data/agent_simlt_res/Runner/{lname}.json'), 'r') as f:
            agent_traces = [item['trace'] for item in json.load(f)]

        human_trace, cplt_rate = load_dict_json(f'{data_rt}/res/{ID}.json', 'trace', 'completing-ratio')

        revision = (16 * 12 - 1) / (16 * 12)
        k = ceil(cplt_rate * revision * len(agent_traces))
        agent_traces = agent_traces[:k]
        human_traces, horizon = [[]], 256
        for p in human_trace:
            if p[0] > horizon:
                human_traces.append([])
            human_traces[-1].append([p[0] + 256 - horizon, p[1]])
        agent_divs += [trace_div(p, q) for p, q in combinations(agent_traces, 2)]
        human_divs += [trace_div(p, q) for p, q in combinations(human_traces, 2)]
    agent_mean, human_mean = np.mean(agent_divs), np.mean(human_divs)
    data = {'agent-mean': agent_mean, 'human-mean': human_mean, 'ratio': agent_mean/human_mean}
    print(data)
    with open(f'{data_rt}/mean_divergence.json', 'w') as f:
        json.dump(data, f)

# def test_randgen_fg(parallel=5):
#     bath_ids = ['0', '1']
#     computing_pool = MyAsyncPool(parallel)
#     fg_res = []
#     for ggval in ggs:
#         computing_pool.push(compute_fg_vals, (ggval,))
#         while computing_pool.get_num_waiting() > 2 * parallel:
#             fg_res += computing_pool.collect()
#             time.sleep(1)
#     fg_res += computing_pool.wait_and_get()
#     with open(getpath('./fg_statistics.json'), 'w') as f:
#         json.dump(fg_res, f)
#     for bid in bath_ids:
#         # for lvl, name in traverse_levels()
#         pass


if __name__ == '__main__':
    # write_metric_vals()
    # test_part2()
    # compute_metric_vals2file()
    # add_agent_fg_vals()
    # print(np.linspace(0.1, 0.25, 16))
    # print(np.linspace(0.6, 0.9, 16))
    # every_goal_div_fl(np.linspace(0.1, 0.6, 26))
    # every_goal_div_fg(np.linspace(0.1, 0.6, 26))
    # every_goal_div_fg(np.linspace(0.6, 0.9, 16))
    # every_goal_div_fg(np.linspace(4.1, 6, 20))

    # every_goal_div_fg([0.1])

    # print(qdf['Q7']['f2bb6e80-5d6a-47f7-9d4f-bffa766cefd6'])
    compute_normalisation_ratio()
    pass
