import glob
import json
import math
import os.path

import numpy as np
import pandas as pds
from analysis.survey.data_root import data_rt
from src.smb.level import MarioLevel, LevelRender, traverse_level_files
from src.smb.proxy import *
from src.utils.filesys import getpath, load_dict_json

def process_jsons(overwrite=False, remove_same=False):
    tar = 'res'
    if remove_same: tar = tar + '-nostop'
    for item in glob.glob(getpath(data_rt, 'data', 'jsons', '*')):
        folder, name = os.path.split(item[:-5])
        if not overwrite and os.path.exists(getpath(data_rt, tar, f'{name}.json')):
            continue
        try:
            with open(item, 'r') as f:
                json_data = json.load(f)
        except json.decoder.JSONDecodeError:
            print(item, 'ERROR')
        elements = json_data['elementData1']
        trace = [[elements[1]['marioX1'], elements[1]['marioY2']]]
        for i in range(2, len(json_data['elementData1'])):
            if elements[i] is None:
                break
            p = [elements[i]['marioX1'], elements[i]['marioY2']]
            if remove_same and p[0] == elements[i-1]['marioX1'] and p[1] == elements[i-1]['marioY2']:
                continue
            trace.append(p)
        completing_ratio = trace[-1][0] / ((MarioLevel.seg_width * 12 - 1) * LevelRender.tex_size)
        res = {'trace': trace, 'completing-ratio': completing_ratio}
        with open(getpath(data_rt, tar, f'{name}.json'), 'w') as f:
            json.dump(res, f)
            pass
    pass


def remove_incomplete_sessions():
    qdf = pds.read_csv(getpath(data_rt, 'questionare.csv'), index_col='ID')
    a1df = pds.read_csv(getpath(data_rt, 'annotation.csv'))
    a2df = pds.read_csv(getpath(data_rt, 'annotation2.csv'))
    qdf = qdf[~qdf.index.duplicated()]

    keys = set(qdf.index)
    complete_keys = {key for key in keys if len(a1df[a1df['ID'] == key]) >= 2 and len(a2df[a2df['ID'] == key]) >= 1}

    qdf.drop(keys - complete_keys).to_csv(getpath(data_rt, 'questionare.csv'))
    a1_removes = [idx for idx in a1df.index if a1df.loc[idx]['ID'] not in complete_keys]
    a2_removes = [idx for idx in a2df.index if a2df.loc[idx]['ID'] not in complete_keys]
    a1df.drop(a1_removes).to_csv(getpath(data_rt, 'annotation.csv'), index=False)
    a2df.drop(a2_removes).to_csv(getpath(data_rt, 'annotation2.csv'), index=False)
    # print(qdf)
    # print(a1df)
    # print(a2df)

# def remove_repfault_sessions():
#     qdf = pds.read_csv(getpath(data_rt, 'questionare.csv'), index_col='ID')
#     a1df = pds.read_csv(getpath(data_rt, 'annotation.csv'))
#     a2df = pds.read_csv(getpath(data_rt, 'annotation2.csv'))
#     keys = set(qdf.index)
#     for key in keys:
#         records1, records2 = a1df[a1df['ID'] == key], a2df[a2df['ID'] == key]
#         fault = False
#         for record in records1:
#
#             pass
#         print(records1)
#         print(records2)
#     pass

# def remove_short_games():
#     qdf = pds.read_csv(getpath(data_rt, 'questionare-raw.csv'), index_col='ID')
#     a1df = pds.read_csv(getpath(data_rt, 'annotation-raw.csv'))
#     a2df = pds.read_csv(getpath(data_rt, 'annotation2-raw.csv'))
#     keys = set(qdf.index)
#     invalid = {key: False for key in keys}
#     for item in os.listdir(getpath(data_rt, 'res')):
#         key, _ = item.split('_')
#         if key not in invalid.keys() or invalid[key]:
#             continue
#         if load_dict_json(os.path.join(data_rt, 'res', item), 'completing-ratio') < (1 / 12):
#             print(item)
#             invalid[key] = True
#     remove_keys = [key for key in keys if invalid[key]]
#     print(remove_keys)
#     print(qdf.drop(remove_keys))
#     print(a1df.drop(remove_keys))
#     print(a2df.drop(remove_keys))
#     # qdf.drop(remove_keys).to_csv(getpath(data_rt, 'questionare.csv'))
#     # a1df.drop(remove_keys).to_csv(getpath(data_rt, 'annotation.csv'), index=False)
#     # a2df.drop(remove_keys).to_csv(getpath(data_rt, 'annotation2.csv'), index=False)
#
#     # print(qdf.drop(remove_keys))
#     # print(a1df.drop(remove_keys))
#     # print(a2df.drop(remove_keys))


def get_participant_numbers():
    qdf_raw = pds.read_csv(getpath(data_rt, 'data', 'questionare.csv'), index_col='ID')
    a1df_raw = pds.read_csv(getpath(data_rt, 'data', 'annotation.csv'))
    a2df_raw = pds.read_csv(getpath(data_rt, 'data', 'annotation2.csv'))

    a1_IDs = set(a1df_raw['ID'].tolist())
    a2_IDs = set(a2df_raw['ID'].tolist())

    pass

def remove_short_games(threshold=1/12):
    qdf_raw = pds.read_csv(getpath(data_rt, 'data', 'questionare.csv'), index_col='ID')
    a1df_raw = pds.read_csv(getpath(data_rt, 'data', 'annotation.csv'))
    a2df_raw = pds.read_csv(getpath(data_rt, 'data', 'annotation2.csv'))

    a1_indexes = []
    a2_indexes = []
    valid_set1 = set()
    valid_set2 = set()
    for index, row in a1df_raw.iterrows():
        # print(ID)
        ID = row['ID']
        gpidA, gpidB = ID + '_lvl' + str(row['A']), ID + '_lvl' + str(row['B'])
        try:
            crA = load_dict_json(os.path.join(data_rt, 'res', f'{gpidA}.json'), 'completing-ratio')
            crB = load_dict_json(os.path.join(data_rt, 'res', f'{gpidB}.json'), 'completing-ratio')
            if crA > threshold and crB > threshold:
                a1_indexes.append(index)
                valid_set1.add(ID)
        except FileNotFoundError as e:
            print(e)
    print()
    for index, row in a2df_raw.iterrows():
        ID = row['ID']
        gpidR = ID + '_' + str(row['R'])
        gpidK = ID + '_' + str(row['K'])
        gpidC = ID + '_' + str(row['C'])
        # threshold = 1 / 12
        try:
            crR = load_dict_json(os.path.join(data_rt, 'res', f'{gpidR}.json'), 'completing-ratio')
            crK = load_dict_json(os.path.join(data_rt, 'res', f'{gpidK}.json'), 'completing-ratio')
            crC = load_dict_json(os.path.join(data_rt, 'res', f'{gpidC}.json'), 'completing-ratio')
            if crR > threshold and crK > threshold and crC > threshold:
                a2_indexes.append(index)
                valid_set2.add(ID)
        except FileNotFoundError as e:
            print(e)
    valid_set_q = valid_set1.union(valid_set2)
    print(len(valid_set1), len(valid_set2), len(valid_set_q))
    print(len(a1_indexes), len(a2_indexes), len(valid_set_q))
    qdf = qdf_raw.loc[list(valid_set_q)]
    a1df = a1df_raw.loc[a1_indexes]
    a2df = a2df_raw.loc[a2_indexes]
    # print(len(set(qdf.index)), len(set(a1df['ID'])), len(set(a2df['ID'])))
    # print(a1df_raw)
    # print(a1df)
    #
    # qdf.to_csv(getpath(data_rt, 'questionare.csv'))
    # a1df.to_csv(getpath(data_rt, 'annotation.csv'))
    # a2df.to_csv(getpath(data_rt, 'annotation2.csv'))

def simulate_agent_traces():
    proxy = MarioProxy()
    for lvl, name in traverse_level_files('exp_data/survey data/formal/lvls'):
        runner_res = MarioProxy.get_seg_infos(proxy.simulate_complete(lvl, MarioJavaAgents.Runner))
        with open(getpath('exp_data/survey data/formal/agent_simlt_res/Runner', f'{name}.json'), 'w') as f:
            json.dump(runner_res, f)
        killer_res = MarioProxy.get_seg_infos(proxy.simulate_complete(lvl, MarioJavaAgents.Killer))
        with open(getpath('exp_data/survey data/formal/agent_simlt_res/Killer', f'{name}.json'), 'w') as f:
            json.dump(killer_res, f)
        collector_res = MarioProxy.get_seg_infos(proxy.simulate_complete(lvl, MarioJavaAgents.Collector))
        with open(getpath('exp_data/survey data/formal/agent_simlt_res/Collector', f'{name}.json'), 'w') as f:
            json.dump(collector_res, f)

def append_with_skill_level(anntdf):
    # qdf = pds.read_csv(getpath(f'{data_rt}', 'questionare.csv'), index_col='ID')

    completions = {tk: [] for tk in set(anntdf['ID'].tolist())}
    for path in glob.glob(getpath(data_rt, 'res', '*.json')):
        _, fname = os.path.split(path)
        tk, lname = fname.split('_')
        # if 'lvl' not in lname:
        #     continue
        cr = load_dict_json(path, 'completing-ratio')
        if tk in completions.keys(): completions[tk].append(cr)
        pass
    mean_crs = []
    for tk in set(anntdf['ID'].tolist()):
    # for tk in anntdf['ID'].tolist():
        mean_cr = np.mean(completions[tk])
        mean_crs.append(mean_cr)
    # print(sorted(mean_crs))
    # print(len(mean_crs))
    # split = sorted(mean_crs)[-32]
    split = np.median(mean_crs)
    sklvs = []
    a, b = 0, 0
    np_a, nb_a, nn_a = 0, 0, 0
    np_b, nb_b, nn_b = 0, 0, 0
    for _, row in anntdf.iterrows():
        mean_cr = np.mean(completions[row['ID']])
        if mean_cr > split:
            sklv = 'A'
            a += 1
            if row['ANNT'] == 'E':
                nb_a += 1
            elif row['ANNT'] == 'N':
                nn_a += 1
            else:
                np_a += 1
        else:
            sklv = 'B'
            b += 1
            if row['ANNT'] == 'E':
                nb_b += 1
            elif row['ANNT'] == 'N':
                nn_b += 1
            else:
                np_b += 1
        sklvs.append(sklv)
    # print(a, b)
    # print(np_a, nb_a, nn_a)
    # print(np_b, nb_b, nn_b)
    anntdf.insert(0, 'skill-level', sklvs)

    pass


if __name__ == '__main__':
    # process_jsons(True, True)
    # process_jsons(True, False)
    # remove_incomplete_sessions()
    # remove_repfault_sessions()
    # remove_short_games(5/12)
    # simulate_agent_traces()

    append_with_skill_level(pds.read_csv(getpath(data_rt, 'annotation.csv')))

    pass


