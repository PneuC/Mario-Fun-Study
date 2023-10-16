import csv
import json
import numpy as np
import pandas as pds
from itertools import product
from src.smb.level import load_batch, MarioLevel
from src.smb.proxy import MarioProxy
from src.utils.filesys import getpath
from src.env.rfunc import LevelSACN, GameplaySACN


def test_metrics(*bids):
    def __foo(bid):
        lvls = load_batch(f'lvls/rand_gen_lvls/batch{bid}.lvls')
        fl, fg = LevelSACN(), GameplaySACN()
        simulator = MarioProxy()
        data = pds.read_csv(getpath('lvls/rand_gen_lvls/metric_vals.csv'))
        evaluated = set(data['ID'].to_list())
        f = open(getpath('lvls/rand_gen_lvls/metric_vals.csv'), 'a', newline='')
        wrtr = csv.writer(f)
        # wrtr.writerow(['ID', 'FL', 'FG', 'DL', 'DG', 'pad'])
        for i, lvl in enumerate(lvls):
            lid = f'{bid}-{i}'
            if lid in evaluated:
                continue
            segs = lvl.to_segs()
            simlt_res = MarioProxy.get_seg_infos(simulator.simulate_complete(lvl))
            fl_val = np.mean(fl.compute_rewards(segs=segs))
            fg_val = np.mean(fg.compute_rewards(segs=segs, simlt_res=simlt_res))
            dl_val, dg_val = fl.mean_div, fg.mean_div
            wrtr.writerow([lid, fl_val, fg_val, dl_val, dg_val, ''])
            f.flush()
        f.close()
    for i in bids:
        __foo(i)
    pass

# def make_archive(granularity=0.05, n=30, gl=0.1, gg=0.12):
#     def _find_cell(s:pds.DataFrame):
#         c = s[s['FL'] > flmin]
#         c = c[c['FL'] <= flmax]
#         c = c[c['FG'] > fgmin]
#         c = c[c['FG'] <= fgmax]
#         if not len(c): return None
#         return c.sort_values('DS').index[len(c) // 2]
#     gsum = gl + gg
#     data = pds.read_csv(getpath(f'lvls/rand_gen_lvls/metric_vals.csv'), index_col='ID')
#     data.insert(4, 'DS', data['DL'].to_numpy() + data['DG'].to_numpy())
#     pset = data[data['DS'] >= gsum]
#     nset = data[data['DS'] < gsum]
#     res = []
#     for i, j in product(range(n), range(n)):
#         flmin, flmax = 1 - (i+1) * granularity, 1 - i * granularity
#         fgmin, fgmax = 1 - (j+1) * granularity, 1 - j * granularity
#         a, b = _find_cell(pset), _find_cell(nset)
#         if a is not None:
#             res.append(a)
#         if b is not None:
#             res.append(b)
#         # candidates.sort()
#         # print(candidates)
#     # archive = [['', ''] for _ in range(n * n)]
#     # archive_dsum = [[gsum, gsum] for _ in range(n * n)]
#     # for token, row in data.iterrows():
#     #     print(row)
#     #     # row = data.loc[i]
#     #     fl, fg, dsum = row['FL'], row['FG'], row['DL'] + row['DG']
#     #     x, y = int((1 - fl) / granularity), int((1 - fg) / granularity)
#     #     if x < n and y < n:
#     #         p = x * n + y
#     #         if dsum > gsum and (archive[p][0] == '' or dsum > archive_dsum[p][0]):
#     #             archive[p][0] = token
#     #             archive_dsum[p][0] = gsum
#     #         if dsum < gsum and (archive[p][1] == '' or dsum < archive_dsum[p][1]):
#     #             archive[p][1] = token
#     #             archive_dsum[p][1] = gsum
#     print(len(res))
#     with open(getpath('lvls/rand_gen_lvls/archive.json'), 'w') as f:
#         json.dump(res, f)


def make_archive(granularity=0.05, n=30):
    def _find_cell():
        c = data[data['FL'] > flmin]
        c = c[c['FL'] <= flmax]
        c = c[c['FG'] > fgmin]
        c = c[c['FG'] <= fgmax]
        if not len(c): return None
        return c.sort_values('FS').index[-1]
    # gsum = gl + gg
    data = pds.read_csv(getpath(f'lvls/rand_gen_lvls/metric_vals.csv'), index_col='ID')
    data.insert(4, 'FS', data['FL'].to_numpy() + data['FG'].to_numpy())
    # pset = data[data['DS'] >= gsum]
    # nset = data[data['DS'] < gsum]
    res = []
    for i, j in product(range(n), range(n)):
        flmin, flmax = 1 - (i+1) * granularity, 1 - i * granularity
        fgmin, fgmax = 1 - (j+1) * granularity, 1 - j * granularity
        item = _find_cell()
        if item is not None:
            res.append(item)

    print(len(res))
    with open(getpath('lvls/rand_gen_lvls/archive.json'), 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    # test_metrics(3)
    make_archive()

    # with open(getpath('lvls/rand_gen_lvls/archive.json'), 'r') as f:
    #     archive = json.load(f)
    # data = pds.read_csv(getpath('lvls/rand_gen_lvls/metric_vals.csv'), index_col='ID')
    # # print(data.loc['0-9224'])
    # i = 0
    # for token in archive:
    #     row = data.loc[token]
    #     if 0.6 < row['FL'] <= 0.65 and 0.8 < row['FG'] <= 0.85:
    #         bid, lid = token.split('-')
    #         lvls = load_batch(f'lvls/rand_gen_lvls/batch{bid}.lvls')
    #         print(len(lvls))
    #         lvl = lvls[int(lid)]
    #         lvl.to_img(f'./l{i}.png')
    #         i += 1
    #     # a = MarioLevel.from_file('')
    pass

