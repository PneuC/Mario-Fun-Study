import json
import numpy as np
import pandas as pds
from matplotlib import pyplot as plt
from src.utils.filesys import getpath
from itertools import chain, repeat, cycle


def plot_participants_stats():
    data = pds.read_csv('questionnaire.csv')
    keys = ['Game Persona', 'Frequency of Playing Games (per Week)', 'Age', 'Gender']
    valuess = [
        ['Runner', 'Killer', 'Collector', 'Others'],
        ['Never', 'seldom', '1-3h per week', '3-10h', '>10h'],
        ['<20', '20-29', '30-39', '40-49', '50-59', '$\leq$60', 'secret'],
        ['Male', 'Female', 'secret']
    ]
    for i, (key, vals) in enumerate(zip(keys, valuess)):
        col = data[f'Q{i+1}']
        n = len(vals)
        y = [0] * n
        for opt in col.values:
            j = ord(opt) - ord('A')
            y[j] = y[j] + 1
        plt.figure(figsize=(4.2, 2.4), dpi=300)
        plt.bar(range(n), y)
        plt.title(key)
        plt.xticks(range(n), vals, rotation=45)
        plt.tight_layout()
        plt.savefig(f'./participants/Q{i+1}.png')

def plot_corr_bars(key, title='', ylim=(-1, 1.2)):
    up_x, up_y = [], []
    down_x, down_y = [], []
    dot_x, dot_y, dot_c = [], [], []
    cross_x, cross_y = [], []
    center_y, center_c = [], []

    data = pds.read_csv('./annt_examples.csv')
    # data.insert(len(data.columns) - 1, 'C', (data[f'A-{key}'].to_numpy() + data[f'B-{key}'].to_numpy()) / 2)
    pset = data[data['Anno'] == 'A'].append(data[data['Anno'] == 'B'])
    sk = [
        ln[f'A-{key}'] - ln[f'B-{key}'] if ln['Anno'] == 'A' else ln[f'B-{key}'] - ln[f'A-{key}']
        for _, ln in pset.iterrows()
    ]
    pset.insert(len(data.columns) - 1, 'DF', sk)
    pset = pset.sort_values('DF')
    plt.figure(figsize=(len(data) / 10, 2.4), dpi=200)
    for i, (_, ln) in enumerate(pset.iterrows()):
        fa, fb, annt = ln[f'A-{key}'], ln[f'B-{key}'], ln['Anno']
        if annt == 'A' and (fa > fb) or annt == 'B' and (fb > fa):
            up_x.append(i)
            up_y.append(max(fa, fb))
            dot_x.append(i)
            dot_y.append(min(fa, fb))
            dot_c.append('blue')
            center_c.append('blue')
            plt.plot([i, i], [fa, fb], color='blue')
        else:
            down_x.append(i)
            down_y.append(min(fa, fb))
            dot_x.append(i)
            dot_y.append(max(fa, fb))
            dot_c.append('red')
            center_c.append('red')
            plt.plot([i, i], [fa, fb], color='red')
    start = len(pset) + 3

    # eset, nset = data[data['Anno'] == 'E'], data[data['Anno'] == 'N']
    enset = data[data['Anno'] == 'E'].append(data[data['Anno'] == 'N'])
    # data.insert(len(data.columns) - 1, 'DF', (data[f'A-{key}'].to_numpy() - data[f'B-{key}'].to_numpy()))
    # data = data.sort_values('DF')
    enset.insert(len(enset.columns) - 1, 'C', (enset[f'A-{key}'].to_numpy() + enset[f'B-{key}'].to_numpy()) / 2)
    enset = enset.sort_values('C', ascending=False)
    for i, (_, ln) in enumerate(enset.iterrows()):
        x = i + start
        fa, fb, c, annt = ln[f'A-{key}'], ln[f'B-{key}'], ln['C'], ln['Anno']
        if annt == 'E':
            dot_x += [x, x]
            dot_y += [fa, fb]
            dot_c += ['green', 'green']
            center_c.append('green')
            plt.plot([x, x], [fa, fb], color='green')
        else:
            cross_x += [x, x]
            cross_y += [fa, fb]
            center_c.append('yellow')
            plt.plot([x, x], [fa, fb], color='yellow')
    plt.scatter(up_x, up_y, c='blue', marker='^', s=14)
    plt.scatter(down_x, down_y, c='red', marker='v', s=14)
    plt.scatter(dot_x, dot_y, c=dot_c, s=10)
    plt.scatter(cross_x, cross_y, c='yellow', marker='x', s=16)
    # plt.scatter(range(len(data)), center_y, c=center_c, marker='_', s=24, zorder=2)
    plt.xticks([])
    plt.xlim((-len(data) * 0.02, (len(data)+2) * 1.02))
    plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    pass

def plot_archive(granularity=0.05, n=30):
    def _foo(s=25, alpha=1.):
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(4, 4), dpi=256)
        plt.scatter(x, y, c='#2C73D2', s=s, alpha=alpha, linewidths=0)
        # plt.colorbar()
        plt.xticks([1-i*granularity for i in range(1, n)], [''] * (n-1))
        plt.yticks([1-i*granularity for i in range(1, n)], [''] * (n-1))
        plt.xlim((-0.5, 1))
        plt.ylim((-0.5, 1))
        plt.xlabel('$f_L$ value', fontsize=16)
        plt.ylabel('$f_G$ value', fontsize=16)
        plt.tight_layout(pad=0.4)
        plt.show()
    with open(getpath('lvls/rand_gen_lvls/archive.json'), 'r') as f:
        archive = json.load(f)
    metrictab = pds.read_csv(getpath('lvls/rand_gen_lvls/metric_vals.csv'), index_col='ID')
    x = metrictab['FL'].to_numpy()
    y = metrictab['FG'].to_numpy()
    # z = metrictab['DL'].to_numpy() + metrictab['DG'].to_numpy()
    u, v = np.where(x > -0.5)[0], np.where(y > -0.5)[0]
    indexes = list(set(u) and set(v))
    # x, y, z = x[indexes], y[indexes], z[indexes]
    x, y = x[indexes], y[indexes]
    _foo(alpha=0.4)
    x, y, z = [], [], []
    for lid in archive:
        if lid == '':
            continue
        row = metrictab.loc[lid]
        fl, fg, dsum = row['FL'], row['FG'], row['DL'] + row['DG']
        x.append(fl)
        y.append(fg)
        z.append(dsum)
    print(len(x))
    _foo()
    # plt.style.use('seaborn-whitegrid')
    # plt.figure(figsize=(5.4, 4.8), dpi=300)
    # plt.scatter(x, y, c=z, cmap='viridis')
    # plt.colorbar()
    # plt.xticks([1-i*0.1 for i in range(1, 15)], [''] * 14)
    # plt.yticks([1-i*0.1 for i in range(1, 15)], [''] * 14)
    # plt.xlim((-0.5, 1))
    # plt.ylim((-0.5, 1))
    # plt.tight_layout()
    # plt.show()



if __name__ == '__main__':
    plot_archive()
    # plot_corr_bars('cr', 'Completing Ratio', (0, 1.1))
    # plot_corr_bars('fl', '$F_L$', ylim=(-0.6, 1.1))
    # plot_corr_bars('fg', '$F_G$ (with true gameplay simlt_res)', ylim=(0.2, 1.1))
    # plot_archive()
    # l = [('a', 'b'), ('a', 'b'), ('a', 'b')]
    # for i, (a, b) in enumerate(l):
    #     print(i, a, b)
    # for item in enumerate(l):
    #     print(item)
    # for item in enumerate(chain(*repeat(l, 2))):
    #     print(item)
    pass
