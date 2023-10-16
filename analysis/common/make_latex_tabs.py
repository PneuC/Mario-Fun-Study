"""
  @Time : 2022/9/7 10:55 
  @Author : Ziqi Wang
  @File : make_latex_tabs.py 
"""
import json
import scipy.stats
import numpy as np
import pandas as pds
from scipy.stats import wilcoxon
from itertools import product
from src.utils.filesys import getpath
from src.utils.strings import float2fixlenstr


group = 'fin'
line_srcs = (
    f'lvls/part2-{group}/Runner-metricvals.json', f'lvls/part2-{group}/Killer-metricvals.json',
    f'lvls/part2-{group}/Collector-metricvals.json'
)
metrics = ['fL', 'fG-R', 'fG-K', 'fG-C', 'P']
line_tokens = ('R', 'K', 'C')


def make_train_res_tab():
    # tab_cols = ['fL', 'fG-R', 'fG-K', 'fG-C', 'R']
    # col_scales = [-4, -1, -1, -2]
    tab_data = []
    # foo = lambda a, s: '& ' + float2fixlenstr(a.mean(), 4, scale=s) + '$\pm$' + float2fixlenstr(a.std(), 3, scale=s)
    for linesrc in line_srcs:
        with open(getpath(linesrc), 'r') as f:
            data = json.load(f)
        # data = pds.read_csv(getpath(linesrc))
        # x = data[tab_cols].to_numpy()
        # x[:,0] = np.sqrt(-x[:,0])
        # x[:,3] = -x[:,3]
        # l = data['EpLength'].to_numpy()
        row_data = (
            np.mean([item['fL'] for item in data]), np.std([item['fL'] for item in data]),
            np.mean([item['fG-R'] for item in data]), np.std([item['fG-R'] for item in data]),
            np.mean([item['fG-K'] for item in data]), np.std([item['fG-K'] for item in data]),
            np.mean([item['fG-C'] for item in data]), np.std([item['fG-C'] for item in data]),
            np.mean([item['P'] for item in data]), np.std([item['fL'] for item in data]),
        )
        tab_data.append(row_data)
    # row_fmt = ' & '.join(['%.3f $\pm$ %.3f'] * 5)
    tmp = ['\multicolumn{3}{c|}{%.3f $\pm$ %.3f}'] * 4
    row_fmt = ' & '.join([*tmp, '\multicolumn{3}{c}{%.3f $\pm$ %.3f}'])
    for tk, row_data in zip(line_tokens, tab_data):
        content = (row_fmt % row_data).replace('0.', '.')
        print(f'{tk} &', content, r'\\')
    pass

def make_sd_tab():
    n = len(line_srcs)
    content = []
    data = []
    for linesrc in line_srcs:
        with open(getpath(linesrc), 'r') as f:
            data.append(json.load(f))
    for itemsA in data:
        content.append([])
        for metric, itemsB in product(metrics, data):
            samplesA = [item[metric] for item in itemsA]
            samplesB = [item[metric] for item in itemsB]
            try:
                _, pval = wilcoxon(samplesA, samplesB)
                if pval > 0.05:
                    content[-1].append(r'$\approx$')
                else:
                    content[-1].append('+' if np.mean(samplesA) > np.mean(samplesB) else '-')
            except ValueError:
                content[-1].append(r'$\times$')
    for item, tk in zip(content, line_tokens):
        print(tk, '&', ' & '.join(item), r'\\')

    pass

if __name__ == '__main__':
    # data_ = pds.read_csv(getpath('exp_data/main/random_rewards.csv'))
    # x = data_[['MeanDivergenceFun', 'LevelSACN', 'GameplaySACN', 'Playability']].to_numpy()
    # l = data_['EpLength'].to_numpy()
    # print(x)
    # print(l)
    # print(x / l.)
    # make_train_res_tab()
    make_sd_tab()
    pass

