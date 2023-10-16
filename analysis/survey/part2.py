import matplotlib
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from scipy import stats

from analysis.survey.data_root import data_rt
from src.utils.filesys import getpath, load_dict_json

qdf = pds.read_csv(getpath(data_rt, 'questionare.csv'))
annt_persona_mapping = {'A': 'R', 'B': 'K', 'C': 'C', 'D': 'N/A'}
persona_annt_mapping = {'R': 'A', 'K': 'B', 'C': 'C', 'N/A': 'D'}
personas = {row['ID']: annt_persona_mapping[row['Q3']] for _, row in qdf.iterrows()}

dt = pds.read_csv(getpath(data_rt, 'annotation2.csv'))
dt.insert(0, column='persona', value=[personas[row['ID']] for _, row in dt.iterrows()])
# matplotlib.rcParams['font.family'] = ['Family1', 'serif', 'Family2']

# plt.style.use('ggplot')

def list_find(arr, val):
    i = 0
    while arr[i] != val:
        i += 1
    return i + 1
    pass

def get_subdt(annt_persona):
    if annt_persona == 'All':
        return dt
    return dt[dt['persona'] == annt_persona]
    pass

def average_rank(annt_persona, agent):
    subdt = get_subdt(annt_persona)
    rank_list = []
    for _, row in subdt.iterrows():
        order = [row['R1'], row['R2'], row['R3']]
        rank = list_find(order, agent)
        rank_list.append(rank)
    return np.mean(rank_list), np.std(rank_list)
    pass

def cr(annt_persona, agent):
    subdt = get_subdt(annt_persona)
    crs, lives = [], []
    for _, row in subdt.iterrows():
        tk, lid = row['ID'], row[agent]
        # with open(getpath('exp_data/survey data/formal/res/{tk}_{lid}.json', f'{tk}_{lid}.json'), 'r') as f:
        path = f'exp_data/survey data/formal/res-reproduce/{tk}_{lid}.json'
        completing_ratio, live = load_dict_json(path, 'completing-ratio', 'lives')
        crs.append(completing_ratio)
        lives.append(live)
    return np.mean(crs) * 100, np.mean(lives)


def kendall(annt_persona, agent):
    if annt_persona == 'All':
        subdt = dt
    else:
        subdt = get_subdt(annt_persona)
    rank_list = []
    scores = [1, 0, -1]
    s = 0
    for _, row in subdt.iterrows():
        order = [row['R1'], row['R2'], row['R3']]
        rank = list_find(order, agent)
        s += scores[rank - 1]
    return s / len(subdt)
    pass

def ranksum(annt_persona, a1, a2):
    if annt_persona == 'All':
        subdt = dt
    else:
        subdt = get_subdt(annt_persona)
    r1, r2 = [], []

    for _, row in subdt.iterrows():
        order = [row['R1'], row['R2'], row['R3']]
        r1.append(list_find(order, a1))
        r2.append(list_find(order, a2))

    return stats.ranksums(r1, r2).pvalue
    pass

def print_latex_tab():
    def _sdc(_p, _a):
        sd = []
        v = kendall(_p, _a)
        score = '+%.3f' % v if v > 0 else '%.3f' % v
        p_values = (ranksum(_p, _a, 'r'), ranksum(_p, _a, 'k'), ranksum(_p, _a, 'c'))
        for _ap, p_values in zip(('r', 'k', 'c'), p_values):
            if _ap == _a:
                sd.append(r'\times')
            elif p_values < 0.05:
                if kendall(_p, _a) > kendall(_p, _ap):
                    sd.append('+')
                else:
                    sd.append('-')
            else:
                sd.append(r'\approx')
        _content = '~'.join(sd)
        return f'${_content}$'
    row_fmt = ' & '.join(['%.1f\\%% & %.2f'] * 5)
    print('  R &', row_fmt % (*cr('R', 'R'), *cr('K', 'R'), *cr('C', 'R'), *cr('N/A', 'R'), *cr('All', 'R')), r'\\')
    print('  K &', row_fmt % (*cr('R', 'K'), *cr('K', 'K'), *cr('C', 'K'), *cr('N/A', 'K'), *cr('All', 'K')), r'\\')
    print('  C &', row_fmt % (*cr('R', 'C'), *cr('K', 'C'), *cr('C', 'C'), *cr('N/A', 'C'), *cr('All', 'C')), r'\\')

    print('\n')

    row_fmt = ' & '.join(['%.3f'] * 5)
    print('  Runner &', row_fmt % (kendall('R', 'r'), kendall('K', 'r'), kendall('C', 'r'), kendall('N/A', 'r'), kendall('All', 'r')), r'\\')
    print('  Killer &', row_fmt % (kendall('R', 'k'), kendall('K', 'k'), kendall('C', 'k'), kendall('N/A', 'k'), kendall('All', 'k')), r'\\')
    print('  Collector &', row_fmt % (kendall('R', 'c'), kendall('K', 'c'), kendall('C', 'c'), kendall('N/A', 'c'), kendall('All', 'c')), r'\\')
    print('\n')
    print('  Runner &', ' & '.join([_sdc('R', 'r'), _sdc('K', 'r'), _sdc('C', 'r'), _sdc('N/A', 'r'), _sdc('All', 'r')]), r'\\')
    print('  Killer &', ' & '.join([_sdc('R', 'k'), _sdc('K', 'k'), _sdc('C', 'k'), _sdc('N/A', 'k'), _sdc('All', 'k')]), r'\\')
    print('  Collector &', ' & '.join([_sdc('R', 'c'), _sdc('K', 'c'), _sdc('C', 'c'), _sdc('N/A', 'c'), _sdc('All', 'c')]), r'\\')


    pass

def plot_persona_bar(persona, save_path=''):
    title_mapping = {'R': 'Runner', 'K': 'Killer', 'C': 'Collector', 'N/A': 'N/A', 'All': 'All'}
    title = title_mapping[persona]
    subdt = get_subdt(persona)
    rank_list = {'r': [], 'k': [], 'c': []}

    for _, row in subdt.iterrows():
        r1, r2, r3 = row['R1'], row['R2'], row['R3']
        rank_list[r1].append(1)
        rank_list[r2].append(2)
        rank_list[r3].append(3)
        # order = [row['R1'], row['R2'], row['R3']]
        # rank = list_find(order, agent)
        # rank_list.append(rank)
    plt.figure(figsize=(4, 2), dpi=256)

    num_r, num_k, num_c = len(rank_list['r']), len(rank_list['k']), len(rank_list['c'])
    up = max(num_r, num_k, num_c) + 1

    runner_rank_freqs = [rank_list['r'].count(1), rank_list['r'].count(2), rank_list['r'].count(3)]
    killer_rank_freqs = [rank_list['k'].count(1), rank_list['k'].count(2), rank_list['k'].count(3)]
    collector_rank_freqs = [rank_list['c'].count(1), rank_list['c'].count(2), rank_list['c'].count(3)]
    X = np.array([0, 1, 2])
    Y1 = [runner_rank_freqs[0], killer_rank_freqs[0], collector_rank_freqs[0]]
    Y2 = [runner_rank_freqs[1], killer_rank_freqs[1], collector_rank_freqs[1]]
    Y3 = [runner_rank_freqs[2], killer_rank_freqs[2], collector_rank_freqs[2]]

    plt.bar(X + 0.15, Y1, width=0.3, label='rank=1', color='#FFC750')
    plt.bar(X + 0.45, Y2, width=0.3, label='rank=2', color='#FF4F8D')
    plt.bar(X + 0.75, Y3, width=0.3, label='rank=3', color='#2C73D2')
    plt.title(title + ' players')
    plt.xticks([0.45, 1.45, 2.45], ['R', 'K', 'C'])
    plt.xlabel('Generator')

    plt.ylim((0, 1.2 * up))
    plt.ylabel('Count of rank')
    plt.legend(ncol=3, loc='upper center', framealpha=0.0, columnspacing=1.2, labelspacing=0.3)
    plt.tight_layout(pad=0.5)

    if not save_path:
        plt.show()
    else:
        plt.savefig(getpath(save_path))
    pass


if __name__ == '__main__':
    # print(len(get_subdt('R')))
    # print(len(get_subdt('K')))
    # print(len(get_subdt('C')))
    # print(len(get_subdt('N/A')))

    # print(average_rank('R', 'r'))
    # print(average_rank('R', 'k'))
    # print(average_rank('R', 'c'))
    # print(average_rank('R', 'r'))

    print_latex_tab()

    # plot_persona_bar('R', 'analysis/survey/results/runner-part2-bars.png')
    # plot_persona_bar('K', 'analysis/survey/results/killer-part2-bars.png')
    # plot_persona_bar('C', 'analysis/survey/results/collector-part2-bars.png')
    # plot_persona_bar('N/A', 'analysis/survey/results/NA-part2-bars.png')
    # plot_persona_bar('All', 'analysis/survey/results/All-part2-bars.png')
    pass

