import glob
import os.path
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
# import PIL
from PIL import Image
from enum import Enum

from analysis.survey.preprocess import append_with_skill_level
from src.utils.filesys import getpath
from analysis.survey.data_root import data_rt



dt = pds.read_csv(getpath(data_rt, 'annotation.csv'))
mv = pds.read_csv(getpath(data_rt, 'metric_vals.csv'), index_col='ID')
append_with_skill_level(dt)

class AnntTypes(Enum):
    PREF = 'P'
    EQUA = 'E'
    NITH = 'N'
    INCMP = 'I'
    ALL = 'A'


def get_subdt(anntype):
    if anntype == AnntTypes.ALL:
        subdt = dt
    elif anntype == AnntTypes.PREF:
        subdt = pds.concat([dt[dt['ANNT'] == 'A'], dt[dt['ANNT'] == 'B']])
    elif anntype == AnntTypes.INCMP:
        subdt = pds.concat([dt[dt['ANNT'] == 'E'], dt[dt['ANNT'] == 'N']])
    else:
        subdt = dt[dt['ANNT'] == anntype.value]
    return subdt

count = {t: len(get_subdt(t)) for t in AnntTypes}

# def num_records(anntype):
#     if anntype == AnntTypes.PREF:
#         return len(dt[dt['ANNT'] == 'A']) + len(dt[dt['ANNT'] == 'B'])
#     elif anntype == AnntTypes.EQUA:
#         return len(dt[dt['ANNT'] == 'E'])
#     elif anntype == AnntTypes.NITH:
#         return len(dt[dt['ANNT'] == 'N'])
#     elif anntype == AnntTypes.ALL:
#         return len(dt)
#     pass

def avg_diff(anntype, metric):
    subdt = get_subdt(anntype)
    # vals =  [row[metric] for _, row in subdt.iterrows()]
    fAs, fBs = [], []
    for _, row in subdt.iterrows():
        # print(row)
        key = row['ID']
        gpidA, gpidB = row['A'], row['B']
        fAs.append(mv[metric][f'{key}_lvl{gpidA}'])
        fBs.append(mv[metric][f'{key}_lvl{gpidB}'])
    avgs = 0.5 * (np.array(fAs) + np.array(fBs))
    diffs = np.abs(np.array(fAs) - np.array(fBs))
    return np.mean(avgs), np.std(avgs), np.mean(diffs), np.std(diffs)

def get_agreement(metric):
    subdt = get_subdt(AnntTypes.PREF)
    summation = 0
    n = 0
    for _, row in subdt.iterrows():
        key = row['ID']
        annt = row['ANNT']
        gpidA, gpidB = row['A'], row['B']
        fA, fB = mv[metric][f'{key}_lvl{gpidA}'], mv[metric][f'{key}_lvl{gpidB}']
        summation += (1 if fA > fB and annt == 'A' or fB > fA and annt == 'B' else -1)
        n += 1
    return summation / n

def get_agree_count(metric):
    global mv
    subdt = get_subdt(AnntTypes.PREF)
    summation = 0
    n = 0
    # bvec = []
    for _, row in subdt.iterrows():
        key = row['ID']
        annt = row['ANNT']
        gpidA, gpidB = row['A'], row['B']
        fA, fB = mv[metric][f'{key}_lvl{gpidA}'], mv[metric][f'{key}_lvl{gpidB}']
        summation += (1 if fA > fB and annt == 'A' or fB > fA and annt == 'B' else 0)
        n += 1
        # if metric == 'fG-T':
        #     bvec.append(1 if fA > fB and annt == 'A' or fB > fA and annt == 'B' else 0)
    # if metric == 'fG-T':
    #     print(bvec)
    #     print(mv[metric])
    #     print(summation)
    return summation, n

def print_latex_tab():
    def __get(anntype):
        fL_avg_avg, fL_avg_std, fL_diff_avg, fL_diff_std = avg_diff(anntype, 'fL')
        fG_avg_avg, fG_avg_std, fG_diff_avg, fG_diff_std = avg_diff(anntype, 'fG-ns')
        return [
            fL_avg_avg, fL_avg_std, fG_avg_avg, fG_avg_std,
            fL_diff_avg, fL_diff_std, fG_diff_avg, fG_diff_std
        ]

    data_fmt = '%d & %.3g$\pm$%.2g & %.3g$\pm$%.2g & %.3g$\pm$%.2g & %.3g$\pm$%.2g'

    tmp = __get(AnntTypes.PREF) + [get_agreement('fL'), get_agreement('fG-ns')]
    row1 = ('  P & ' + data_fmt +  r' & %.3f & %.3f') % (count[AnntTypes.PREF], *tmp)
    row2 = ('  EN & ' + data_fmt +  ' & -/- & -/-') % (count[AnntTypes.EQUA], *__get(AnntTypes.EQUA))
    # row3 = ('  N & ' + data_fmt +  ' & -/- & -/-') % (count[AnntTypes.NITH], *__get(AnntTypes.NITH))
    row4 = ('  All & ' + data_fmt +  ' & -/- & -/-') % (count[AnntTypes.ALL], *__get(AnntTypes.ALL))
    print(row1, r'\\')
    print(row2, r'\\')
    # print(row3, r'\\')
    print(r'  \hline')
    print(row4, r'\\')
    pass

def plot_metrics_wrt_gvals(name, metric, notation):
    global mv
    gvals = []
    p_avgs, p_diffs = [], []
    e_avgs, e_diffs = [], []
    n_avgs, n_diffs = [], []
    agrees = []
    # g, p_avg, p_diff, e_avg, e_diff, n_avg, n_diff, agree
    res = []
    for mvpath in glob.glob(getpath(data_rt, 'vary-goal-div', f'{name}*.csv')):
        mv = pds.read_csv(mvpath, index_col='ID')
        # print(mvpath)
        item = [float(os.path.split(mvpath)[1][-9:-4])]
        # gvals.append(float(os.path.split(mvpath)[1][4:-4]))
        avg, _, diff, _ = avg_diff(AnntTypes.PREF, metric)
        # p_avgs.append(avg)
        # p_diffs.append(diff)
        item += [avg, diff]
        avg, _, diff, _ = avg_diff(AnntTypes.EQUA, metric)
        # e_avgs.append(avg)
        # e_diffs.append(diff)
        item += [avg, diff]
        avg, _, diff, _ = avg_diff(AnntTypes.NITH, metric)
        # n_avgs.append(avg)
        # n_diffs.append(diff)
        item += [avg, diff]
        agree = get_agreement(metric)
        # agrees.append(agree)
        item.append(agree)
        agree_cnt, n = get_agree_count(metric)
        # n_avgs.append(avg)
        # n_diffs.append(diff)
        item += [agree_cnt, n]
        res.append(item)

    x_ticks = np.linspace(0.1, 0.3, 5) if name == 'fL' else np.linspace(0, 6, 7)
    res.sort(key=lambda x:x[0])
    res = np.array(res)

    # plt.style.use('seaborn')
    # plt.figure(figsize=(3,2), dpi=256)
    # plt.plot(res[:, 0], res[:, 1], label='P')
    # plt.plot(res[:, 0], res[:, 3], label='E')
    # plt.plot(res[:, 0], res[:, 5], label='N')
    # plt.xticks(x_ticks)
    # if name == 'fL':
    #     plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.20', '0.40', '0.60', '0.80', '1.00'])
    # else:
    #     plt.yticks([-9, -7, -5, -3, -1, 1], ['-9.0', '-7.0', '-5.0', '-3.0', '-1.0', '1.0'])
    # plt.legend()
    # plt.title(f'${notation}$-Average')
    # plt.tight_layout(pad=0.5)
    # plt.show()
    #
    # plt.figure(figsize=(3,2), dpi=256)
    # plt.plot(res[:, 0], res[:, 2], label='P')
    # plt.plot(res[:, 0], res[:, 4], label='E')
    # plt.plot(res[:, 0], res[:, 6], label='N')
    # plt.xticks(x_ticks)
    # if name == 'fL':
    #     plt.yticks([0.0, 0.1, 0.2, 0.3], ['0.00', '0.10', '0.20', '0.30'])
    # else:
    #     plt.yticks([0, 2, 4, 6], ['0.00', '2.00', '4.00', '6.00'])
    # plt.legend()
    # plt.title(f'${notation}$-Difference')
    # plt.tight_layout(pad=0.5)
    # plt.show()

    plt.figure(figsize=(3,2), dpi=256)
    plt.plot(res[:, 0], res[:, 8])
    plt.plot([min(res[:, 0]), max(res[:, 0])], [res[0, 9] / 2, res[0, 9] / 2], ls='--')
    plt.xticks(x_ticks)
    plt.yticks([0, 10, 20, 30, 40])
    plt.title(f'Agreement count')
    plt.tight_layout(pad=0.2)
    plt.show()

def plot_agree_gval_curves():
    gvals = []
    def __foo(name, metric):
        global mv
        res = []
        for mvpath in glob.glob(getpath(data_rt, 'vary-goal-div', f'{name}*.csv')):
            mv = pds.read_csv(mvpath, index_col='ID')
            agree_cnt, n = get_agree_count(metric)
            tau = (2 * agree_cnt - n) / n
            # res.append([float(os.path.split(mvpath)[1][-9:-4]), agree_cnt, n])
            res.append([float(os.path.split(mvpath)[1][-9:-4]), tau, n])
        res.sort(key=lambda x: x[0])
        return np.array(res)

    x_ticks = np.linspace(0.1, 0.6, 6)
    # gl_data = __foo('fL', 'fval')
    # gg_data = __foo('fG-norm', 'fg')
    fl_data = __foo('fL', 'fval')
    fg_t_data = __foo('fG', 'fG-T')
    fg_r_data = __foo('fG', 'fG-R')
    print(fl_data[:, 1].max(), fg_t_data[:, 1].max(), fg_r_data[:, 1].max())
    # fg_k_data = __foo('fG', 'fG-K')
    # fg_c_data = __foo('fG', 'fG-C')
    # fg_a_data = __foo('fG', 'fG-A')
    # print(fg_t_data)
    plt.style.use('seaborn-v0_8-dark-palette')
    plt.figure(figsize=(2.5,2.5), dpi=256)
    plt.plot([0.1, 0.6], [0, 0], ls='--', color='#808080')
    # plt.plot(gl_data[:, 0], gl_data[:, 1], color='#FF4F8D', label='$f_L$')
    # plt.plot(gg_data[:, 0], gg_data[:, 1], color='#2C73D2', label='$f_G$')
    plt.plot(fl_data[:, 0], fl_data[:, 1], label='$f_L$')
    plt.plot(fg_t_data[:, 0], fg_t_data[:, 1], label='$f_G^H$')
    plt.plot(fg_r_data[:, 0], fg_r_data[:, 1], label='$f_G^R$')
    # plt.plot(fg_k_data[:, 0], fg_k_data[:, 1], label='$f_G^K$')
    # plt.plot(fg_c_data[:, 0], fg_c_data[:, 1], label='$f_G^C$')
    # plt.plot(fg_a_data[:, 0], fg_a_data[:, 1], label='$f_G^A$')
    plt.xticks(x_ticks)
    plt.ylim((-0.35, 0.35))
    # plt.xlim((0.075, 0.625))
    plt.xlabel('$g_L$ and $g_G$ value')
    plt.ylabel(r'$\tau$ coefficient')
    # plt.yticks(range(0, 91, 10))
    # plt.title(f'Count of agreement records')
    plt.legend(ncol=3, loc='lower center', columnspacing=1.0)
    plt.tight_layout(pad=0.2)
    plt.show()

def plot_agree_gval_heat():
    agree_map = np.zeros([26, 26])
    # global mv
    glvals = np.linspace(0.1, 0.6, 26)
    ggvals = np.linspace(0.1, 0.6, 26)
    subdt = get_subdt(AnntTypes.PREF)
    for i, j in product(range(26), range(26)):
        gl, gg = glvals[i], ggvals[j]
        fL = pds.read_csv(getpath(data_rt, 'vary-goal-div', f'fL-{gl:.3f}.csv'), index_col='ID')
        fG = pds.read_csv(getpath(data_rt, 'vary-goal-div', f'fG-{gg:.3f}.csv'), index_col='ID')
        agree_cnts = 0
        for _, row in subdt.iterrows():
            key = row['ID']
            annt = row['ANNT']
            gpidA, gpidB = row['A'], row['B']
            fA = fL['fval'][f'{key}_lvl{gpidA}'] + fG['fG-R'][f'{key}_lvl{gpidA}']
            fB = fL['fval'][f'{key}_lvl{gpidB}'] + fG['fG-R'][f'{key}_lvl{gpidB}']
            agree_cnts += (1 if fA > fB and annt == 'A' or fB > fA and annt == 'B' else 0)
        agree_map[-i-1, j] = agree_cnts
    agree_map = (2 * agree_map - len(subdt)) / len(subdt)

    print(agree_map[-3, 10])
    min_agree, max_agree = np.min(agree_map), np.max(agree_map)
    print(min_agree, max_agree)
    print(np.where(agree_map == min_agree))
    print(np.where(agree_map == max_agree))
    # plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(3.75, 3), dpi=512)
    plt.imshow(agree_map, vmin=min_agree, vmax=max_agree, cmap='spring', interpolation='none', aspect='auto')
    plt.xticks(range(0, 26, 5), map(lambda x: '%.2f' % x, np.linspace(0.1, 0.6, 6))),
    plt.yticks(range(25, -1, -5), map(lambda x: '%.2f' % x, np.linspace(0.1, 0.6, 6)))
    plt.xlabel('$g_G$ value')
    plt.ylabel('$g_L$ value')
    plt.colorbar()
    plt.tight_layout(pad=0.2)
    # plt.grid(color='black', alpha=1.0)
    plt.show()

def plot_preference_records(metric, ylim, yticks, notation='', skill_lvl='', title='', save_path=''):
    clr_agree = '#FF4F8D' # '#764900'
    clr_disagree = '#0077FF' # '#452BE1'
    clr_equal = '#009474' # '#009763'
    clr_neither = '#8B51F8' # '#CA0AC2'

    # data = pds.read_csv(getpath(data_rt, 'annotation.csv'))
    # metadata = pds.read_csv(getpath(data_rt, 'questionare.csv'), index_col='ID')
    data = dt[dt['skill-level'] == skill_lvl] if skill_lvl else dt
    mvals = pds.read_csv(getpath(data_rt, 'metric_vals.csv'), index_col='ID')

    if metric[:5] == 'mixed':
        t = metric[-1]
        v = lambda a, b: mvals['fL']['%s_lvl%s' % (a['ID'], a[b])] + mvals[f'fG-{t}']['%s_lvl%s' % (a['ID'], a[b])]
    else:
        v = lambda a, b: mvals[metric]['%s_lvl%s' % (a['ID'], a[b])]

    pset = data[data['ANNT'] == 'A'].append(data[data['ANNT'] == 'B'])
    eset = data[data['ANNT'] == 'E'].append(data[data['ANNT'] == 'N'])
    wr = [len(pset) / len(data), len(eset) / len(data)]
    plt.style.use('seaborn-v0_8-dark-palette')
    fig, ax = plt.subplots(1, 2, figsize=((len(data)+4) / 10, 1.75), dpi=256, sharey='row', width_ratios=wr)

    sk = [
        v(ln, 'A') - v(ln, 'B') if ln['ANNT'] == 'A' else v(ln, 'B') - v(ln, 'A')
        for _, ln in pset.iterrows()
    ]
    pset.insert(len(data.columns) - 1, 'DF', sk)
    pset = pset.sort_values('DF')

    up_x, up_y = [], []
    down_x, down_y = [], []
    dot_x, dot_y, dot_c = [], [], []

    agree, disagree = 0, 0
    for i, (_, ln) in enumerate(pset.iterrows()):
        fa, fb = v(ln, 'A'), v(ln, 'B')
        annt = ln['ANNT']
        if annt == 'A' and (fa > fb) or annt == 'B' and (fb > fa):
            up_x.append(i)
            up_y.append(max(fa, fb))
            dot_x.append(i)
            dot_y.append(min(fa, fb))
            dot_c.append(clr_agree)
            ax[0].plot([i, i], [fa, fb], color=clr_agree)
            agree += 1
        else:
            down_x.append(i)
            down_y.append(min(fa, fb))
            dot_x.append(i)
            dot_y.append(max(fa, fb))
            dot_c.append(clr_disagree)
            ax[0].plot([i, i], [fa, fb], color=clr_disagree)
            disagree += 1

    ax[0].scatter(down_x, down_y, c=clr_disagree, marker='v', s=14)
    ax[0].scatter(up_x, up_y, c=clr_agree, marker='^', s=14)
    ax[0].scatter(dot_x, dot_y, c=dot_c, s=10)
    # tau = (agree - disagree) / (agree + disagree)
    # ax[0].text(
    #     (len(pset) - 1) / 2, ylim[0] + (ylim[1] - ylim[0]) / 10,
    #     r'$\tau=' + ('+%.3f' % tau if tau > 0 else '%.3f' % tau) + '$',
    #     va='center', ha='center'
    # )
    if notation:
        ax[0].set_ylabel(f'${notation}$ value')
    ax[0].set_xlim([-1, len(pset)])
    ax[0].set_xticks([])
    ax[0].grid(False)
    if not title:
        ax[0].plot([-10, -10], [-10, -10], color=clr_agree, label='Agreement', lw=8)
        ax[0].plot([-10, -10], [-10, -10], color=clr_disagree, label='Disagreement', lw=8)
        ax[0].set_title('Preference')
        # ax[0].legend(ncol=2, borderpad=0.0, borderaxespad=0.3, columnspacing=0.6, loc=(2., ylim[0]), framealpha=0)

    dot_x.clear()
    dot_y.clear()
    cross_x, cross_y = [], []
    eset_avgs = [(v(ln, 'A') + v(ln, 'B')) / 2 for _, ln in eset.iterrows()]
    eset.insert(len(eset.columns) - 1, 'avg', eset_avgs)
    eset = eset.sort_values('avg', ascending=False)
    for i, (_, ln) in enumerate(eset.iterrows()):
        fa, fb = v(ln, 'A'), v(ln, 'B')
        c, annt = ln['avg'], ln['ANNT']
        if annt == 'E':
            dot_x += [i, i]
            dot_y += [fa, fb]
            ax[1].plot([i, i], [fa, fb], color=clr_equal)
        else:
            cross_x += [i, i]
            cross_y += [fa, fb]
            ax[1].plot([i, i], [fa, fb], color=clr_neither)
    ax[1].scatter(dot_x, dot_y, c=clr_equal, s=10)
    ax[1].scatter(cross_x, cross_y, c=clr_neither, marker='x', s=16)
    ax[1].set_xlim([-1, len(eset)])
    ax[1].set_xticks([])
    ax[1].grid(False)
    if not title:
        ax[1].plot([-10, -10], [-10, -10], color=clr_neither, label='Neither was Fun', lw=8)
        ax[1].plot([-10, -10], [-10, -10], color=clr_equal, label='Equally Fun', lw=8)
        ax[1].set_title('Neutral')
        # ax[1].legend(ncol=2, borderpad=0.0, borderaxespad=0.3, columnspacing=0.6, loc=(2., ylim[0]), framealpha=0)

    plt.ylim(ylim)
    if ylim[0] > 0:
        plt.yticks(yticks, [*map(lambda x: '$~$%.1f' % x, yticks)])
    else:
        plt.yticks(yticks, [*map(lambda x: '%.1f' % x, yticks)])
    if title:
        fig.suptitle(title)
    fig.tight_layout(pad=0.5)

    if save_path:
        plt.savefig(getpath(save_path))
        plt.clf()
    else:
        plt.show()
    pass

# def make_statistical_tab(p=0.05):
#     global mv
#     fC = mv['fL'].to_numpy() + mv['fG-R'].to_numpy()
#     mv.insert(0, 'fC-R', fC)
#     fC = mv['fL'].to_numpy() + mv['fG-T'].to_numpy()
#     mv.insert(0, 'fC-T', fC)
#     theta = {
#         'fL': (mv['fL'].max() - mv['fL'].min()) * p,
#         'fG-R': (mv['fG-R'].max() - mv['fG-R'].min()) * p,
#         'fG-T': (mv['fG-T'].max() - mv['fG-T'].min()) * p,
#         'fC-R': (mv['fC-R'].max() - mv['fC-R'].min()) * p,
#         'fC-T': (mv['fC-T'].max() - mv['fC-T'].min()) * p,
#     }
#     def _count(_subdt):
#         aL, bL, eL = 0, 0, 0
#         aG, bG, eG = 0, 0, 0
#         aM, bM, eM = 0, 0, 0
#         for _, r in _subdt.iterrows():
#             fA, fB = mv['fL']['%s_lvl%s' % (r['ID'], r['A'])], mv['fL']['%s_lvl%s' % (r['ID'], r['B'])]
#             if fA - fB > theta['fL']:
#                 aL += 1
#             elif fB - fA > theta['fL']:
#                 bL += 1
#             else:
#                 eL += 1
#             fA, fB = mv['fG-R']['%s_lvl%s' % (r['ID'], r['A'])], mv['fG-R']['%s_lvl%s' % (r['ID'], r['B'])]
#             if fA - fB > theta['fG-R']:
#                 aG += 1
#             elif fB - fA > theta['fG-R']:
#                 bG += 1
#             else:
#                 eG += 1
#             fA = mv['fL']['%s_lvl%s' % (r['ID'], r['A'])] + mv['fG-R']['%s_lvl%s' % (r['ID'], r['A'])]
#             fB = mv['fL']['%s_lvl%s' % (r['ID'], r['B'])] + mv['fG-R']['%s_lvl%s' % (r['ID'], r['B'])]
#             if fA - fB > theta['fC-R']:
#                 aM += 1
#             elif fB - fA > theta['fC-R']:
#                 bM += 1
#             else:
#                 eM += 1
#         return aL, bL, eL, aG, bG, eG, aM, bM, eM
#
#     dtA = dt[dt['ANNT'] == 'A']
#     dtB = dt[dt['ANNT'] == 'B']
#     dtI = dt[dt['ANNT'] == 'E'].append(dt[dt['ANNT'] == 'N'])
#     # print(dtI)
#     fmt1 = r'  $A \succ B$ & ' + ' & '.join(['%02d'] * 12) + r' \\'
#     fmt2 = r'  $B \succ A$ & ' + ' & '.join(['%02d'] * 12) + r' \\'
#     fmt3 = r'  $A \approx B$ & ' + ' & '.join(['%02d'] * 12) + r' \\'
#     fmt4 = r'  sum & ' + ' & '.join(['%02d'] * 12)+ r' \\'
#
#     contents = np.zeros([4, 12], int)
#
#     tmp = _count(dtA)
#     contents[:3, 0], contents[:3, 4], contents[:3, 8] = tmp[:3], tmp[3:6], tmp[6:]
#     tmp = _count(dtB)
#     contents[:3, 1], contents[:3, 5], contents[:3, 9] = tmp[:3], tmp[3:6], tmp[6:]
#     tmp = _count(dtI)
#     contents[:3, 2], contents[:3, 6], contents[:3, 10] = tmp[:3], tmp[3:6], tmp[6:]
#
#     # contents[:3, 3], contents[:3, 7], contents[:3, 11] = tmp[:3], tmp[3:]
#     contents[:3, 3], contents[:3, 7], contents[:3, 11] = \
#         np.sum(contents[:3, :3], axis=1), np.sum(contents[:3, 4:7], axis=1), np.sum(contents[:3, 8:11], axis=1)
#     contents[3, :3], contents[3, 4:7], contents[3, 8:11] = \
#         np.sum(contents[:3, :3], axis=0), np.sum(contents[:3, 4:7], axis=0), np.sum(contents[:3, 8:11], axis=0)
#     contents[3, 3], contents[3, 7], contents[3, 11] = \
#         np.sum(contents[:3, 3]), np.sum(contents[:3, 7]), np.sum(contents[:3, 11])
#
#     print(fmt1 % tuple(contents[0]))
#     print(fmt2 % tuple(contents[1]))
#     print(fmt3 % tuple(contents[2]))
#     print('  \midrule')
#     print(fmt4 % tuple(contents[3]))
#     pass

def make_classification_tab(p=0.05):
    global mv
    fC = mv['fL'].to_numpy() + mv['fG-R'].to_numpy()
    mv.insert(0, 'fC-R', fC)
    fC = mv['fL'].to_numpy() + mv['fG-T'].to_numpy()
    mv.insert(0, 'fC-T', fC)
    theta = {
        'fL': (mv['fL'].max() - mv['fL'].min()) * p,
        'fG-R': (mv['fG-R'].max() - mv['fG-R'].min()) * p,
        'fG-T': (mv['fG-T'].max() - mv['fG-T'].min()) * p,
        'fC-R': (mv['fC-R'].max() - mv['fC-R'].min()) * p,
        'fC-T': (mv['fC-T'].max() - mv['fC-T'].min()) * p,
    }

    def _count(_m):
        pp, pn, pe = 0, 0, 0
        nn, ne = 0, 0
        for _, r in dt.iterrows():
            ia, ib = '%s_lvl%s' % (r['ID'], r['A']), '%s_lvl%s' % (r['ID'], r['B'])
            delta = mv[_m][ia] - mv[_m][ib]
            if r['ANNT'] == 'A':
                if delta > theta[_m]: pp += 1
                elif delta < -theta[_m]: pn += 1
                else: pe += 1
            elif r['ANNT'] == 'B':
                if delta < -theta[_m]: pp += 1
                elif delta > theta[_m]: pn += 1
                else: pe += 1
            else:
                if abs(delta) > theta[_m]: nn += 1
                else: ne += 1
        # preci_p = pp / (pp + pn + pe) * 100
        preci_p = (pp - pn) / (pp + pn)
        preci_n = ne / (ne + pe) * 100
        acc = (pp + ne) / len(dt) * 100
        return pp, pn, pe, preci_p, nn, ne, preci_n, acc
    fmt = ' & '.join(['%02d', '%02d', '%02d', '%.3f', '%02d', '%02d', '%.1f\\%%', '%.1f\\%%'])
    print('  $f_L$ &', fmt % _count('fL'), r'\\')
    print('  $f_G^R$ &', fmt % _count('fG-R'), r'\\')
    print('  $f_G^H$ &', fmt % _count('fG-T'), r'\\')
    print('  $f_L+f_G^R$ &', fmt % _count('fC-R'), r'\\')
    print('  $f_L+f_G^H$ &', fmt % _count('fC-T'), r'\\')

range_cfgs = {
    'mixed-R': [(-0.5, 2.0), np.linspace(-0.5, 2.0, 6)], 'mixed-K': [(-0.5, 2.0), np.linspace(-0.5, 2.0, 6)],
    'mixed-C': [(-0.5, 2.0), np.linspace(-0.5, 2.0, 6)], 'mixed-T': [(-50, 1.0), np.linspace(-50, 0, 6)],
    'fG-T': [(-50, 1.0), np.linspace(-50, 0, 6)], 'fG-R': [(-1.5, 1.0), np.linspace(-1.5, 1.0, 6)],
    'fG-K': [(-1.5, 1.0), np.linspace(-1.5, 1.0, 6)], 'fG-C': [(-1.5, 1.0), np.linspace(-1.5, 1.0, 6)],
    'fL': [(0.4, 1.0), np.linspace(0.4, 1.0, 7)]
    # 'fG-K': []
}
notations = {
    'mixed-R': 'f_L + f_G', 'mixed-K': 'f_L + f_G^K', 'mixed-C': 'f_L + f_G^C',
    'mixed-T': 'f_L + f_G^H', 'fG-R': 'f_G', 'fG-K': 'f_G^K', 'fG-C': 'f_G^C',
    'fG-T': 'f_G^H', 'fL': 'f_L'
}
result_rt = 'analysis/survey/results'
def plot_all_preferences(metric, notation, identification=''):
    if not identification: identification = metric
    r, n = range_cfgs[metric], notations[metric]
    path = f'{result_rt}/{metric}'
    os.makedirs(getpath(path), exist_ok=True)
    plot_preference_records(metric, *r, n, save_path=f'{path}/all.png')
    plot_preference_records(metric, *r, n, 'A', title='Expert', save_path=f'{path}/expert.png')
    plot_preference_records(metric, *r, skill_lvl='B', title='Non-expert', save_path=f'{path}/nonexpert.png')
    # plot_preference_records(metric, *r, skill_lvl='C', title='Intermediate Level', save_path=f'{path}/normal.png')
    # plot_preference_records(metric, *r, skill_lvl='D', title='Beginner Level', save_path=f'{path}/freshman.png')

    # upper = pygame.image.load(getpath(f'{path}/all.png'))
    upper = Image.open(getpath(f'{path}/all.png'))
    img_a = Image.open(getpath(f'{path}/expert.png'))
    img_b = Image.open(getpath(f'{path}/nonexpert.png'))
    # img_c = Image.open(getpath(f'{path}/normal.png'))
    # img_d = Image.open(getpath(f'{path}/freshman.png'))
    lower = Image.new('RGB', (img_a.width + img_b.width, img_a.height), 'white')
    x = 0
    for item in (img_a, img_b):
        lower.paste(item, (x, 0))
        x += item.width
    rescale = upper.width / lower.width
    lower = lower.resize((upper.width, int(rescale * lower.height)))

    canvas = Image.new('RGB', (upper.width, upper.height + lower.height), color='white')
    canvas.paste(upper, (0, 0))
    canvas.paste(lower, (0, upper.height))
    canvas.save(getpath(f'{path}/combined.png'))
    pass


if __name__ == '__main__':
    # print_latex_tab()

    # plot_metrics_wrt_gvals('fL', 'fval', 'f_L')
    # plot_metrics_wrt_gvals('fG-norm', 'fg-ns', 'f_G')

    # plot_agree_gval_heat()
    # plot_agree_gval_curves()
    # print(len(get_subdt(AnntTypes.PREF)))

    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.PREF, 'fL'))
    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.PREF, 'fG-ns'))
    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.EQUA, 'fL'))
    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.EQUA, 'fG-ns'))
    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.NITH, 'fL'))
    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.NITH, 'fG-ns'))
    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.ALL, 'fL'))
    # print('%.3f$\pm$%.2f & %.3f$\pm$%.2f' % avg_diff(AnntTypes.ALL, 'fG-ns'))

    # print(get_agreement('fL'))
    # print(get_agreement('fG-ns'))
    # plot_preference_records('fL', (0.5, 1.0), np.linspace(0.5, 1.0, 6), 'f_L', title=f'Visualisation with $f_L$')
    # plot_preference_records('fG-R', (-0.5, 1.0), np.linspace(-0.5, 1.0, 6), 'f_L', title=f'Visualisation with $f_L$')


    plot_all_preferences('fL', 'f_L', 'level-based metric')
    # plot_all_preferences('fG-R', 'f_G^R', 'gameplay-based metric with runner traces')
    # plot_all_preferences('fG-T', 'f_G^H', 'gameplay-based metric with human traces')
    # plot_all_preferences('mixed-R', 'f_L+f_G^R', 'combined metric with runner traces')
    # plot_all_preferences('mixed-T', 'f_L+f_G^H', 'combined metric with human traces')

    # plot_preference_records('fG-T', (-55, 1.0), np.linspace(-50, 0, 6), 'f^T_G')
    # plot_preference_records('fG-R', (-1.5, 1.0), np.linspace(-1.5, 1.0, 6), 'f^R_G')
    # plot_preference_records('mixed', (-0.5, 2.0), np.linspace(-0.5, 2, 6), '(f_L + f_G^R)')

    # make_classification_tab()
    pass

