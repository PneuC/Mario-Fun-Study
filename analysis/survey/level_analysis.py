import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.smb.level import *
from src.smb.proxy import MarioProxy
from src.utils.filesys import getpath
from src.utils.img import make_img_sheet
from scipy.stats import wilcoxon


def test_agent_trace():
    for level, name in traverse_level_files('exp_data/survey data/formal/lvls'):
        if 'lvl' in name:
            continue
        proxy = MarioProxy()
        trace = proxy.simulate_complete(level)['full_trace']
        # simlt_res = MarioProxy.get_seg_infos(proxy.simulate_complete(level))
        with open(getpath(f'exp_data/survey data/formal/lvl_traces/{name}.json'), 'w') as f:
            json.dump(trace, f)
        # print(name)
        pass
    pass

def plot_compression_scatter(distmat, labels, colors, save_path='', title=''):
    # if colors is None:
    #     colors = [None] * len(samples)
    # x = np.concatenate(samples, axis=0)
    # splits = [0]
    # for i in range(len(samples)):
    #     splits.append(splits[i] + len(samples[i]))
    #
    # # metric1 = lambda A, B: np.mean([tile_pattern_js_div(a, b) for a, b in zip(A.to_segs(), B.to_segs())])
    # # metric2 = lambda A, B: np.mean([trace(a, b) for a, b in zip(A.to_segs(), B.to_segs())])
    #
    color_map = {'Runner': colors[0], 'Killer': colors[1], 'Collector': colors[2]}
    ts = TSNE(n_components=2, learning_rate='auto', metric='precomputed', n_iter=2000, perplexity=10)
    embx = np.array(ts.fit_transform(distmat))

    # print(splits)
    # embs = [embx[splits[i]:splits[i+1]] for i in range(len(samples))]
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(2.5, 2.25), dpi=384)
    c = [color_map[l] for l in labels]
    plt.scatter(embx[:100, 0], embx[:100, 1], c=colors[0], s=12, linewidths=0, label='Runner')
    plt.scatter(embx[100:200, 0], embx[100:200, 1], c=colors[1], s=12, linewidths=0, label='Killer')
    plt.scatter(embx[200:, 0], embx[200:, 1], c=colors[2], s=12, linewidths=0, label='Collector')
    for i in range(4):
        plt.text(embx[i+100, 0], embx[i+100, 1], str(i+1))
        plt.text(embx[i+200, 0], embx[i+200, 1], str(i+1))
        pass
    # for emb, lb, c in zip(embs, labels,colors):
    #     plt.scatter(emb[:,0], emb[:,1], c=c, label=lb, alpha=0.15, linewidths=0, s=7)

    xspan = 1.2 * max(abs(embx[:, 0].max()), abs(embx[:, 0].min()))
    yspan = 1.05 * max(abs(embx[:, 1].max()), abs(embx[:, 1].min()))
    plt.xlim([-xspan, xspan])
    plt.ylim([-yspan, 1.4 * yspan])
    plt.legend(ncol=3, loc='upper center', handletextpad=0.02, labelspacing=0.05, columnspacing=0.2)
    plt.xticks([-xspan, -0.5 * xspan, 0, 0.5 * xspan, xspan], [''] * 5)
    plt.yticks([-yspan, -0.5 * yspan, 0, 0.6 * yspan, yspan], [''] * 5)
    plt.title(title)
    plt.legend(loc='upper center', ncol=3, fontsize=9, handlelength=.5, handletextpad=0.5, framealpha=0.)
    plt.tight_layout(pad=0.2)
    if save_path:
        plt.savefig(getpath(save_path))
    else:
        plt.show()

def compute_distmat():
    hm_metric = lambda A, B: normalised_hamming_dis(A[:, 16:], B[:, 16:])
    js_metric = lambda A, B: np.mean([tile_pattern_js_div(a[1:], b[1:]) for a, b in zip(A.to_segs(), B.to_segs())])
    # trace_metric = lambda A, B: np.mean([trace_div(a, b) for a, b in zip(A, B)])

    runner_levels = [MarioLevel.from_file(fname) for fname in glob.glob(getpath('lvls/double-check/part2-fin/Runner-*.lvl'))]
    killer_levels = [MarioLevel.from_file(fname) for fname in glob.glob(getpath('lvls/double-check/part2-fin/Killer-*.lvl'))]
    collector_levels = [MarioLevel.from_file(fname) for fname in glob.glob(getpath('lvls/double-check/part2-fin/Collector-*.lvl'))]

    lvls = runner_levels + killer_levels + collector_levels
    mat = [[js_metric(lvl1, lvl2) for lvl1 in lvls] for lvl2 in lvls]
    mat = np.array(mat)
    np.save(getpath('exp_data/survey data/formal/js_mat_doublecheck.npy'), mat)
    mat = [[hm_metric(lvl1, lvl2) for lvl1 in lvls] for lvl2 in lvls]
    mat = np.array(mat)
    np.save(getpath('exp_data/survey data/formal/hm_mat_doublecheck.npy'), mat)

    # traces = []
    # for fname in glob.glob(getpath(f'exp_data/survey data/formal/lvl_traces/*')):
    #     with open(fname, 'r') as f:
    #         traces.append(json.load(f))
    #
    # mat = [[trace_metric(t1, t2) for t1 in traces] for t2 in traces]
    # mat = np.array(mat)
    # np.save(getpath('exp_data/survey data/formal/dtw_mat.npy'), mat)

def cat_example_levels():
    with open(getpath('lvls/part2-fin/Runner-metricvals.json'), 'r') as f:
        rlvl_mvs = json.load(f)
    with open(getpath('lvls/part2-fin/Killer-metricvals.json'), 'r') as f:
        klvl_mvs = json.load(f)
    with open(getpath('lvls/part2-fin/Collector-metricvals.json'), 'r') as f:
        clvl_mvs = json.load(f)

    plt.style.use('seaborn-v0_8-dark-palette')
    plt.figure(figsize=(4, 4), dpi=256)
    plt.scatter([item['fL'] for item in rlvl_mvs], [item['fG-R'] for item in rlvl_mvs], s=6, label='R')
    plt.scatter([item['fL'] for item in klvl_mvs], [item['fG-K'] for item in klvl_mvs], s=6, label='K')
    plt.scatter([item['fL'] for item in clvl_mvs], [item['fG-C'] for item in clvl_mvs], s=6, label='C')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.xlim((-1.5, 1))
    plt.ylim((-1.5, 1))
    plt.show()

    i = np.argmin([item['fL'] for item in rlvl_mvs])
    j = np.argmin([item['fL'] for item in klvl_mvs])
    k = np.argmin([item['fL'] for item in clvl_mvs])
    print(i, j, k)
    rlvl = MarioLevel.from_file(f'lvls/part2-fin/Runner-{i}.lvl')[:, 16:]
    klvl = MarioLevel.from_file(f'lvls/part2-fin/Killer-{j}.lvl')[:, 16:]
    clvl = MarioLevel.from_file(f'lvls/part2-fin/Collector-{k}.lvl')[:, 16:]
    rimg, kimg, cimg = rlvl.to_img(None), klvl.to_img(None), clvl.to_img(None)
    make_img_sheet([rimg, kimg, cimg], 1, 24, save_path='analysis/survey/results/testset2-illustrate/maxL.png')

    i = np.argmax([item['fG-R'] for item in rlvl_mvs])
    j = np.argmax([item['fG-K'] for item in klvl_mvs])
    k = np.argmax([item['fG-C'] for item in clvl_mvs])
    print(i, j, k)
    rlvl = MarioLevel.from_file(f'lvls/part2-fin/Runner-{i}.lvl')[:, 16:]
    klvl = MarioLevel.from_file(f'lvls/part2-fin/Killer-{j}.lvl')[:, 16:]
    clvl = MarioLevel.from_file(f'lvls/part2-fin/Collector-{k}.lvl')[:, 16:]
    rimg, kimg, cimg = rlvl.to_img(None), klvl.to_img(None), clvl.to_img(None)
    make_img_sheet([rimg, kimg, cimg], 1, 24, save_path='analysis/survey/results/testset2-illustrate/maxG.png')

    i = np.argmin([item['fL'] for item in rlvl_mvs])
    j = np.argmin([item['fL'] for item in klvl_mvs])
    k = np.argmin([item['fL'] for item in clvl_mvs])
    print(i, j, k)
    rlvl = MarioLevel.from_file(f'lvls/part2-fin/Runner-{i}.lvl')[:, 16:]
    klvl = MarioLevel.from_file(f'lvls/part2-fin/Killer-{j}.lvl')[:, 16:]
    clvl = MarioLevel.from_file(f'lvls/part2-fin/Collector-{k}.lvl')[:, 16:]
    rimg, kimg, cimg = rlvl.to_img(None), klvl.to_img(None), clvl.to_img(None)
    make_img_sheet([rimg, kimg, cimg], 1, 24, save_path='analysis/survey/results/testset2-illustrate/minL.png')

    i = np.argmin([item['fG-R'] for item in rlvl_mvs])
    j = np.argmin([item['fG-K'] for item in klvl_mvs])
    k = np.argmin([item['fG-C'] for item in clvl_mvs])
    print(i, j, k)
    rlvl = MarioLevel.from_file(f'lvls/part2-fin/Runner-{i}.lvl')[:, 16:]
    klvl = MarioLevel.from_file(f'lvls/part2-fin/Killer-{j}.lvl')[:, 16:]
    clvl = MarioLevel.from_file(f'lvls/part2-fin/Collector-{k}.lvl')[:, 16:]
    rimg, kimg, cimg = rlvl.to_img(None), klvl.to_img(None), clvl.to_img(None)
    make_img_sheet([rimg, kimg, cimg], 1, 24, save_path='analysis/survey/results/testset2-illustrate/minG.png')

    pass

def level_statisitc():
    def _get_stats(_lvls):
        tmp = [l.count_gaps() for l in _lvls]
        gaps = [item[0] for item in tmp]
        longest_gaps = [item[1] for item in tmp]
        enemies = [l.n_enemies for l in _lvls]
        coins = [l.n_coins for l in _lvls]
        return gaps, longest_gaps, enemies, coins

    def _fmt_stats(gaps, longest_gaps, enemies, coins):
        fmt = ' & '.join(['%.2f $\pm$ %.2f'] * 4)
        return fmt % (
            np.mean(gaps), np.std(gaps), np.mean(longest_gaps), np.std(longest_gaps),
            np.mean(enemies), np.std(enemies), np.mean(coins), np.std(coins)
        )
    rlevels = [MarioLevel.from_file(fname) for fname in glob.glob(getpath('lvls/part2-fin/Runner-*.lvl'))]
    klevels = [MarioLevel.from_file(fname) for fname in glob.glob(getpath('lvls/part2-fin/Killer-*.lvl'))]
    clevels = [MarioLevel.from_file(fname) for fname in glob.glob(getpath('lvls/part2-fin/Collector-*.lvl'))]

    print('  Runner &', _fmt_stats(*_get_stats(rlevels)), r'\\')
    print('  Killer &', _fmt_stats(*_get_stats(klevels)), r'\\')
    print('  Collector &', _fmt_stats(*_get_stats(clevels)), r'\\')
    keys = ('gaps', 'longest_gaps', 'enemies', 'coins')
    r_stats = {k: v for k, v in zip(keys, _get_stats(rlevels))}
    k_stats = {k: v for k, v in zip(keys, _get_stats(klevels))}
    c_stats = {k: v for k, v in zip(keys, _get_stats(clevels))}

    for key in keys:
        print(' & ', end='')
        print('*' if wilcoxon(r_stats[key], k_stats[key])[1] < 0.05 else r'\times', end=' & ')
        print('*' if wilcoxon(r_stats[key], c_stats[key])[1] < 0.05 else r'\times', end=' & ')
    print(r'\\')
    for key in keys:
        print('*' if wilcoxon(k_stats[key], r_stats[key])[1] < 0.05 else r'\times', end=' & ')
        print(' & ', end='')
        print('*' if wilcoxon(k_stats[key], c_stats[key])[1] < 0.05 else r'\times', end=' & ')
    print(r'\\')
    for key in keys:
        print('*' if wilcoxon(c_stats[key], r_stats[key])[1] < 0.05 else r'\times', end=' & ')
        print('*' if wilcoxon(c_stats[key], k_stats[key])[1] < 0.05 else r'\times', end=' & ')
        print(' & ', end='')
    pass


if __name__ == '__main__':
    # test_agent_trace()
    # compute_distmat()

    # lbs = ['Runner'] * 100 + ['Killer'] * 100 + ['Collector'] * 100
    # distmat = np.load(getpath('exp_data/survey data/formal/hm_mat.npy'))
    # plot_compression_scatter(distmat, lbs, ('#FFC750', '#FF4F8D', '#2C73D2'), title='Level-Hamming Measurement')
    #
    # lbs = ['Runner'] * 100 + ['Killer'] * 100 + ['Collector'] * 100
    # distmat = np.load(getpath('exp_data/survey data/formal/hm_mat_doublecheck.npy'))
    # plot_compression_scatter(distmat, lbs, ('#FFC750', '#FF4F8D', '#2C73D2'), title='Level-Hamming Measurement')
    #
    # lbs = ['Collector'] * 100 + ['Killer'] * 100 + ['Runner'] * 100
    # distmat = np.load(getpath('exp_data/survey data/formal/js_mat.npy'))
    # plot_compression_scatter(distmat, lbs, ('#FFC750', '#FF4F8D', '#2C73D2'), title='Level-facet Measurement')
    #
    # lbs = ['Collector'] * 100 + ['Killer'] * 100 + ['Runner'] * 100
    # distmat = np.load(getpath('exp_data/survey data/formal/js_mat_doublecheck.npy'))
    # plot_compression_scatter(distmat, lbs, ('#FFC750', '#FF4F8D', '#2C73D2'), title='Level-facet Measurement')
    #
    # lbs = ['Collector'] * 100 + ['Killer'] * 100 + ['Runner'] * 100
    # distmat = np.load(getpath('exp_data/survey data/formal/dtw_mat.npy'))
    # plot_compression_scatter(distmat, lbs, ('#FFC750', '#FF4F8D', '#2C73D2'), title='Gameplay-facet Measurement')

    # cat_example_levels()
    level_statisitc()
    pass

