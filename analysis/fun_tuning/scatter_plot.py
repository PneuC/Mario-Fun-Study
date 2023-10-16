"""
  @Time : 2022/3/19 14:03 
  @Author : Ziqi Wang
  @File : make_map.py 
"""
import json
import PIL
import matplotlib.pyplot as plt
from src.smb.level import *



gc_lb, gc_ub = 0.1, 0.16
gb_lb, gb_ub = 0.24, 0.36
itvc, itvb = 0.01, 0.02
n_rows, n_cols = int((gb_ub - gb_lb + 1e-5) / itvb) + 1, int((gc_ub - gc_lb + 1e-5) / itvc) + 1


def compute_index(gc, gb):
    vc, vb = (gc - gc_lb) / itvc, (gb - gb_lb) / itvb
    if -1e-3 < round(vc) - vc < 1e-3 and -1e-3 < round(vb) - vb < 1e-3:
        return round(vb), round(vc)
    return -1, -1
    pass


if __name__ == '__main__':
    # X axis: gameplay, Y axis: level
    # rows: gb, cols: gc
    fig_size=3
    dpi=200
    bounds = {}
    plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    # for n in [3 ,4]:
    # with open(f'./fun_statistics.json', 'r') as f:
    #     data = json.load(f)
    with open(f'./fl_statistics.json', 'r') as f:
        fl_data = json.load(f)
    with open(f'./fg_statistics.json', 'r') as f:
        fg_data = json.load(f)

    # with open(f'./fun')
    for fl_infos, fg_infos in product(fl_data, fg_data):
        gl_val, gg_val = fl_infos['g'], fg_infos['g']
        i, j = compute_index(gl_val, gg_val)
        # print(i, j, n_rows, n_cols)
        if 0 <= i < n_rows and 0 <= j < n_cols:
            x1, y1 = fl_infos['rand'], fg_infos['rand']
            x2, y2 = fl_infos['ori'], fg_infos['ori']
            plt.scatter(x1, y1, color='red', alpha=0.12, linewidths=0, s=15)
            plt.scatter(x2, y2, color='blue', alpha=0.12, linewidths=0, s=15)
            plt.xlabel('$f_L$', size=12)
            plt.ylabel('$f_G$', size=12, rotation=0.)
            plt.title(f'$g_L={gl_val:.3f}$, $g_G={gg_val:.3f}$')
            plt.grid()
            # if n_init <= 4:
            plt.xlim((-3, 1.1))
            plt.ylim((-3, 1.1))
            # else:
            # plt.xlim((-3, 1.1))
            # plt.ylim((-3, 1.1))
            ticks = [-4., -3., -2., -1., 0., 1.]
            plt.xticks(ticks, map(lambda v: '%.1f' % v, ticks))
            plt.yticks(ticks, map(lambda v: '%.1f' % v, ticks))
            plt.tight_layout()
            plt.savefig(f'./fun_scatter_plots/{i}_{j}.png')
            plt.cla()
    size = int(fig_size * dpi)
    full_img = PIL.Image.new('RGB', (size * n_rows, size * n_cols), 'white')
    print(n_cols, n_rows)
    for i, j in product(range(n_rows), range(n_cols)):
        try:
            img = PIL.Image.open(f'./fun_scatter_plots/{i}_{j}.png')
            full_img.paste(img, (size * i, size * j, size * (i + 1), size * (j + 1)))
            full_img.save(f'./fun_scatter_plots/scatter_sheets.png')
        except FileNotFoundError:
            continue
        pass


