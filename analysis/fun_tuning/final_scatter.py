import json
from itertools import product

from matplotlib import pyplot as plt

if __name__ == '__main__':
    with open(f'./fl_statistics.json', 'r') as f:
        fl_data = json.load(f)
    with open(f'./fg_statistics.json', 'r') as f:
        fg_data = json.load(f)

    # with open(f'./fun')
    for fl_infos, fg_infos in product(fl_data, fg_data):
        gl_val, gg_val = fl_infos['g'], fg_infos['g']
        if abs(gl_val - 0.14) < 1e-4 and abs(gg_val - 0.3) < 1e-4:
            x1, y1 = fl_infos['rand'], fg_infos['rand']
            x2, y2 = fl_infos['ori'], fg_infos['ori']
            plt.figure(figsize=(4, 4), dpi=256)
            plt.scatter(x1, y1, color='red', alpha=0.2, linewidths=0, s=24, edgecolors='red')
            plt.scatter(x2, y2, color='blue', alpha=0.2, linewidths=0, s=24, edgecolors='blue')
            plt.xlabel('$f_L$', size=12)
            plt.ylabel('$f_G$', size=12, rotation=0.)
            plt.title(f'$g_L={gl_val:.3f}$, $g_G={gg_val:.3f}$')
            plt.grid()
            # if n_init <= 4:
            plt.xlim((-1, 1.05))
            plt.ylim((-1, 1.05))
            # else:
            # plt.xlim((-3, 1.1))
            # plt.ylim((-3, 1.1))
            ticks = [-1., -0.5, 0., 0.5, 1.]
            plt.xticks(ticks, map(lambda v: '%.1f' % v, ticks))
            plt.yticks(ticks, map(lambda v: '%.1f' % v, ticks))
            plt.tight_layout()
            plt.show()
            # plt.savefig(f'./fun_scatter_plots/{i}_{j}.png')
            # plt.cla()
    # size = int(fig_size * dpi)
    # full_img = PIL.Image.new('RGB', (size * n_rows, size * n_cols), 'white')
    # print(n_cols, n_rows)
    # for i, j in product(range(n_rows), range(n_cols)):
    #     try:
    #         img = PIL.Image.open(f'./fun_scatter_plots/{i}_{j}.png')
    #         full_img.paste(img, (size * i, size * j, size * (i + 1), size * (j + 1)))
    #         full_img.save(f'./fun_scatter_plots/scatter_sheets.png')
    #     except FileNotFoundError:
    #         continue
