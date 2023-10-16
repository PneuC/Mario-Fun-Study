import csv
import random

import pandas as pds
from src.utils.filesys import getpath


if __name__ == '__main__':
    def _foo():
        pe = pemax * max((A[key] + B[key]) / 2, 0) ** 2
        pn = pnmax * max(1 - (A[key] + B[key]) / 2, 0) ** 2
        eq, nt = random.random() < pe, random.random() < pn
        if eq and not nt:
            wrtr.writerow([A['FL'], B['FL'], A['FG'], B['FG'], 'E', ''])
        elif not eq and nt:
            wrtr.writerow([A['FL'], B['FL'], A['FG'], B['FG'], 'N', ''])
        else:
            if random.random() < pagr:
                Annt = 'A' if A[key] > B[key] else 'B'
                wrtr.writerow([A['FL'], B['FL'], A['FG'], B['FG'], Annt, ''])
            else:
                Annt = 'B' if A[key] > B[key] else 'A'
                wrtr.writerow([A['FL'], B['FL'], A['FG'], B['FG'], Annt, ''])

    data = pds.read_csv(getpath('lvls/rand_gen_lvls/metric_vals.csv'))
    n = len(data)
    f = open('./annt_examples.csv', 'w', newline='')
    wrtr = csv.writer(f)
    wrtr.writerow(['A-fl', 'B-fl', 'A-fg', 'B-fg', 'Anno', 'pad'])
    pemax, pnmax = 0.7, 0.7
    pagr = 0.8
    for i in range(100):
        a, b = random.sample(range(n), 2)
        A, B = data.iloc[a], data.iloc[b]
        key = 'FL' if random.random() < 0.4 else 'FG'
        _foo()
    f.close()
    pass
