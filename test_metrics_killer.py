from analysis.algo.analyse_designer import test_metrics
from src.olgen.designer import Designer

if __name__ == '__main__':
    dpath = 'exp_data/main/LGP_K'
    designer = Designer(dpath)
    test_metrics(designer, dpath, 'metric_vals.json', 2)
