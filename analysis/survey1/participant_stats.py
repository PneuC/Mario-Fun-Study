import matplotlib.pyplot as plt
import pandas as pds

if __name__ == '__main__':
    # data = pds.read_csv('questionnaire.csv')
    # print(data)
    # pass
    #
    # keys = ['Game Persona', 'Frequency of Playing Games (per Week)', 'Age', 'Gender']
    # valuess = [
    #     ['Runner', 'Killer', 'Collector', 'Others'],
    #     ['Never', 'seldom', '1-3h per week', '3-10h', '>10h'],
    #     ['<20', '20-29', '30-39', '40-49', '50-59', '$\leq$60', 'secret'],
    #     ['Male', 'Female', 'secret']
    # ]
    # for j, (key, vals) in enumerate(zip(keys, valuess)):
    #     col = data[f'Q{j+1}']
    #     n = len(vals)
    #     y = [0] * n
    #     for opt in col.values:
    #         j = ord(opt) - ord('A')
    #         y[j] = y[j] + 1
    #     plt.figure(figsize=(4.2, 2.4), dpi=300)
    #     plt.bar(range(n), y)
    #     plt.title(key)
    #     plt.xticks(range(n), vals, rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(f'./participants/Q{j+1}.png')
    pass
