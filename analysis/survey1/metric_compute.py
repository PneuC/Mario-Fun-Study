import csv

import numpy as np
import pandas as pds
from src.smb.proxy import MarioProxy
from src.smb.level import MarioLevel
from src.env.rfunc import LevelSACN, GameplaySACN

data = pds.read_csv('./annotation.csv')
lvlset = {v for v in data['A-ID'].values} and {v for v in data['A-ID'].values}
# print(data)
proxy = MarioProxy()
fun_level = LevelSACN()
fun_gameplay = GameplaySACN()

def get_info(ip, lid):
    lvl = MarioLevel.from_file(f'analysis/survey1/levels/lvl{lid}.lvl')
    repath = f'analysis/survey1/reps/{ip}lvl{lid}.rep'
    gameres = proxy.replay(lvl, repath, 5, False)
    tmp = {'simlt_res': gameres['simlt_res'], 'restarts': []}
    simlt_res = MarioProxy.get_seg_infos(tmp)
    n = len(simlt_res)
    segs = lvl.to_segs()[:n]
    fl = np.mean(fun_level.compute_rewards(segs=segs))
    fg = np.mean(fun_gameplay.compute_rewards(segs=segs, simlt_res=simlt_res))
    print(repath, gameres['completing-ratio'], n, fl, fg)
    return gameres['completing-ratio'], fl, fg
    pass

if __name__ == '__main__':
    f = open('./annt_with_metric.csv', 'w', newline='')
    wrtr = csv.writer(f)
    wrtr.writerow(['A-cr', 'B-cr', 'A-fl', 'B-fl', 'A-fg', 'B-fg', 'Annt', ''])
    for _, ln in data.iterrows():
        participant_ip, aid, bid, annt = ln['IP'], ln['A-ID'], ln['B-ID'], ln['Anno']
        cr_A, fl_A, fg_A = get_info(participant_ip, aid)
        cr_B, fl_B, fg_B = get_info(participant_ip, bid)
        wrtr.writerow([cr_A, cr_B, fl_A, fl_B, fg_A, fg_B, annt, ''])
        # print(ln['A-ID'])
    f.close()
        # lvl = MarioLevel.from_file(f'./levels/lvl{token}')

    # print(data)
    # for
    pass
