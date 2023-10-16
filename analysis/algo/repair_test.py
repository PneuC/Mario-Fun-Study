import glob
import json
import time
import numpy as np
from matplotlib import pyplot as plt
from src.utils.filesys import getpath
from src.repair.repairer import Repairer
from src.smb.level import traverse_level_files, MarioLevel

def test():
    # lvls = []
    repairer = Repairer()
    log_infos = []
    n = 0
    start_time = time.time()
    for lvl, name in traverse_level_files('lvls/part1'):
        if 'repaired' in name:
            continue
        n += 1
        if n > 50:
            break
        _, loginfo = repairer.repair(lvl)
        log_infos.append(loginfo)
        print('%d levels repaired, %.1fs elapsed' % (n, time.time() - start_time))

    fitness_y = [[item[0] for item in loginfo] for loginfo in log_infos]
    wrongs_y = [[item[1] for item in loginfo] for loginfo in log_infos]
    uncertains_y = [[item[2] for item in loginfo] for loginfo in log_infos]
    changes_y = [[item[3] for item in loginfo] for loginfo in log_infos]

    with open(getpath('exp_data/repair/fitness.json'), 'w') as f:
        json.dump(fitness_y, f)
    with open(getpath('exp_data/repair/wrongs.json'), 'w') as f:
        json.dump(wrongs_y, f)
    with open(getpath('exp_data/repair/uncertains.json'), 'w') as f:
        json.dump(uncertains_y, f)
    with open(getpath('exp_data/repair/changes.json'), 'w') as f:
        json.dump(changes_y, f)

    # print(fitness_y)
    # print(wrongs_y)
    # print(uncertains_y)
    # print(changes_y)

    # lvls = [lname if name.contain('repaired') for lname, name in traverse_level_files('lvls/part1')]

    # for lname, name in traverse_level_files('lvls/part1'):
    #
    #     pass

def plot():
    with open(getpath('exp_data/repair/fitness.json'), 'r') as f:
        fitness = np.array(json.load(f))
    with open(getpath('exp_data/repair/wrongs.json'), 'r') as f:
        wrongs = np.array(json.load(f))
    with open(getpath('exp_data/repair/uncertains.json'), 'r') as f:
        uncertains = np.array(json.load(f))
    with open(getpath('exp_data/repair/changes.json'), 'r') as f:
        changes = np.array(json.load(f))
    plt.style.use('seaborn')
    plt.figure(figsize=(4, 2.5), dpi=256)

    plt.plot(list(range(500)), fitness.mean(axis=0), label='Fitness')
    plt.plot(list(range(500)), wrongs.mean(axis=0), label='#wrong')
    plt.plot(list(range(500)), uncertains.mean(axis=0), label='#uncertain')
    plt.plot(list(range(500)), changes.mean(axis=0), label='#change')

    plt.legend(ncol=2, loc='lower right')
    plt.show()
    pass


if __name__ == '__main__':
    # plot()
    # lname = MarioLevel.from_file('analysis/algo/repair_cannon/cannon-test.lvl')
    # lname.to_img('analysis/algo/repair_cannon/cannon-test.png')

    repairer = Repairer()
    lvl1 = MarioLevel.from_file('analysis/algo/repair_cannon/lvl5.lvl')
    lvl2 = MarioLevel.from_file('analysis/algo/repair_cannon/cannon-test.lvl')
    lvl1_repaired, _ = repairer.repair(lvl1)
    lvl2_repaired, _ = repairer.repair(lvl2)
    lvl1_repaired.to_img('analysis/algo/repair_cannon/lvl5-repaired.png')
    lvl2_repaired.to_img('analysis/algo/repair_cannon/cannon-test-repaired.png')
    pass

