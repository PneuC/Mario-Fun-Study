"""
  @Time : 2022/8/6 14:38 
  @Author : Ziqi Wang
  @File : play.py 
"""
import json

from src.smb.level import *
from src.smb.proxy import MarioProxy
from misc.make_rand_lvls import make_lvls

if __name__ == '__main__':
    make_lvls()
    make_lvls()

    # proxy = MarioProxy()
    # for lname, name in traverse_level_files('smb/levels'):
    #     simlt_res = proxy.simulate_complete(lname)
    #     with open(f'smb/levels/{name}_simlt_res.json', 'w') as f:
    #         json.dump(simlt_res, f)
    #     pass
    # lname = MarioLevel.from_file('exp_data/gl020_gg030/designer_FL+FP-C/samples/sample15.lvl')
    # proxy = MarioProxy()
    # proxy.play_game(lname, 15, verbose=True)
    pass

