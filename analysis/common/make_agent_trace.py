"""
  @Time : 2022/7/22 21:39 
  @Author : Ziqi Wang
  @File : make_agent_trace.py 
"""

import json
import pygame.image
from src.utils.filesys import getpath
from src.smb.proxy import MarioProxy, MarioJavaAgents
from src.smb.level import traverse_level_files, MarioLevel, LevelRender


def simulate_inpath(path):
    proxy = MarioProxy()

    for lvl, name in traverse_level_files(path):
        rtrace = proxy.simulate_complete(lvl, MarioJavaAgents.Runner, 8)['full_trace']
        ktrace = proxy.simulate_complete(lvl, MarioJavaAgents.Killer, 20)['full_trace']
        ctrace = proxy.simulate_complete(lvl, MarioJavaAgents.Collector, 20)['full_trace']
        # if save_simlt:
        data = {'Runner': rtrace, 'Killer': ktrace, 'Collector': ctrace}
        with open(getpath(f'{path}/{name}_traces.json'), 'w') as f:
            json.dump(data, f)
        # if save_img:
        #     lname.to_img_with_trace(fg_res['simlt_res'], f'{path}/{name}_with_{agent.name}_trace.png')
        pass
    pass

def vis_with_all_traces(path):
    for lvl, name in traverse_level_files(path):
        with open(getpath(f'{path}/{name}_traces.json'), 'r') as f:
            data = json.load(f)
        lvlimg = lvl.to_img()
        LevelRender.draw_trace_on(lvlimg, data['Runner'])
        # LevelRender.draw_trace_on(lvlimg, data['Killer'], 'red')
        # LevelRender.draw_trace_on(lvlimg, data['Collector'], 'white')
        pygame.image.save(lvlimg, getpath(f'{path}/{name}_with_all_trace.png'))
    pass

if __name__ == '__main__':
    simulate_inpath('exp_data/main/LGP_R/samples')
    # simulate_inpath('exp_data/main/LGP_K/samples')
    # simulate_inpath('exp_data/main/LGP_C/samples')
    vis_with_all_traces('exp_data/main/LGP_R/samples')
    # vis_with_all_traces('exp_data/main/LGP_K/samples')
    # vis_with_all_traces('exp_data/main/LGP_C/samples')
    pass

