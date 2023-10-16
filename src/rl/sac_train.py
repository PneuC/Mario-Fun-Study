"""
  @Time : 2022/7/21 23:24 
  @Author : Ziqi Wang
  @File : sac_train.py
"""

import os
import json
import time
import torch
import importlib
from src.gan.gans import nz
from src.rl.rep_mem import ReplayMem
from src.utils.filesys import auto_dire, getpath
from src.env.environment import make_vec_offrew_env
from src.rl.sac import SAC_Model, OffRewSAC_Trainer
from src.rl.actor_critic import SquashedGaussianMLPActor, MLPQFunction


def set_parser(parser):
    """
    Configure the arguments to run training
    :param parser:
    :return:
    """
    parser.add_argument(
        '--n_envs', type=int, default=5,
        help='Number of parallel environments.'
    )
    parser.add_argument(
        '--eplen', type=int, default=50,
        help='Maximum nubmer of segments to generate in the generation enviroment.'
    )
    parser.add_argument(
        '--total_steps', type=int, default=int(1e5),
        help='Total time steps (frames) for training SAC designer.'
    )
    parser.add_argument('--gamma', type=float, default=0.8, help='Discount factor of RL')
    parser.add_argument('--tar_entropy', type=float, default=-nz, help='Targe entropy parameter of SAC (default: dimensionality of latent vector, i.e., action)')
    parser.add_argument('--tau', type=float, default=0.02, help='Target net smoothing parameter of SAC')
    parser.add_argument('--update_freq', type=int, default=2, help='Update networks for once after how many steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--mem_size', type=int, default=int(1e6), help='Capacity of replay memory')
    parser.add_argument(
        '--gpuid', type=int, default=0,
        help='ID of GPU to train the SAC designer. CPU will be used if gpuid < 0'
    )
    parser.add_argument(
        '--rfunc_name', type=str, default='default',
        help='Name of the file where the reward function located. '
             'The file must be put in the \'src.reward_functions\' package.'
    )
    parser.add_argument(
        '--res_path', type=str, default='',
        help='Path relateed to \'/exp_data\'to save the training log. '
             'If not specified, a new rep_folder named exp{id} will be created.'
    )
    parser.add_argument(
        '--play_style', type=str, default='Runner',
        help='Path relateed to \'/exp_data\'to save the training log. '
             'If not specified, a new rep_folder named exp{id} will be created.'
    )
    parser.add_argument('--init_one', action='store_true')
    parser.add_argument(
        '--check_points', type=int, nargs='+',
        help='check points to save deisigner, specified by the number of time steps.'
    )

def train_designer(cfgs):
    if not cfgs.res_path:
        res_path = auto_dire('exp_data', 'Designer')
    else:
        res_path = getpath('exp_data/' + cfgs.res_path)
        try:
            os.makedirs(res_path)
        except FileExistsError:
            if os.path.exists(f'{res_path}/actor.pth'):
                print(f'Training is cancelled due to \`{res_path}\` has been occupied')
                return
    device = 'cpu' if cfgs.gpuid < 0 or not torch.cuda.is_available() else f'cuda:{cfgs.gpuid}'

    rfunc = (
        importlib.import_module('src.env.rfuncs')
        .__getattribute__(f'{cfgs.rfunc_name}')
    )
    with open(res_path + '/run_config.txt', 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M') + '\n')
        f.write('---------SAC---------\n')
        args_strlines = [
            f'{key}={val}\n' for key, val in vars(cfgs).items()
            if key not in {'rfunc_name', 'res_path', 'entry', 'check_points'}
        ]
        f.writelines(args_strlines)
        f.write('-' * 50 + '\n')
        f.write(str(rfunc))
    N = rfunc.get_n()
    with open(f'{res_path}/N.json', 'w') as f:
        json.dump(N, f)

    env = make_vec_offrew_env(
        cfgs.n_envs, rfunc, res_path, cfgs.eplen, N=N, play_style=cfgs.play_style,
        log_itv=cfgs.n_envs, device=device, log_targets=['file', 'std'], init_one=cfgs.init_one
    )

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    designer = SAC_Model(
        lambda: SquashedGaussianMLPActor(obs_dim, act_dim, [256, 256]),
        lambda: MLPQFunction(obs_dim, act_dim, [256, 256]),
        gamma=cfgs.gamma, tar_entropy=cfgs.tar_entropy, tau=cfgs.tau, device=device
    )
    d_trainer = OffRewSAC_Trainer(
        env, cfgs.total_steps, cfgs.update_freq, cfgs.batch_size, ReplayMem(cfgs.mem_size),
        res_path, cfgs.check_points
    )
    d_trainer.train(designer)

