"""
  @Time : 2022/7/21 23:01 
  @Author : Ziqi Wang
  @File : environment.py 
"""

import gym
import time
import random
from torch import tensor
from src.smb.level import *
from src.gan.gankits import *
from src.env.logger import InfoCollector
from src.utils.datastruct import RingQueue
from typing import Optional, Callable, List
from src.env.rfunc import RewardFunc, defaults
from src.repair.repairer import DivideConquerRepairer
from src.smb.proxy import MarioProxy, MarioJavaAgents
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class OffRewardGenerationEnv(gym.Env):
    """
        Training environment.
    """
    def __init__(self, rfunc=None, N=defaults['n'], eplen=25, return_lvl=False, init_one=False, play_style='Runner'):
        self.rfunc = RewardFunc() if rfunc is None else rfunc
        self.mario_proxy = MarioProxy() if self.rfunc.require_simlt else None
        self.action_space = gym.spaces.Box(-1, 1, (nz,))
        self.N = N
        self.observation_space = gym.spaces.Box(-1, 1, (N * nz,))
        self.segs = []
        self.latvec_archive = RingQueue(N)
        self.eplen = eplen
        self.counter = 0
        self.repairer = DivideConquerRepairer()
        self.init_one = init_one
        self.backup_latvecs = None
        self.backup_strsegs = None
        self.return_lvl = return_lvl
        self.jagent = MarioJavaAgents.__getitem__(play_style)
        self.simlt_k = 80 if play_style == 'Runner' else 320

    def receive(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def step(self, data):
        action, strseg = data
        seg = MarioLevel(strseg)
        self.latvec_archive.push(action)

        self.counter += 1
        self.segs.append(seg)
        done = self.counter >= self.eplen
        if done:
            full_level = lvlhcat(self.segs)
            full_level = self.repairer.repair(full_level)
            w = MarioLevel.seg_width
            segs = [full_level[:, s: s + w] for s in range(0, full_level.w, w)]
            if self.mario_proxy:
                raw_simlt_res = self.mario_proxy.simulate_complete(lvlhcat(segs), self.jagent, self.simlt_k)
                simlt_res = MarioProxy.get_seg_infos(raw_simlt_res)
            else:
                simlt_res = None
            rewards = self.rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
            info = {}
            total_score = 0
            if self.return_lvl:
                info['LevelStr'] = str(full_level)
            for key in rewards:
                info[f'{key}_reward_list'] = rewards[key][-self.eplen:]
                info[f'{key}'] = sum(rewards[key][-self.eplen:])
                total_score += info[f'{key}']
            info['TotalScore'] = total_score
            info['EpLength'] = self.counter
        else:
            info = {}
        return self.__get_obs(), 0, done, info

    def reset(self):
        self.segs.clear()
        self.latvec_archive.clear()
        for latvec, strseg in zip(self.backup_latvecs, self.backup_strsegs):
            self.latvec_archive.push(latvec)
            self.segs.append(MarioLevel(strseg))

        self.backup_latvecs, self.backup_strsegs = None, None
        self.counter = 0
        return self.__get_obs()

    def __get_obs(self):
        lack = self.N - len(self.latvec_archive)
        pad = [np.zeros([nz], np.float32) for _ in range(lack)]
        return np.concatenate([*pad, *self.latvec_archive.to_list()])

    def render(self, mode='human'):
        pass


class VecGenerationEnv(SubprocVecEnv):
    def __init__(
        self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None, N=defaults['n'],
        init_one=False, log_path=None, log_itv=-1, log_targets=None, device='cuda:0'
    ):
        super(VecGenerationEnv, self).__init__(env_fns, start_method)
        self.generator = get_generator('models/generator.pth', device=device)

        if log_path:
            self.logger = InfoCollector(log_path, log_itv, log_targets)
        else:
            self.logger = None
        self.N = N
        self.total_steps = 0
        self.start_time = time.time()
        self.device = device
        self.init_one = init_one
        self.latvec_set = np.load(getpath('smb/init_latvecs.npy'))

    def step_async(self, actions: np.ndarray) -> None:
        with torch.no_grad():
            z = torch.clamp(tensor(actions.astype(np.float32), device=self.device), -1, 1).view(-1, nz, 1, 1)
            segs = process_onehot(self.generator(z))
        for remote, action, seg in zip(self.remotes, actions, segs):
            remote.send(("step", (action, str(seg))))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self.total_steps += self.num_envs
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        envs_to_send = [i for i in range(self.num_envs) if dones[i]]
        self.send_reset_data(envs_to_send)

        if self.logger is not None:
            for i in range(self.num_envs):
                if infos[i]:
                    infos[i]['TotalSteps'] = self.total_steps
                    infos[i]['TimePassed'] = time.time() - self.start_time
            self.logger.on_step(dones, infos)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        self.send_reset_data()
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        self.send_reset_data()
        return _flatten_obs(obs, self.observation_space)

    def send_reset_data(self, env_ids=None):
        if env_ids is None:
            env_ids = [*range(self.num_envs)]
        target_remotes = self._get_target_remotes(env_ids)

        n_inits = 1 if self.init_one else self.N
        # latvecs = [sample_latvec(n_inits, tensor=False) for _ in range(len(env_ids))]

        latvecs = [self.latvec_set[random.sample(range(len(self.latvec_set)), n_inits)] for _ in range(len(env_ids))]
        with torch.no_grad():
            segss = [[] for _ in range(len(env_ids))]
            for i in range(len(env_ids)):
                z = tensor(latvecs[i]).view(-1, nz, 1, 1).to(self.device)
                segss[i] = process_onehot(self.generator(z))
        for remote, latvec, segs in zip(target_remotes, latvecs, segss):
            kwargs = {'backup_latvecs': latvec, 'backup_strsegs': [str(seg) for seg in segs]}
            remote.send(("env_method", ('receive', [], kwargs)))
        for remote in target_remotes:
            remote.recv()

    def close(self) -> None:
        super().close()
        if self.logger is not None:
            self.logger.close()


def make_vec_offrew_env(
        num_envs, rfunc=None, log_path=None, eplen=25, log_itv=-1, N=defaults['n'], init_one=False,
        play_style='Runner', device='cuda:0', log_targets=None, return_lvl=False
    ):
    return make_vec_env(
        OffRewardGenerationEnv, n_envs=num_envs, vec_env_cls=VecGenerationEnv,
        vec_env_kwargs={
            'log_path': log_path,
            'log_itv': log_itv,
            'log_targets': log_targets,
            'device': device,
            'N': N,
            'init_one': init_one
        },
        env_kwargs={
            'rfunc': rfunc,
            'eplen': eplen,
            'return_lvl': return_lvl,
            'play_style': play_style,
            'N': N,
            'init_one': init_one
        }
    )

