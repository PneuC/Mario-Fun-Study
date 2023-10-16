"""
  @Time : 2022/7/21 23:05 
  @Author : Ziqi Wang
  @File : rfunc.py 
"""

from abc import abstractmethod

import numpy as np

from src.utils.mymath import a_clip
from src.smb.level import trace_div, tile_pattern_js_div, lvlhcat, MarioLevel

defaults = {'n': 5, 'gl': 0.14, 'gg': 0.30, 'wl': 2, 'wg': 10}


class RewardFunc:
    def __init__(self, *args):
        self.terms = args
        self.require_simlt = any(term.require_simlt for term in self.terms)

    def get_rewards(self, **kwargs):
        return {
            term.get_name(): term.compute_rewards(**kwargs)
            for term in self.terms
        }

    def get_n(self):
        n = 0
        for term in self.terms:
            try:
                n = max(n, term.n)
            except AttributeError:
                pass
        return n

    def __str__(self):
        return 'Reward Function:\n' + ',\n'.join('\t' + str(term) for term in self.terms)


class RewardTerm:
    def __init__(self, require_simlt):
        self.require_simlt = require_simlt

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def compute_rewards(self, **kwargs):
        pass


class Playability(RewardTerm):
    def __init__(self, magnitude=1):
        super(Playability, self).__init__(True)
        self.magnitude=magnitude

    def compute_rewards(self, **kwargs):
        simlt_res = kwargs['simlt_res']
        return [0 if item['playable'] else -self.magnitude for item in simlt_res[1:]]

    def __str__(self):
        return f'{self.magnitude} * Playability'


class MeanDivergenceFun(RewardTerm):
    def __init__(self, goal_div=defaults['gl'], n=defaults['n'], s=8):
        super().__init__(False)
        self.l = goal_div * 0.26 / 0.6
        self.u = goal_div * 0.94 / 0.6
        self.n = n
        self.s = s
        # self.weight_revise = weight_revise

    def compute_rewards(self, **kwargs):
        segs = kwargs['segs']
        rewards = []
        for i in range(1, len(segs)):
            seg = segs[i]
            histroy = lvlhcat(segs[max(0, i - self.n): i])
            k = 0
            divergences = []
            while k * self.s <= (min(self.n, i) - 1) * MarioLevel.seg_width:
                cmp_seg = histroy[:, k * self.s: k * self.s + MarioLevel.seg_width]
                # print(j, k, cmp_seg.shape)
                divergences.append(tile_pattern_js_div(seg, cmp_seg))
                k += 1
            mean_d = sum(divergences) / len(divergences)
            if mean_d < self.l:
                rewards.append(-(mean_d - self.l) ** 2)
            elif mean_d > self.u:
                rewards.append(-(mean_d - self.u) ** 2)
            else:
                rewards.append(0)
            # if self.weight_revise:
            #     rewards[-1] = max(1, (self.n - i + 1)) * rewards[-1]
        return rewards


class SACNovelty(RewardTerm):
    def __init__(self, magnitude, goal_div, require_simlt, n):
        super().__init__(require_simlt)
        self.g = goal_div
        self.magnitude = magnitude
        self.n = n
        self.mean_div = 0.
        # self.weight_revise = weight_revise

    def compute_rewards(self, **kwargs):
        n_segs = len(kwargs['segs'])
        rewards = []
        tmp = []
        for i in range(1, n_segs):
            reward = 0
            r_sum = 0
            for k in range(1, self.n + 1):
                j = i - k
                if j < 0:
                    break
                r = (1 - k / (self.n + 1))
                r_sum += r
                div = self.disim(i, j, **kwargs)
                tmp.append(div)
                reward += a_clip(div, self.g, r)
            rewards.append(reward * self.magnitude / r_sum)
            # if self.weight_revise:
            #     rewards[-1] = max(1, (self.n - i + 1)) * rewards[-1]
        self.mean_div = np.mean(tmp)
        return rewards

    @abstractmethod
    def disim(self, i, j, **kwargs):
        pass


class LevelSACN(SACNovelty):
    def __init__(self, magnitude=1, g=defaults['gl'], w=defaults['wl'], n=defaults['n']):
        super(LevelSACN, self).__init__(magnitude, g, False, n)
        self.w = w

    def disim(self, i, j, **kwargs):
        segs = kwargs['segs']
        seg1, seg2 = segs[i], segs[j]
        # print(lvlhcat([seg1, seg2]))
        return tile_pattern_js_div(seg1, seg2, self.w)

    def __str__(self):
        s = f'{self.magnitude} * LevelSACN(g={self.g:.3g}, w={self.w}, n={self.n})'
        # if self.weight_revise:
        #     s += ', weight revised'
        return s


class GameplaySACN(SACNovelty):
    def __init__(self, magnitude=1, g=defaults['gg'], w=defaults['wg'], n=defaults['n'], side=0):
        super(GameplaySACN, self).__init__(magnitude, g, True, n)
        self.w = w

    def disim(self, i, j, **kwargs):
        simlt_res = kwargs['simlt_res']
        trace1, trace2 = simlt_res[i]['trace'], simlt_res[j]['trace']
        return trace_div(trace1, trace2, self.w)

    def __str__(self):
        s = f'{self.magnitude} * GameplaySACN(g={self.g:.3g}, w={self.w}, n={self.n})'
        # if self.weight_revise:
        #     s += ', weight revised'
        return s

