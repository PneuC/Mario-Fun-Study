"""
  @Time : 2022/7/18 11:16 
  @Author : Ziqi Wang
  @File : level.py 
"""

import re
import glob
from math import nan
import numpy as np
import pygame as pg
from dtw import dtw
from itertools import product
from src.utils.mymath import jsdiv
from src.utils.filesys import getpath

'''
    Encoding:
    X ( 0) -> Solid tile
    S ( 1) -> Breakable block
    - ( 2) -> Empty tile
    % ( 3) -> Mushroom platform
    t ( 4) -> Normal tube
    T ( 5) -> Flower tube
    b ( 6) -> Bullet bill body
    B ( 7) -> Bullet bill head
    o ( 8) -> Coin
    Q ( 9) -> Coin question block
    @ (10) -> Mushroom question block
    U (11) -> Mushroom breakable block
    L (12) -> 1UP block
    1 (13) -> Invisible 1UP block
    2 (14) -> Invisible coin block
    g (15) -> Goomba
    k (16) -> Koopa (green)
    r (17) -> Koopa (red)
    K (18) -> Flying Koopa (green)
    R (19) -> Flying Koppa (red)
    y (20) -> Spiky
    | (--) -> Mushroom stalk
'''

class MarioLevel:
    height = 16
    seg_width = 16
    mapping = {
        'i-c': (
            'X', 'S', '-', '%', 't', 'T', 'b', 'B', 'o', 'Q', '@',
            'U', 'L', '1', '2', 'g', 'k', 'r', 'K', 'R', 'y'
        ),
        'c-i': {
            'X': 0, 'S': 1, '-': 2, '%': 3, 't': 4, 'T': 5, 'b': 6, 'B': 7, 'o': 8, 'Q': 9, '@': 10,
            'U': 11, 'L': 12, '1': 13, '2': 14, 'g': 15, 'k': 16, 'r': 17, 'K': 18, 'R': 19, 'y': 20,
            '#': 0, '|': 2, 'F': 2, 'M': 2, 'C': 8, '!': 9, '?': 10, 'E': 15
        }
    }
    n_types = len(mapping['i-c'])
    pipeset = {'t', 'T', 'b', 'B'}
    enemyset = {'g', 'k', 'r', 'K', 'R', 'y'}
    solidset = {'X', '#', 'S', 't', 'T', '%', 'Q', '@'}

    def __init__(self, content):
        if isinstance(content, np.ndarray):
            self.content = content
        else:
            tmp = [list(line) for line in content.split('\n')]
            while not tmp[-1]:
                tmp.pop()
            self.content = np.array(tmp)
        self.h, self.w = self.content.shape
        self.__tile_pttr_cnts = {}
        self.attr_dict = {}

    def to_num_arr(self):
        res = np.zeros((self.h, self.w), int)
        for i, j in product(range(self.h), range(self.w)):
            char = self.content[i, j]
            res[i, j] = MarioLevel.mapping['c-i'][char]
        return res

    def to_img(self, save_path=None) -> pg.Surface:
        img = LevelRender.render(self)
        if save_path:
            safe_path = getpath(save_path)
            pg.image.save(img, safe_path)
        return img

    def to_img_with_trace(self, trace, save_path='render_with_trace.png', color='black', lw=3):
        img = LevelRender.render(self)
        LevelRender.draw_trace_on(img, trace, color, lw)
        if save_path is not None:
            pg.image.save(img, getpath(save_path))
        return img

    def save(self, fpath):
        safe_path = getpath(fpath)
        if safe_path[-4:] != '.lvl':
            safe_path += '.lvl'
        with open(safe_path, 'w') as f:
            f.write(str(self))

    def tile_pattern_counts(self, w=2):
        if not w in self.__tile_pttr_cnts.keys():
            counts = {}
            for i, j in product(range(self.h - w + 1), range(self.w - w + 1)):
                key = ''.join(self.content[i+x][j+y] for x, y in product(range(w), range(w)))
                count = counts.setdefault(key, 0)
                counts[key] = count + 1
            self.__tile_pttr_cnts[w] = counts
        return self.__tile_pttr_cnts[w]

    def tile_pattern_distribution(self, w=2):
        counts = self.tile_pattern_counts(w)
        C = (self.h - w + 1) * (self.w - w + 1)
        return {key: val / C for key, val in counts.items()}

    def __getattr__(self, item):
        if item == 'shape':
            return self.content.shape
        elif item == 'h':
            return self.content.shape[0]
        elif item == 'w':
            return self.content.shape[1]
        elif item not in self.attr_dict.keys():
            if item == 'n_gaps':
                empty_map1 = np.where(self.content[-1] in MarioLevel.empty_chars, 1, 0)
                empty_map2 = np.where(self.content[-2] in MarioLevel.empty_chars, 1, 0)
                res = len(np.where(empty_map1 + empty_map2 == 2))
                self.attr_dict['n_gaps'] = res
            elif item == 'n_enemies':
                sself = str(self)
                self.attr_dict['n_enemies'] = sum(sself.count(c) for c in MarioLevel.enemyset)
            elif item == 'n_coins':
                self.attr_dict['n_coins'] = str(self).count('o')
            elif item == 'n_questions':
                self.attr_dict['n_questions'] = str(self).count('Q')
            elif item == 'n_empties':
                empty_map = np.where(self.content in MarioLevel.empty_chars)
                self.attr_dict['n_questions'] = len(empty_map)
        return self.attr_dict[item]

    def count_gaps(self):
        longest = 0
        number = 0
        for i in range(self.w):
            length = 0
            if self[-1, i] == '-' and self[-2, i] == '-':
                length += 1
                if length == 1:
                    number += 1
                longest = max(longest, length)
            else:
                length = 0
        return number, longest

    def __str__(self):
        lines = [''.join(line) + '\n' for line in self.content]
        return ''.join(lines)

    def __add__(self, other):
        concated = np.concatenate([self.content, other.content], axis=1)
        return MarioLevel(concated)

    def __getitem__(self, item):
        try:
            content = self.content[item]
            if type(content) == np.ndarray:
                return MarioLevel(self.content[item])
            else:
                return str(content)
        except IndexError:
            return None

    def to_segs(self):
        W = MarioLevel.seg_width
        return [self[:, s:s+W] for s in range(0, self.w, W)]

    def copy(self):
        return MarioLevel.from_num_arr(self.to_num_arr())

    @staticmethod
    def from_num_arr(num_arr):
        h, w = num_arr.shape
        res = np.empty((h, w), str)
        for i, j in product(range(h), range(w)):
            if num_arr[i, j] == 0:
                res[i, j] = 'X' if i >= MarioLevel.height - 2 else '#'
            else:
                tile_id = num_arr[i, j]
                if type(tile_id) != int:
                    tile_id = round(tile_id)
                res[i, j] = MarioLevel.mapping['i-c'][tile_id]
        visited = set()
        for i, j in product(range(h), range(w)):
            if res[i, j] == '%':
                if f'{i}-{j}' in visited:
                    continue
                s, e = j, j
                while e < w and res[i, e] == '%':
                    visited.add(f'{i}-{e}')
                    e += 1
                if (e - s) <= 2:
                    stalk_cols = range(s, e)
                else:
                    stalk_cols = range(s + 1, e - 1)
                for q in stalk_cols:
                    p = i + 1
                    while p < h and res[p, q] not in MarioLevel.solidset:
                        if res[p, q] == '-':
                            res[p, q] = '|'
                        p += 1
        return MarioLevel(res)

    @staticmethod
    def from_file(fpath):
        safe_path = getpath(fpath)
        with open(safe_path, 'r') as f:
            return MarioLevel(f.read())

    @staticmethod
    def from_one_hot_arr(one_hot_arr: np.ndarray):
        num_lvl = one_hot_arr.argmax(axis=0)
        return MarioLevel.from_num_arr(num_lvl)


class LevelRender:
    # BG_COLOR = (109, 143, 252)
    BG_COLOR = (138, 165, 253)
    tubeset = {'t', 'T'}
    tex_size = 16
    textures = {
        re.split('[/\\\\]',fpath)[-1][:-4]: pg.image.load(fpath)
        for fpath in glob.glob(getpath('smb/assets/*.png'))
    }

    @staticmethod
    def render(level):
        ts = LevelRender.tex_size
        img = pg.Surface((level.w * ts, level.h * ts))
        # img.fill((109, 143, 252))
        img.fill(LevelRender.BG_COLOR)
        reconded_lvl = MarioLevel.from_num_arr(level.to_num_arr())
        j_t_platforms, tubes, chompers = LevelRender.__get_objects(reconded_lvl)
        LevelRender.__render_objects(img, j_t_platforms, tubes, chompers, reconded_lvl)
        LevelRender.__render_tiles(img, reconded_lvl)
        return img

    @staticmethod
    def __get_objects(level):
        h, w = level.shape
        visited = set()
        j_t_platforms = []
        tubes = []
        chompers = []
        for i, j in product(range(h), range(w)):
            c = level[i, j]
            if f'{i}-{j}' in visited:
                continue
            if c == '%':
                s, e = j, j
                while level[i, e] == '%':
                    visited.add(f'{i}-{e}')
                    e += 1
                j_t_platforms.append({'row': i, 'col-start': s, 'col-end': e})
            if c == 'T' and level[i, j - 1] not in LevelRender.tubeset and level[i, j + 1] in LevelRender.tubeset:
                chompers.append((i - 1, j))
            if c in LevelRender.tubeset:
                single = level[i, j + 1] not in LevelRender.tubeset
                start = (i, j)
                left_height = 0
                right_height = None if single else 0
                visited.add(f'{i}-{j}')
                if not single:
                    visited.add(f'{i}-{j+1}')
                while level[i + left_height, j] in LevelRender.tubeset:
                    visited.add(f'{i + left_height}-{j}')
                    left_height += 1
                if not single:
                    while level[i + right_height, j + 1] in LevelRender.tubeset:
                        visited.add(f'{i + right_height}-{j+1}')
                        right_height += 1
                tubes.append({'start': start, 'left-height': left_height, 'right-height': right_height})
        return j_t_platforms, tubes, chompers

    @staticmethod
    def __render_objects(img, j_t_platforms, tubes, chompers, level):
        ts = LevelRender.tex_size
        textures = LevelRender.textures
        for j_t_platform in j_t_platforms:
            row, col_start, col_end = j_t_platform['row'], j_t_platform['col-start'], j_t_platform['col-end']
            stalk_start, stalk_end = col_start, col_end
            if col_end - col_start == 1:
                img.blit(textures['MS'], (col_start * ts, row * ts))
            else:
                img.blit(textures['ML'], (col_start * ts, row * ts))
                img.blit(textures['MR'], ((col_end - 1) * ts, row * ts))
                for j in range(col_start + 1, col_end - 1):
                    img.blit(textures['MM'], (j * ts, row * ts))
            if col_end - col_start > 2:
                stalk_start += 1
                stalk_end -= 1
            for j in range(stalk_start, stalk_end):
                i = row + 1
                while i < level.h and level[i, j] not in MarioLevel.solidset:
                    img.blit(textures['stalk'], (j * ts, i * ts))
                    i += 1
        for chomper in chompers:
            i, j = chomper
            img.blit(textures['chomper'], ((j + 0.5) * ts, i * ts))
        for tube in tubes:
            (i, j), left_height, right_height = tube['start'], tube['left-height'], tube['right-height']
            if right_height is None:
                img.blit(textures['TSP'], (j * ts, i * ts))
                for k in range(1, left_height):
                    img.blit(textures['BSP'], (j * ts, (i + k) * ts))
            else:
                img.blit(textures['TLP'], (j * ts, i * ts))
                img.blit(textures['TRP'], ((j + 1) * ts, i * ts))
                for k in range(1, left_height):
                    img.blit(textures['['], (j * ts, (i + k) * ts))
                for k in range(1, left_height):
                    img.blit(textures[']'], ((j + 1) * ts, (i + k) * ts))

    @staticmethod
    def __render_tiles(img, level):
        ts = LevelRender.tex_size
        for i, j in product(range(level.h), range(level.w)):
            target = (j * ts, i * ts, ts, ts)
            tile = level[i, j]
            if tile in {'-', 't', 'T', '%', '|', 'F', 'M'}:
                continue
            elif tile == 'b':
                t = level[i - 1, j]
                if t == 'B':
                    img.blit(LevelRender.textures['CB1'], target)
                else:
                    img.blit(LevelRender.textures['CB2'], target)
            elif tile == 'K':
                img.blit(LevelRender.textures['wingk'], target)
            elif tile == 'R':
                img.blit(LevelRender.textures['wingr'], target)
            else:
                img.blit(LevelRender.textures[tile], target)

    @staticmethod
    def draw_trace_on(lvlimg, trace, color='black', lw=3):
        p = 0
        while p < len(trace) and trace[p][0] < lvlimg.get_width():
            p += 1
        pg.draw.lines(lvlimg, color, False, [(x, y-8) for x, y in trace[:p]], lw)
        return lvlimg

def trace_div(trace1, trace2, w=10):
    h, ts = MarioLevel.height, LevelRender.tex_size
    t1, t2 = np.array(trace1) / ts, np.array(trace2) / ts
    dist_metric = (lambda x, y: np.linalg.norm(x - y))
    dtw_val, *_ = dtw(t1, t2, dist_metric, w=max(w, abs(len(t1) - len(t2))))
    return dtw_val / (h * MarioLevel.seg_width)

def tile_pattern_js_div(seg1: MarioLevel, seg2: MarioLevel, w=2):
    counts1 = seg1.tile_pattern_counts(w)
    counts2 = seg2.tile_pattern_counts(w)
    all_keys = counts1.keys().__or__(counts2.keys())
    p = np.array([counts1.setdefault(key, 0) for key in all_keys])
    q = np.array([counts2.setdefault(key, 0) for key in all_keys])
    return jsdiv(p, q)

def lvl_manhhatan_div(seg1: MarioLevel, seg2: MarioLevel):
    if seg1.h != seg2.h or seg1.w != seg2.w:
        return nan
    size = seg1.h * seg1.w
    return np.sum(np.where(seg1.content != seg2.content, 1, 0)) / size

def normalised_hamming_dis(seg1, seg2):
    assert seg1.h == seg2.h and seg1.w == seg2.w
    return np.where(seg1.content != seg2.content, 1, 0).sum() / (seg1.h * seg1.w)

def lvlhcat(lvls) -> MarioLevel:
    if type(lvls[0]) == MarioLevel:
        concated_content = np.concatenate([l.content for l in lvls], axis=1)
    else:
        concated_content = np.concatenate([l for l in lvls], axis=1)
    return MarioLevel(concated_content)

def traverse_level_files(path):
    for lvl_path in glob.glob(getpath(f'{path}/*.lvl')):
        lvl = MarioLevel.from_file(lvl_path)
        name = re.split('[/\\\\]', lvl_path)[-1][:-4]
        yield lvl, name

def traverse_batched_level_files(path):
    for lvl_path in glob.glob(getpath(f'{path}\\*.lvls')):
        name = re.split('[/\\\\]', lvl_path)[-1][:-5]
        with open(lvl_path, 'r') as f:
            txt = f.read()
        lvls = []
        for lvlstr in txt.split(';\n'):
            if len(lvlstr) < 10:
                continue
            lvls.append(MarioLevel(lvlstr))
        yield lvls, name

def save_batch(lvls, fname):
    contents = [str(lvl).strip() for lvl in lvls]
    content = '\n;\n'.join(contents)
    if len(fname) <= 5 or fname[-5:] != '.lvls':
        fname += '.lvls'
    with open(getpath(fname), 'w') as f:
        f.write(content)
    pass

def load_batch(fname):
    with open(getpath(fname), 'r') as f:
        content = f.read()
    return [MarioLevel(c) for c in content.split('\n;\n')]


if __name__ == '__main__':
    lvl = MarioLevel.from_file('./log.lvl')
    lvl.to_img(getpath('./log.png'))
    # for lname, name in traverse_level_files('smb/levels'):
    #     lname.to_img(f'smb/levels_render/{name}.png')
    #     tmp = lname.to_num_arr()
    #     MarioLevel.from_num_arr(tmp).to_img(f'smb/levels_render/{name}-recoding.png')
    pass
