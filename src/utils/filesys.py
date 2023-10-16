"""
  @Time : 2022/3/10 11:16 
  @Author : Ziqi Wang
  @File : filesys.py 
"""
import os
import re
import json
import pandas as pds
from root import PRJROOT


def auto_dire(path=None, name='trial'):
    dire_id = 0
    prefix = PRJROOT if path is None else f'{getpath(path)}/'
    tar = f'{prefix}{name}{dire_id}'
    while os.path.exists(tar):
        dire_id += 1
        tar = f'{prefix}{name}{dire_id}'
    os.makedirs(tar)
    return tar


def getpath(*args):
    """ if is absolute root_folder or related path, return {path}, else return {PRJROOT/path} """
    path = os.path.join(*args)
    if os.path.isabs(path) or re.match(r'\.+[\\/].*', path):
        return path
    else:
        return os.path.join(PRJROOT, path)

def load_dict_json(path, *keys):
    with open(getpath(path), 'r') as f:
        data = json.load(f)
    if len(keys) == 1:
        return data[keys[0]]
    return tuple(data[key] for key in keys)

def load_singlerow_csv(path, *keys):
    data = pds.read_csv(getpath(path))
    if len(keys) == 1:
        return data[keys[0]][0]
    return (data[key][0] for key in keys)

