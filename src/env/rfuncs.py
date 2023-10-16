"""
  @Time : 2022/1/4 10:03 
  @Author : Ziqi Wang
  @File : rfuncs.py 
"""

from src.env.rfunc import *


default = RewardFunc(LevelSACN(), GameplaySACN(), Playability())
fun_c = RewardFunc(LevelSACN(), Playability())
fun_b = RewardFunc(GameplaySACN(), Playability())
fun_cb = RewardFunc(LevelSACN(), GameplaySACN(), Playability())

f = RewardFunc(MeanDivergenceFun(), Playability())
l = RewardFunc(LevelSACN(), Playability())
g = RewardFunc(GameplaySACN(), Playability())
lg = RewardFunc(LevelSACN(), GameplaySACN(), Playability())
