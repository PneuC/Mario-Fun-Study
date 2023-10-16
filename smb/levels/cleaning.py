"""
  @Time : 2022/7/29 10:33 
  @Author : Ziqi Wang
  @File : process.py 
"""

from itertools import product
from src.smb.level import traverse_level_files, MarioLevel


# def clean():
#     # unknows = set()
#     mappings = {'M': '-', 'U': '@', '?': '@', '!': 'Q', 'F': '-', 'C': 'o', 'E': 'g'}
#     levels_contain_cannon = set()
#     for lname, name in traverse_level_files('smb/levels/original'):
#         content = lname.content
#         h, w = lname.shape
#         for j, j in product(range(h), range(w)):
#             if content[j, j] == '*':
#                 levels_contain_cannon.add(name)
#
#             if content[j, j] in mappings.keys():
#                 content[j, j] = mappings[content[j, j]]
#         lname.save(f'smb/levels/processed/{name}')
#     print(levels_contain_cannon)

def check():
    unknowns = set()
    levels_has_star = set()
    for lvl, name in traverse_level_files('smb/levels'):
        content = lvl.content
        h, w = lvl.shape
        for i, j in product(range(h), range(w)):
            if content[i, j] not in MarioLevel.mapping['c-i'].keys():
                unknowns.add(content[i,j])
            if content[i,j] == '*':
                levels_has_star.add(name)
    print('Levels with *:', levels_has_star)
    print('Unknown tiles:', unknowns)


if __name__ == '__main__':
    check()
