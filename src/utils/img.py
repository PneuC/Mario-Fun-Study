"""
  @Time : 2022/3/10 11:22 
  @Author : Ziqi Wang
  @File : img.py 
"""

import pygame.image
from math import ceil
from src.utils.filesys import getpath


def vsplit_img(img, w, lw=6, lcolor='white', save_path='./image.png'):
    n = ceil(img.get_width() / w)
    h = img.get_height()
    canvas = pygame.Surface((img.get_width() + (n-1) * lw, h))
    canvas.fill(lcolor)
    for i in range(n):
        canvas.blit(img.subsurface(i*w, 0, min(w, img.get_width()-i*w), h), (i*(w + lw), 0))
    pygame.image.save(canvas, getpath(save_path))
    pass


def make_img_sheet(imgs, ncols, x_margin=6, y_margin=6, margin_color='white', save_path='./image.png'):
    nrows = ceil(len(imgs) / ncols)
    w, h = imgs[0].get_size()

    canvas = pygame.Surface((
        (w + x_margin) * ncols - x_margin, (h + y_margin) * nrows - y_margin
    ))
    canvas.fill(margin_color)
    for i in range(len(imgs)):
        row_id, col_id = i // ncols, i % ncols
        canvas.blit(imgs[i], ((w + x_margin) * col_id, (h + y_margin) * row_id))

    pygame.image.save(canvas, getpath(save_path))

