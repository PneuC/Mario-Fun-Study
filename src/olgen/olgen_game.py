"""
  @Time : 2022/4/19 15:18 
  @Author : Ziqi Wang
  @File : olgen_game.py
"""

import jpype
import pygame as pg
import multiprocessing as mp
from root import PRJROOT
from src.smb.proxy import JVMPath
from src.smb.level import MarioLevel
from src.olgen.designer import Designer
from src.gan.gankits import get_generator
from src.olgen.ol_generator import OnlineGenerator
from src.utils.filesys import getpath


def _ol_gen_worker(remote, parent_remote, d_path, g_path, g_device):
    parent_remote.close()
    designer = Designer(f'{d_path}/actor.pth')
    generator = get_generator('models/generator.pth') if g_path == '' else get_generator(getpath(g_path))
    ol_generator = OnlineGenerator(designer, generator, g_device)
    remote.send(ol_generator.step())
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(ol_generator.step())
            elif cmd == "close":
                remote.close()
                break
        except EOFError:
            break
    pass


class MarioOnlineGenGame:
    def __init__(self, d_path, g_path='', g_device='cuda:0'):
        if not jpype.isJVMStarted():
            jpype.startJVM(
                jpype.getDefaultJVMPath() if JVMPath is None else JVMPath,
                f"-Djava.class.path={PRJROOT}Mario-AI-Framework.jar", '-Xmx4g'
            )
        self.d_path, self.g_path, self.g_device = d_path, g_path, g_device
        self.ol_gen_remote, self.process = None, None

    def play(self, max_length):
        self.__init_ol_gen_remote()
        seg_str = self.ol_gen_remote.recv()
        print(seg_str)
        game = jpype.JClass("MarioOnlineGenGame")(jpype.JString(seg_str))
        clk = pg.time.Clock()
        finish = False
        n_seg = 1
        self.ol_gen_remote.send(('step', None))
        while not finish:
            finish = bool(game.gameStep())
            if n_seg < max_length and int(game.getTileDistantToExit()) < MarioLevel.default_seg_width:
                seg_str = self.ol_gen_remote.recv()
                game.appendSegment(jpype.JString(seg_str))
                n_seg += 1
                self.ol_gen_remote.send(('step', None))
            clk.tick(30)
        self.close()

    def __init_ol_gen_remote(self):
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)
        self.ol_gen_remote, work_remote = ctx.Pipe()
        args = (work_remote, self.ol_gen_remote, self.d_path, self.g_path, self.g_device)
        self.process = ctx.Process(target=_ol_gen_worker, args=args, daemon=True)
        self.process.start()

    def close(self):
        self.ol_gen_remote.send(('close', None))
        self.process.join()

