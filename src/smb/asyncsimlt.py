"""
  @Time : 2022/9/23 19:03 
  @Author : Ziqi Wang
  @File : asyncsimlt.py
"""

import time
# import importlib
import multiprocessing as mp
from src.smb.level import *
from src.smb.proxy import MarioProxy
from queue import Queue, Full as FullExpection
from src.env.rfunc import LevelSACN, GameplaySACN


def _simlt_worker(remote, parent_remote):
    # rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(rfunc_name)()
    # W = MarioLevel.seg_width
    fl, fg = LevelSACN(), GameplaySACN()
    simulator = MarioProxy()
    parent_remote.close()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'evaluate':
                token, strlvl = data
                lvl = MarioLevel(strlvl)
                segs = lvl.to_segs()
                simlt_res = MarioProxy.get_seg_infos(simulator.simulate_complete(lvl))
                # rewards = rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
                fl_vals = fl.compute_rewards(segs=segs)
                fg_vals = fg.compute_rewards(segs=segs, simlt_res=simlt_res)
                remote.send((token, fl_vals, fg_vals))
                pass
            elif cmd == 'check_eval':
                token, strlvl = data
                lvl = MarioLevel(strlvl)
                standable = False
                for i in range(lvl.h):
                    if lvl[i,0] in MarioLevel.solidset:
                        standable = True
                        break
                if standable:
                    simlt_res = simulator.simulate_game(lvl, realTimeLim=lvl.w / 16)
                    if simlt_res['status'] != 'WIN':
                        remote.send((token, strlvl, None))
                    else:
                        kwargs = {'restarts': [], 'full_trace': simlt_res['trace']}
                        simlt_res = MarioProxy.get_seg_infos(kwargs)
                        segs = lvl.to_segs()
                        fl_val = np.mean(fl.compute_rewards(segs=segs))
                        fg_val = np.mean(fg.compute_rewards(segs=segs, simlt_res=simlt_res))
                        dl_val, dg_val = fl.mean_div, fg.mean_div
                        remote.send((token, strlvl, (fl_val, fg_val, dl_val, dg_val)))
                else:
                    remote.send((False, token))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise KeyError(f'Unknown command for simulation worker: {cmd}')
        except EOFError:
            break
    pass


class AsycSimltPool:
    def __init__(self, poolsize, queuesize=None, verbose=True):
        self.np, self.nq = poolsize, poolsize if queuesize is None else queuesize
        self.waiting_queue = Queue(self.nq)
        self.ready = [True] * poolsize
        self.__init_remotes()
        self.res_buffer = []
        # self.histlen = AsycSimltPool.get_histlen(rfunc_name)
        self.verbose = verbose

    # @staticmethod
    # def get_histlen(rfunc_name):
    #     rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(rfunc_name)()
    #     return rfunc.get_n()
    #     pass

    def put(self, cmd, args):
        """
            Put a new evaluation task into the pool. If the pool and waiting queue is full,
            wait until a process is free
        """
        putted = False
        for i, remote in enumerate(self.remotes):
            if self.ready[i]:
                remote.send((cmd, args))
                self.ready[i] = False
                putted = True
                break
        while not putted:
            try:
                self.waiting_queue.put((cmd, args), timeout=0.01)
                putted = True
            except FullExpection:
                self.refresh()

    def get(self, wait=False):
        if wait:
            self.__wait()
        self.refresh()
        occp, occq = self.get_occupied()
        if self.verbose:
            print(f'Workers: {occp}/{self.np}, Queue: {occq}/{self.nq}, Buffer: {len(self.res_buffer)}')
        res = self.res_buffer
        self.res_buffer = []
        return res

    def get_occupied(self):
        process_occupied = sum(0 if r else 1 for r in self.ready)
        return process_occupied, self.waiting_queue.qsize()

    def refresh(self):
        """ Recive ready results and cache them in buffer, then assign tasks in waiting queue to free workers """
        for i, remote in enumerate(self.remotes):
            if remote.poll():
                self.res_buffer.append(remote.recv())
                self.ready[i] = True
        for i, remote in enumerate(self.remotes):
            if self.waiting_queue.empty():
                break
            if self.ready[i]:
                cmd, args = self.waiting_queue.get()
                remote.send((cmd, args))
                self.ready[i] = False

    def blocking(self):
        self.refresh()
        return self.waiting_queue.full()

    def __init_remotes(self):
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.np)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_simlt_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

    def __wait(self):
        finish = False
        while not finish:
            self.refresh()
            finish = all(r for r in self.ready)
            time.sleep(0.01)

    def terminate(self):
        for p in self.processes:
            p.terminate()


    def close(self):
        # finish = False
        # while not finish:
        #     self.refresh()
        #     finish = all(r for r in self.ready)
        #     time.sleep(0.01)
        # self.__wait()
        res = self.get(True)
        for remote, p in zip(self.remotes, self.processes):
            remote.send(('close', None))
            p.join()
        return res