"""
  @Time : 2021/9/8 17:05 
  @Author : Ziqi Wang
  @File : smb.py
"""
import glob
import json
import os

import jpype
from math import ceil
from enum import Enum
from root import PRJROOT
from jpype import JString, JInt, JBoolean, JLong
from typing import Union, Dict
from src.smb.level import MarioLevel, LevelRender
from src.utils.filesys import getpath

JVMPath = None
# JVMPath = '/home/cseadmin/java/jdk1.8.0_301/jre/lib/amd64/server/libjvm.so'


class MarioJavaAgents(Enum):
    Runner = 'agents.robinBaumgarten'
    Killer = 'agents.killer'
    Collector = 'agents.collector'

    def __str__(self):
        return self.value + '.Agent'


class MarioProxy:
    # __jmario = jpype.JClass("MarioProxy")()

    def __init__(self):
        if not jpype.isJVMStarted():
            jarPath = getpath('smb/Mario-AI-Framework.jar')
            # print(f"-Djava.class.path={jarPath}/Mario-AI-Framework.jar")
            jpype.startJVM(
                jpype.getDefaultJVMPath() if JVMPath is None else JVMPath,
                f"-Djava.class.path={jarPath}", '-Xmx2g'
            )
            """
                -Xmx{size} set the heap size.
            """
        jpype.JClass("java.lang.System").setProperty('user.dir', f'{PRJROOT}/smb')
        self.__proxy = jpype.JClass("MarioProxy")()

    @staticmethod
    def extract_res(jresult, get_lives=False):
        res = {
            'status': str(jresult.getGameStatus().toString()),
            'completing-ratio': float(jresult.getCompletionPercentage()),
            '#kills': int(jresult.getKillsTotal()),
            '#kills-by-fire': int(jresult.getKillsByFire()),
            '#kills-by-stomp': int(jresult.getKillsByStomp()),
            '#kills-by-shell': int(jresult.getKillsByShell()),
            'trace': [
                [float(item.getMarioX()), float(item.getMarioY())]
                for item in jresult.getAgentEvents()
            ],
            # 'JAgentEvents': jresult.getAgentEvents()
        }
        if get_lives:
            res['lives'] = int(jresult.getLives())
        return res

    def play_game(self, level: Union[str, MarioLevel], lives=0, verbose=False, scale=2):
        if type(level) == str:
            level = MarioLevel.from_file(level)
        jresult = self.__proxy.playGame(JString(str(level)), JInt(lives), JBoolean(verbose), JInt(scale))
        return MarioProxy.extract_res(jresult)

    def simulate_game(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Runner,
        render: bool=False,
        realTimeLim: int = 0
    ) -> Dict:
        """
        Run simulation with an agent for a given level
        :param level: if type is str, must be path of a valid level file.
        :param agent: type of the agent.
        :param render: render or not.
        :param realTimeLim: Real-time limit, in unit of second.
        :return: dictionary of the results.
        """
        # start_time = time.perf_counter()
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_file(level)
        # real_time_limit_ms = 2 * (level.w * 15 + 1000)
        # real_time_limit_ms = level.w * 115 + 1000 if not render else 200000
        fps = 24 if render else 0
        jresult = self.__proxy.simulateGame(JString(str(level)), jagent, JBoolean(render), JInt(fps), JLong(realTimeLim * 1000))
        # Refer to Mario-AI-Framework.engine.core.MarioResult, add more entries if need be.
        return MarioProxy.extract_res(jresult)

    def simulate_complete(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Runner,
        segTimeK: int=100
    ) -> Dict:
        # start_time = time.perf_counter()
        ts = LevelRender.tex_size
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_file(level)
        reached_tile = 0
        res = {'restarts': [], 'full_trace': []}
        dx = 0
        win = False
        while not win and reached_tile < level.w - 1:
            jresult = self.__proxy.simulateWithSegmentwiseTimeout(
                JString(str(level[:, reached_tile:])), jagent, JInt(segTimeK))
            pyresult = MarioProxy.extract_res(jresult)
            reached = pyresult['trace'][-1][0]
            reached_tile += ceil(reached / ts)
            if pyresult['status'] != 'WIN':
                res['restarts'].append(reached_tile)
            else:
                win = True
            res['full_trace'] += [[dx + item[0], item[1]] for item in pyresult['trace']]
            dx = reached_tile * ts
        return res

    @staticmethod
    def get_seg_infos(full_info, check_points=None):
        restarts, trace = full_info['restarts'], full_info['full_trace']
        W = MarioLevel.seg_width
        ts = LevelRender.tex_size
        if check_points is None:
            end = ceil(trace[-1][0] / ts)
            check_points = [x for x in range(W, end, W)]
            check_points.append(end)
        res = [{'trace': [], 'playable': True} for _ in check_points]
        s, e, i = 0, 0, 0
        restart_pointer = 0
        # dx = 0
        while True:
            while e < len(trace) and trace[e][0] < ts * check_points[i]:
                if restart_pointer < len(restarts) and restarts[restart_pointer] < check_points[i]:
                    res[i]['playable'] = False
                    restart_pointer += 1
                e += 1
            x0 = trace[s][0]
            res[i]['trace'] = [[item[0] - x0, item[1]] for item in trace[s:e]]
            # x0, y0 = simlt_res[s]
            # data[j]['simlt_res'] = [[item[0] - x0, item[1] - y0] for item in simlt_res[s:e]]
            # dx = ts * check_points[j]
            i += 1
            if i == len(check_points):
                break
            s = e
        return res

    @staticmethod
    def save_rep(path, JAgentEvents):
        # tmp = jpype.JClass("agents.replay.ReplayAgent")()
        print(type(JAgentEvents))
        jpype.JClass("agents.replay.ReplayUtils").saveReplay(JString(getpath(path)), JAgentEvents)

    def replay(self, level, filepath, lives=0, visuals=True):
        # replay_agent = jpype.JClass("agents.replay.ReplayUtils").repAgentFromFile(get_path('levels/train/mario-1-1.rep'))
        if type(level) == MarioLevel:
            jres = self.__proxy.replayGame(JString(str(level)), JString(getpath(filepath)), JInt(lives), JBoolean(visuals))
        else:
            jres = self.__proxy.replayGame(JString(level), JString(getpath(filepath)), JInt(lives), JBoolean(visuals))
        return MarioProxy.extract_res(jres)


class MarioSurveyProxy:
    # __jmario = jpype.JClass("MarioProxy")()

    def __init__(self):
        jarPath = getpath('smb/Mario-AI-Interface.jar')
        # print(f"-Djava.class.path={jarPath}/Mario-AI-Framework.jar")
        jpype.startJVM(
            jpype.getDefaultJVMPath() if JVMPath is None else JVMPath,
            f"-Djava.class.path={jarPath}"
        )
        jpype.JClass("java.lang.System").setProperty('user.dir', f'{PRJROOT}/smb')
        self.__proxy = jpype.JClass("MarioSurveyProxy")()

    def reproduce(self, lvl_path, rep_path):
        jres = self.__proxy.reproduceGameResults(JString(getpath(lvl_path)), JString(getpath(rep_path)))
        return MarioProxy.extract_res(jres, True)


if __name__ == '__main__':
    simulator = MarioSurveyProxy()
    # res = simulator.reproduce(
    #     'lvls/Collector-21.lvl',
    #     'exp_data/survey data/2023-03-29_LOG_innertest/reps/e13f1fc9-59f4-4585-aa3b-eb807afb3db9Collector-21.rep'
    # )
    # print(res)

    rep_folder = getpath('exp_data/survey data/formal/data/reps')
    lvl_folder = getpath('exp_data/survey data/formal/lvls')
    for rep_file in os.listdir(rep_folder):
        print(rep_file)
        p_lvl = rep_file.find('lvl')
        p_collector = rep_file.find('Collector')
        p_killer = rep_file.find('Killer')
        p_runner = rep_file.find('Runner')
        p_ext = rep_file.find('.rep')
        lvl_path = ''
        if p_lvl >= 0:
            lvl_path = lvl_folder + '/' + rep_file[p_lvl:p_ext] + '.lvl'
        elif p_collector >= 0:
            lvl_path = lvl_folder + '/' + rep_file[p_collector:p_ext] + '.lvl'
        elif p_killer >= 0:
            lvl_path = lvl_folder + '/' + rep_file[p_killer:p_ext] + '.lvl'
        elif p_runner >= 0:
            lvl_path = lvl_folder + '/' + rep_file[p_runner:p_ext] + '.lvl'
        print(lvl_path)
        if os.path.exists(getpath('exp_data/survey data/formal/res-reproduce/%s.json' % rep_file[:p_ext])):
            continue

        try:
            result = simulator.reproduce(lvl_path, f'{rep_folder}/{rep_file}')
            with open(getpath('exp_data/survey data/formal/res-reproduce/%s.json' % rep_file[:p_ext]), 'w') as fp:
                json.dump(result, fp)
        except Exception:
            pass
    pass

