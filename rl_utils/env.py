import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections


def _allow(neighbor: int, mode: str) -> bool:
    """
    check if the neighbor is allowed to travel of the given mode: static, TG, GG, GSD, TS
    :param neighbor: int, element of a list of 9 elements, the 4-th element is the grid itself, the other 8 elements are the neighbors
    :param mode: str, 'TG', 'GG', 'GSD', 'static'
    :return: bool, True if the neighbor is allowed to travel of the given mode, False otherwise
    """
    if mode == 'TG' or mode == 'static':
        return neighbor>>1 & 1 == 1
    elif mode == 'GG':
        return neighbor>>3 & 1 == 1
    elif mode =='GSD':
        return neighbor>>2 & 1 == 1 or neighbor>>5 & 1 == 1
    elif mode == 'TS':
        return neighbor>>6 & 1 == 1 or neighbor>>1 & 1 == 1
    elif mode == 'static':
        return True


class MapEnv:
    def __init__(self, mapdata:dict, traj:pd.DataFrame, test_mode=False, testid_start=0, test_num=8, train_num = 13):
        self.mapdata = mapdata
        self.traj = traj
        self.hashmap = set()
        self.step_cnt = 0
        self.train_num = train_num

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num

        self.traj_cnt = 0 # TRAJCNT is the index of the traj record + 2

    def reset(self):
        # reset env by using next two traj record
        # for example, 1st interation, start = traj[0], goal = traj[1]; 2nd interation, start = traj[1], goal = traj[2]...
        self.step_cnt = 0

        if self.isTest:
            mod = self.test_num
        else:
            mod = self.train_num

        if self.traj.loc[self.traj_cnt%mod, 'ID'] != self.traj.loc[self.traj_cnt%mod+1, 'ID']:
            self.traj_cnt += 1# if the mode is different, then reset the env
        locx_start = float(self.traj.loc[self.testid_start+self.traj_cnt%mod, 'locx'])
        locy_start = float(self.traj.loc[self.testid_start+self.traj_cnt%mod, 'locy'])
        locx_end = float(self.traj.loc[self.testid_start+self.traj_cnt%mod + 1 , 'locx'])
        locy_end = float(self.traj.loc[self.testid_start+self.traj_cnt%mod + 1, 'locy'])

        # when test model, using serval traj records
        self.traj_cnt += 1

        self.mode = self.traj.loc[self.traj_cnt%mod, 'mode']
        self.state = np.array([0,0])
        self.goal = np.array([locx_end - locx_start, locy_end - locy_start])
        # max step is the mahattan distance between start and goal
        self.max_step = np.abs(locx_start - locx_end) + np.abs(locy_start - locy_end)
        return np.hstack((self.state, self.goal))

    def step(self, action:int):
        # agent will move to 8 directions,action is tuple of (dx,dy)

        self.step_cnt += 1
        dxdy_dict = {0:(1,0), 1:(1,1), 2:(0,1), 3:(-1,1), 4:(-1,0), 5:(-1,-1), 6:(0,-1), 7:(1,-1)}
        d = dxdy_dict[action]
        self.state += np.array(d)

        # to encourage the agent travel in the shortest path
        reward = -1  if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) > 0 else 0

        if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) == 0 or self.step_cnt == 30:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map

        return np.hstack((self.state, self.goal)), reward, done