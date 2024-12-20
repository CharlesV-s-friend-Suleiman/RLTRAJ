import torch
import numpy as np
import pandas as pd

from get_data_distribution import z_score
from rl_utils.tools import mapdata_to_modelmatrix, get_neighbor
from rl_utils.descrete_rl_methods import VAnet, Qnet, Policy

from scipy.stats import norm
import pickle


# glabal variables
dxdy_dict = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
mode_v_dict = {'TG': 172, 'GG': 78, 'GSD': 40, 'TS': 94} # expected speed in km/h
modelist = ['GSD', 'GG', 'TS', 'TG']
with open('data/vdistribution.pkl','rb') as f:
    processed_data = pickle.load(f)

mean_std_dict = {'GG':[], 'GSD':[], 'TS':[], 'TG':[]}
for mode in processed_data:
    mean_std_dict[mode] = [np.mean(processed_data[mode]), np.std(processed_data[mode])]



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
    def __init__(self, mapdata:dict, traj:pd.DataFrame,
                 trainid_start=0, test_mode=False, testid_start=0, test_num=8, train_num = 13,
                 use_real_map=False,realmap_row=326, realmap_col = 364,
                 is_lower=False, dummy_mode=None):

        self.traj = traj
        self.step_cnt = 0
        self.train_num = train_num
        self.trainid_start = trainid_start

        if use_real_map:
            self.mapdata = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)
        self.is_lower = is_lower # when is_lower is True, the env is used for lower model, and mode info is not used
        self.dummy_mode = dummy_mode

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num
        self.distance_hold =1 if test_mode else 1
        self.traj_cnt = 0 # TRAJCNT is the index of the traj record + 2

    def reset(self):
        # reset env by using next two traj record
        # for example, 1st interation, start = traj[0], goal = traj[1]; 2nd interation, start = traj[1], goal = traj[2]...
        self.step_cnt = 0

        if self.isTest:
            mod = self.test_num
            start_id = self.testid_start
        else:
            mod = self.train_num
            start_id = self.trainid_start

        if self.traj.loc[self.traj_cnt%mod, 'ID'] != self.traj.loc[self.traj_cnt%mod+1, 'ID']:
            self.traj_cnt += 1# if the mode is different, then reset the env
        locx_start = float(self.traj.loc[start_id+self.traj_cnt%mod, 'locx'])
        locy_start = float(self.traj.loc[start_id+self.traj_cnt%mod, 'locy'])
        #print(start_id, self.traj_cnt, mod, start_id+self.traj_cnt%mod, self.traj.loc[start_id+self.traj_cnt%mod, 'ID'])
        try:
            locx_end = float(self.traj.loc[start_id+self.traj_cnt%mod + 1 , 'locx'])
        except:
            print(start_id, self.traj_cnt, mod)
            raise ValueError('out of range')
        locy_end = float(self.traj.loc[start_id+self.traj_cnt%mod + 1, 'locy'])

        # when test lower_model, using serval traj records
        self.traj_cnt += 1

        self.mode = self.traj.loc[self.traj_cnt%mod, 'mode'] if not self.is_lower else self.dummy_mode
        # delta is the relative position of the start_position and 0,0; delta only change when start_position change(when reset)
        # neighbor is the 8 elements list of the grid not including itself, 0-8 are the neighbors from 1,0 to 1,-1
        self.neighbor = np.array(get_neighbor(self.mapdata[self.mode], locx_start, locy_start))
        self.delta = np.array([locx_start, locy_start])
        self.state = np.array([0,0])
        self.goal = np.array([locx_end - locx_start, locy_end - locy_start])
        # max step is the mahattan distance between start and goal
        self.max_step = np.abs(locx_start - locx_end) + np.abs(locy_start - locy_end)
        return np.hstack((self.state, self.goal, self.neighbor))

    def step(self, action:int):
        # agent will move to 8 directions,action is tuple of (dx,dy)
        reward = 0
        self.step_cnt += 1
        d = dxdy_dict[action]

        # update state of position
        self.state += np.array(d)

        # not in the available neighbor
        if self.neighbor[action]==0:
            reward -=0.33

        # update neighbor
        self.neighbor = np.array(get_neighbor(self.mapdata[self.mode], self.state[0]+self.delta[0], self.state[1]+self.delta[1]))

        # to encourage the agent travel in the shortest path
        reward -= 1 if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) > self.distance_hold else 0

        if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) == self.distance_hold or self.step_cnt == 100:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map
        return np.hstack((self.state, self.goal, self.neighbor)), reward, done

    def step_2d_action(self,action:tuple):
        # action is a tuple of (dx,dy) dx,dy denote to -1,0,1
        reward = 0

        self.step_cnt += 1
        self.state += np.array(action)
        real_x = self.state[0] + self.delta[0]
        real_y = self.state[1] + self.delta[1]
        # not in the available neighbor
        if self.mapdata[self.mode][real_x][real_y] == 0:
            reward -= 0.25
        # update neighbor
        self.neighbor = np.array(get_neighbor(self.mapdata[self.mode], real_x, real_y))
        # to encourage the agent travel in the shortest path
        reward -= 1 if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) > self.distance_hold else 0

        if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) == self.distance_hold or self.step_cnt == 100:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map
        return np.hstack((self.state, self.goal, self.neighbor)), reward, done



class UpperEnv:
    def __init__(self,mapdata:dict, traj:pd.DataFrame,
                 trainid_start=0, test_mode=False, testid_start=0,
                 test_num=8, train_num = 10000,
                 m = 5,
                 use_real_map=False,realmap_row=326, realmap_col = 364, lower_model_config=None):
        """

        :param mapdata: dict of 5 elements, key is the mode, value is the mapdata of the mode
        :param traj: traj data in csv format
        :param trainid_start: start index of the training data
        :param test_mode: ==True if test
        :param testid_start:  start index of the test data
        :param test_num: pass
        :param train_num: pass
        :param use_real_map: default is False
        :param realmap_row: pass
        :param realmap_col: pass
        :param m: the hyperpara of steps to calculate the avg reward
        :param lower_model_config: dict of lower model(qnet) size key=state_dim, hidden_dim, action_dim,model_path
        """
        self.step_cnt=0
        self.traj = traj
        self.traj_idx = 0
        self.train_num = train_num
        self.trainid_start = trainid_start
        self.m = m
        self.max_step  = 10 # max step of the upper model
        if use_real_map:
            self.mapdata = mapdata
            self.mapmatrice = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num

        if self.isTest:
            self.mod = self.test_num
            start_id = self.testid_start
        else:
            self.mod = self.train_num
            start_id = self.trainid_start

        # import lower model
        try:
            state_dim_lower = lower_model_config['state_dim']
            hidden_dim_lower = lower_model_config['hidden_dim']
            action_dim_lower = lower_model_config['action_dim']
            model_path = lower_model_config['model_path']
        except:
            raise ValueError('lower_model_config is not correct, check the key of the dict or model path')

        lower_agent = VAnet(state_dim_lower, hidden_dim_lower, action_dim_lower)
        lower_agent.load_state_dict(torch.load(model_path))
        lower_agent.eval()
        self.lower_agent = lower_agent

    def step(self, action:int):
        """

        :param action:int , [0, 1, 2, 3] denotes 4 modes: [GSD, GG, TS, TG]
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TS', 'TG']
        upper_mode = action_mode_duels[action]
        reward = 0

        # t_lower is the time cost of the lower model to reach the goal
        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx+self.step_cnt)%self.mod, test_num=self.train_num,
                           use_real_map=True, realmap_row=326, realmap_col=364,
                           is_lower=True, dummy_mode=upper_mode)
        #print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = []
        self.is_match_compute_tuple = [0,0] #total, match

        while not lower_done:
            lower_step_cnt += 1
            q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
            sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
            for j in range(len(sorted_actions)):
                tmp_action = sorted_actions[j]
                if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                    lower_action = tmp_action
                    break
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            #print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2]+ lower_env.delta)

        t_lower = 0
        v_expected = mode_v_dict[upper_mode]/60
        v_rural = 0.5 # 36km/h

        for i, coord in enumerate(lower_path[1:]):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            if self.mapmatrice[upper_mode][x][y]==0:
                t_lower += (abs(lower_path[i][0]-lower_path[i-1][0]) + abs(lower_path[i][1]-lower_path[i-1][1]))/v_rural
                #print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                t_lower += (abs(lower_path[i][0]-lower_path[i-1][0])+ abs(lower_path[i][1]-lower_path[i-1][1]))/v_expected

        # t_upper calculated by the given data
        idx=(self.traj_idx+self.step_cnt)% self.mod
        if idx == 0:
            t_upper = 0
        else:
            t_upper = max(0,self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod +1, 'time'] \
                      - self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod, 'time'])

        if self.isTest:
            print('in upper iteration ',self.step_cnt,'t_lower:', t_lower, 't_upper:', t_upper, 'predict mode', modelist[action])

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        reward -= abs(t_lower - t_upper) #/(max(t_lower, t_upper) + 1) # +1 avoid div0
        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model
        self.step_cnt += 1
        self.r_avg += (reward-self.r_avg)/(self.step_cnt+1)
        #print('current reward:', reward, 'avg reward:', self.r_avg)

        # embedding the map info to the state
        #print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod - 1, ['locx', 'locy']])
        goal_pos =tuple( self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod , ['locx', 'locy']])
        self.rts_nums = [0,0,0,0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                if neighbor == 1:
                    self.rts_nums[mode_idx] = neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                if neighbor ==1:
                    self.rts_nums[mode_idx] = neighbor

            # TS can travel on TG
            if mode_idx == 2:
                for neighbor in get_neighbor(self.mapmatrice[modelist[3]], x1, y1):
                    if neighbor == 1:
                        self.rts_nums[mode_idx] = neighbor
                for neighbor in get_neighbor(self.mapmatrice[modelist[3]], x2, y2):
                    if neighbor == 1:
                        self.rts_nums[mode_idx] = neighbor

        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])/(t_upper+1)
        self.v_avg += (v_upper - self.v_avg)/(self.step_cnt+1)

        cos = 1
        if self.step_cnt>1:
            pre_pos = tuple(self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod - 2, ['locx', 'locy']])
            inner_product = ((goal_pos[0]-start_pos[0])*(goal_pos[0]-pre_pos[0]) + (goal_pos[1]-start_pos[1])*(goal_pos[1]-pre_pos[1]))
            length_product2 = ((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)*((goal_pos[0]-pre_pos[0])**2 + (goal_pos[1]-pre_pos[1])**2)
            cos = inner_product/(length_product2**0.5+1)

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        #relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = ((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)**0.5

        return np.array([relative_dis] + self.rts_nums), reward, done

    def step_with20action(self, action:int):
        """

        :param action:int , [0,20,30...400] denoted discreted v
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TG','TS']

        matchedmode = None
        min_v_bias = 400
        for mode in action_mode_duels:
            if min_v_bias > abs(action*17 - mode_v_dict[mode]):
                matchedmode = mode
            min_v_bias = min(abs(action*17 - mode_v_dict[mode]), min_v_bias)

        upper_mode = matchedmode
        reward = 0

        # t_lower is the time cost of the lower model to reach the goal
        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx+self.step_cnt)%self.mod, test_num=self.train_num,
                           use_real_map=True, realmap_row=326, realmap_col=364,
                           is_lower=True, dummy_mode=upper_mode)
        #print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = []
        self.is_match_compute_tuple = [0,0] #total, match

        while not lower_done:
            lower_step_cnt += 1
            q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
            sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
            for j in range(len(sorted_actions)):
                tmp_action = sorted_actions[j]
                if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                    lower_action = tmp_action
                    break
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            #print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2]+ lower_env.delta)

        t_lower = 0
        v_expected = mode_v_dict[upper_mode]/60
        v_rural = 0.3

        for i, coord in enumerate(lower_path[1:]):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            if self.mapmatrice[upper_mode][x][y]==0:
                t_lower += (abs(lower_path[i][0]-lower_path[i-1][0]) + abs(lower_path[i][1]-lower_path[i-1][1]))/v_rural
                #print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                t_lower += (abs(lower_path[i][0]-lower_path[i-1][0])+ abs(lower_path[i][1]-lower_path[i-1][1]))/v_expected

        # t_upper calculated by the given data
        idx=(self.traj_idx+self.step_cnt)% self.mod
        if idx == 0:
            t_upper = 0
        else:
            t_upper = max(0,self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod +1, 'time'] \
                      - self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod, 'time'])

        if self.isTest:
            print('in upper iteration ',self.step_cnt,'t_lower:', t_lower, 't_upper:', t_upper, 'predict mode', upper_mode)

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        #reward -= abs(t_lower - t_upper) *(1 - (self.is_match_compute_tuple[1]/(self.is_match_compute_tuple[0]+0.1)))

        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model
        self.step_cnt += 1
        self.r_avg += (reward-self.r_avg)/(self.step_cnt+1)
        #print('current reward:', reward, 'avg reward:', self.r_avg)

        # embedding the map info to the state
        #print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod - 1, ['locx', 'locy']])
        goal_pos =tuple( self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod , ['locx', 'locy']])
        self.rts_nums = [0,0,0,0]

       # state computing with neighbor rts
        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                if neighbor == 1:
                    self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                if neighbor ==1:
                    self.rts_nums[mode_idx] += neighbor

            # TS can travel on TG
            if mode_idx == 2:
                for neighbor in get_neighbor(self.mapmatrice[modelist[3]], x1, y1):
                    if neighbor == 1:
                        self.rts_nums[mode_idx] += neighbor
                for neighbor in get_neighbor(self.mapmatrice[modelist[3]], x2, y2):
                    if neighbor == 1:
                        self.rts_nums[mode_idx] += neighbor

        # using v_avg as state, v_avg_upper = distance/delta_t
        self.v_avg = 0
        v_upper = (abs(goal_pos[0] - start_pos[0])**2 + abs(goal_pos[1] - start_pos[1])**2)**0.5 /(t_upper+1)
        self.v_avg += (v_upper - self.v_avg)/(self.step_cnt+1)

        # calculate reward using v
        # reward -= (abs(v_upper-v_expected)) *(1-(self.is_match_compute_tuple[1]/(self.is_match_compute_tuple[0]+0.1)))

        # calculate reward using cl
        mean, std = mean_std_dict[upper_mode]
        z_score = (v_upper - mean)/std
        cl = 2*(1-norm.cdf(abs(z_score)))

        reward += cl* (self.is_match_compute_tuple[1]/(self.is_match_compute_tuple[0]+0.1))

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        #relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = ((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)**0.5
        self.upper_mode = upper_mode
        return np.array([relative_dis] + self.rts_nums), reward, done


    def reset(self):
        """
        state of upper model in time t is s_t = [r_avg, r_t-1, a_t-1] avg means the avg reward of the last m steps
        :return:self.state: np.array, the state of the environment
        """

        self.r_avg = 0
        self.step_cnt = 0
        self.max_step = 0

        self.traj_idx += 1
        i = self.traj_idx
        #print('reset upper',self.traj.loc[i,'ID'], self.traj.loc[i+1, 'ID'])
        while self.traj.loc[i%self.mod, 'ID'] == self.traj.loc[(i+1)% self.mod, 'ID']:
            self.max_step += 1
            i += 1
        #print('upper reset', 'maxstep',self.max_step,'trajstart', self.traj_idx)

        # embedding the map info to the state
        start_pos = tuple(self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod, ['locx', 'locy']])
        goal_pos =tuple( self.traj.loc[(self.traj_idx+self.step_cnt)%self.mod + 1, ['locx', 'locy']])
        self.rts_nums = [0,0,0,0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                self.rts_nums[mode_idx] += neighbor
        self.v_avg = 0

        relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = ((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)**0.5

        return np.array( [relative_dis] + self.rts_nums)

