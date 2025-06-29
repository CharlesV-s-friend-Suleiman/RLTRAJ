import torch
import numpy as np
import pandas as pd

from rl_utils.policy_based_rl_methods import Policy
from rl_utils.tools import mapdata_to_modelmatrix, get_neighbor
from rl_utils.value_based_rl_methods import VAnet
import pickle

# glabal variables
dxdy_dict = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
modelist = ['GSD', 'GG', 'TS', 'TG']
with open('data/vdistributiondatainrealworld.pkl', 'rb') as f:
    processed_data = pickle.load(f)


def _allow(neighbor: int, mode: str) -> bool:
    """
    check if the neighbor is allowed to travel of the given mode: static, TG, GG, GSD, TS
    :param neighbor: int, element of a list of 9 elements, the 4-th element is the grid itself, the other 8 elements are the neighbors
    :param mode: str, 'TG', 'GG', 'GSD', 'static'
    :return: bool, True if the neighbor is allowed to travel of the given mode, False otherwise
    """
    if mode == 'TG' or mode == 'static':
        return neighbor >> 1 & 1 == 1
    elif mode == 'GG':
        return neighbor >> 3 & 1 == 1
    elif mode == 'GSD':
        return neighbor >> 2 & 1 == 1 or neighbor >> 5 & 1 == 1
    elif mode == 'TS':
        return neighbor >> 6 & 1 == 1 or neighbor >> 1 & 1 == 1
    elif mode == 'static':
        return True


class UpperEnv:
    def __init__(self, mapdata: dict, traj: pd.DataFrame,
                 trainid_start=0, test_mode=False, testid_start=0,
                 test_num=8, train_num=10000,
                 m=5,
                 use_real_map=False, realmap_row=326, realmap_col=364, lower_model_config=None):
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
        self.step_cnt = 0
        self.traj = traj
        self.traj_idx = 0
        self.train_num = train_num
        self.trainid_start = trainid_start
        self.m = m
        self.max_step = 20  # max step of the upper model
        self.rewalmap_row = realmap_row
        self.realmap_col = realmap_col

        if use_real_map:
            print('using real map', realmap_row, realmap_col)
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
            self.lower_type = lower_model_config['model_type']  # str , 'DQN' or 'SAC' or other
            state_dim_lower = lower_model_config['state_dim']
            hidden_dim_lower = lower_model_config['hidden_dim']
            action_dim_lower = lower_model_config['action_dim']
            model_path = lower_model_config['model_path']
        except:
            raise ValueError('lower_model_config is not correct, check the key of the dict or model path')

        lower_agent = None
        if self.lower_type == 'DQN':
            lower_agent = VAnet(state_dim_lower, hidden_dim_lower, action_dim_lower)
        elif self.lower_type == 'SAC':
            lower_agent = Policy(state_dim_lower, hidden_dim_lower, action_dim_lower)
        lower_agent.load_state_dict(torch.load(model_path))
        lower_agent.eval()
        self.lower_agent = lower_agent

    def step(self, action: int):
        """
        :param action:int , [0, 1, 2, 3] denotes 4 modes: [GSD, GG, TS, TG]
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TS', 'TG']
        upper_mode = action_mode_duels[action]

        # t_lower is the time cost of the lower model to reach the goal
        # print(self.traj_idx , self.step_cnt)
        if self.traj_idx + self.step_cnt >= self.mod:
            self.traj_idx = 0

        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod - 1, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.rewalmap_row, realmap_col=self.realmap_col,
                           is_lower=True, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = [lower_state[:2] + lower_env.delta]
        self.is_match_compute_tuple = [0, 0]  # total, match

        while not lower_done:
            lower_step_cnt += 1
            if self.lower_type == 'DQN':
                q_values = self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0))
                sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
                for j in range(len(sorted_actions)):
                    tmp_action = sorted_actions[j]
                    if tuple(lower_state[:2] + dxdy_dict[tmp_action]) not in lower_set:
                        lower_action = tmp_action
                        break
            elif self.lower_type == 'SAC':
                lower_action = int(
                    self.lower_agent(torch.tensor(lower_state, dtype=torch.float32).unsqueeze(0)).argmax())
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)
        # print(lower_path)
        t_lower = 0  # min
        v_expected = processed_data[upper_mode]['mean'] / 60  # convert to km/min
        v_rural = 0.5  # 0.5km/min=30km/h

        for i, coord in enumerate(lower_path[1:]):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            if self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0]) ** 2 + (
                        lower_path[i][1] - lower_path[i - 1][1]) ** 2) ** 0.5 / v_rural
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                # t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                #     lower_path[i][1] - lower_path[i - 1][1])) / v_expected
                t_lower += ((lower_path[i][0] - lower_path[i - 1][0])**2 + (
                    lower_path[i][1] - lower_path[i - 1][1])**2 )**0.5 / v_expected

        # t_upper calculated by the given data
        t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time'])

        # if self.isTest:
        #     print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
        #           modelist[action])

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        reward = -float(abs(t_lower - t_upper)) / max(t_lower,
                                                      t_upper)  # TODO# /(max(t_lower, t_upper) + 1) # +1 avoid div0
        match_rate = self.is_match_compute_tuple[1] / (self.is_match_compute_tuple[0] + 0.1) if (
                lower_step_cnt <= lower_env.max_step) else 0
        reward *= (1 - match_rate)
        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model

        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # todo: meng,zhang: 这里要改成适配新数据集的，把旧版本分开的locx locy改成 locx_o locy_o，locy_d, locy_d
        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])
        start_pos = tuple(
            self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])  # TODO 这里要改
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        self.upper_mode = upper_mode
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                if neighbor == 1:
                    self.rts_nums[mode_idx] = neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                if neighbor == 1:
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
        v_upper = float(abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])) / (
                    t_upper + 0.000001)  # km/min
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        cos = 1
        if self.step_cnt > 1:
            pre_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 2, ['locx_o',
                                                                                           'locy_o']])  # TODO 凡是涉及到locx locy都需要修改
            inner_product = (
                    (goal_pos[0] - start_pos[0]) * (goal_pos[0] - pre_pos[0]) + (goal_pos[1] - start_pos[1]) * (
                    goal_pos[1] - pre_pos[1]))
            length_product2 = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) * (
                    (goal_pos[0] - pre_pos[0]) ** 2 + (goal_pos[1] - pre_pos[1]) ** 2)
            cos = inner_product / (length_product2 ** 0.5 + 1)

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False

        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) ** 0.5
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.step_cnt += 1
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums), reward, done

    def step_with20action(self, action: int):
        """

        :param action:int , [0,20,30...400] denoted discreted v
        :return: next_state, reward, done
        """
        action_mode_duels = ['GSD', 'GG', 'TG', 'TS']
        print(processed_data)
        matchedmode = None
        min_v_bias = 400
        for mode in action_mode_duels:
            if min_v_bias > abs(action * 17 - processed_data[mode]['mean']):
                matchedmode = mode
            min_v_bias = min(abs(action * 17 - processed_data[mode]['mean']), min_v_bias)

        upper_mode = matchedmode
        reward = 0

        # t_lower is the time cost of the lower model to reach the goal
        lower_env = MapEnv(self.mapdata, self.traj, test_mode=True,
                           testid_start=(self.traj_idx + self.step_cnt) % self.mod, test_num=self.train_num,
                           use_real_map=True, realmap_row=self.rewalmap_row, realmap_col=self.realmap_col,
                           is_lower=False, dummy_mode=upper_mode)
        # print('upper' , 'step',self.step_cnt, 'traj+start id ',self.traj_idx)
        lower_state = lower_env.reset()
        lower_done = False
        lower_set = set()
        lower_set.add(tuple(lower_state[:2]))
        lower_step_cnt = 0
        lower_path = []
        self.is_match_compute_tuple = [0, 0]  # total, match

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
            lower_next_state, lower_reward, lower_done = lower_env.step(lower_action)
            # print("    lower_next_state:", lower_next_state[:4], "lower_reward:", lower_reward, "lower_done:", lower_done)
            lower_state = lower_next_state
            lower_set.add(tuple(lower_state[:2]))
            lower_path.append(lower_state[:2] + lower_env.delta)

        t_lower = 0
        v_expected = processed_data[upper_mode]['mean'] / 60
        v_rural = 0.3
        v_lower = 0

        for i, coord in enumerate(lower_path[1:]):
            x, y = int(coord[0]), int(coord[1])
            self.is_match_compute_tuple[0] += 1
            v_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(lower_path[i][1] - lower_path[i - 1][1]))
            # restrict x,y to avoid OutOfRange err
            if (0 <= x < self.realmap_col and 0 <= y < self.rewalmap_row) and self.mapmatrice[upper_mode][x][y] == 0:
                t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                    lower_path[i][1] - lower_path[i - 1][1])) / v_rural
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                self.is_match_compute_tuple[1] += 1
                t_lower += (abs(lower_path[i][0] - lower_path[i - 1][0]) + abs(
                    lower_path[i][1] - lower_path[i - 1][1])) / v_expected

        # t_upper calculated by the given data
        idx = (self.traj_idx + self.step_cnt) % self.mod
        if idx == 0:
            t_upper = 0
        else:
            t_upper = max(0, self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod + 1, 'time'] \
                          - self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod, 'time'])

        if self.isTest:
            print('in upper iteration ', self.step_cnt, 't_lower:', t_lower, 't_upper:', t_upper, 'predict mode',
                  upper_mode)

        # calculate the difference t_lower and t_upper in each step, record the percentage of traj in mode
        # reward -= abs(t_lower - t_upper) *(1 - (self.is_match_compute_tuple[1]/(self.is_match_compute_tuple[0]+0.1)))

        self.t_lower = t_lower
        self.t_upper = t_upper

        # update state of the upper model
        self.step_cnt += 1
        self.r_avg += (reward - self.r_avg) / (self.step_cnt + 1)
        # print('current reward:', reward, 'avg reward:', self.r_avg)

        # embedding the map info to the state
        # print('upper step',self.step_cnt, 'traj+start id-1 ',self.traj_idx,'curidx',self.traj_idx+self.step_cnt-1,
        #      'maxstep',self.max_step, 'coord', self.traj.loc[self.traj_idx+self.step_cnt-1,'locx'], self.traj.loc[self.traj_idx+self.step_cnt-1,'locy'])

        start_idx = (self.traj_idx + self.step_cnt) % self.mod - 1
        # TODO: check the start_idx WARNING
        if start_idx < 0:
            start_idx += 1
        start_pos = tuple(self.traj.loc[start_idx, ['locx_o', 'locy_o']])
        goal_pos = tuple(self.traj.loc[(start_idx + 1) % self.mod, ['locx_o', 'locy_o']])
        self.rts_nums = [0, 0, 0, 0]

        # state computing with neighbor rts
        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                if neighbor == 1:
                    self.rts_nums[mode_idx] += neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                if neighbor == 1:
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
        v_upper = (abs(goal_pos[0] - start_pos[0]) ** 2 + abs(goal_pos[1] - start_pos[1]) ** 2) ** 0.5 / (
                t_upper + 1) * 60
        self.v_avg += (v_upper - self.v_avg) / (self.step_cnt + 1)

        # calculate reward using v
        v_lower = v_lower / (t_lower + 1) * 60
        # reward -= (abs(v_lower-v_expected)) *(1-(self.is_match_compute_tuple[1]/(self.is_match_compute_tuple[0]+0.1)))

        mean, std = processed_data[upper_mode]['mean'], processed_data[upper_mode]['std']
        z_score = (v_upper - mean) / std
        # factor_dict = {'GSD':1.5, 'GG':1, 'TS':0.75, 'TG':0.75}
        reward -= abs(z_score) * (1 - self.is_match_compute_tuple[1] / (
                self.is_match_compute_tuple[0] + 0.1))  # *factor_dict[upper_mode]

        if self.step_cnt == self.max_step:
            self.traj_idx += self.step_cnt
            done = True
        else:
            done = False
        v_differ = [v_upper - processed_data[i]['mean'] for i in modelist]
        # relative_pos = [goal_pos[0]-start_pos[0], goal_pos[1]-start_pos[1]]
        relative_dis = (abs((goal_pos[0] - start_pos[0]) ** 2 + abs(goal_pos[1] - start_pos[1])) ** 2) ** 0.5
        self.upper_mode = upper_mode
        return np.array([v_upper] + v_differ + self.rts_nums), reward, done

    def reset(self):
        """
        state of upper model in time t is s_t = [r_avg, r_t-1, a_t-1] avg means the avg reward of the last m steps
        :return:self.state: np.array, the state of the environment
        """

        self.r_avg = 0
        self.step_cnt = 0
        self.max_step = 1

        self.traj_idx += 1
        i = self.traj_idx
        # print('reset upper',self.traj.loc[i,'ID'], self.traj.loc[i+1, 'ID'])
        while self.traj.loc[i % self.mod, 'ID'] == self.traj.loc[(i + 1) % self.mod, 'ID']:
            self.max_step += 1
            i += 1
        # print('upper reset', 'maxstep',self.max_step,'trajstart', self.traj_idx)

        # embedding the map info to the state
        start_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_o', 'locy_o']])
        goal_pos = tuple(self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, ['locx_d', 'locy_d']])
        t = self.traj.loc[(self.traj_idx + self.step_cnt) % self.mod - 1, 'time']
        self.rts_nums = [0, 0, 0, 0]

        for mode_idx in range(4):
            x1, y1 = start_pos
            x2, y2 = goal_pos
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x1, y1):
                if neighbor == 1:
                    self.rts_nums[mode_idx] = neighbor
            for neighbor in get_neighbor(self.mapmatrice[modelist[mode_idx]], x2, y2):
                if neighbor == 1:
                    self.rts_nums[mode_idx] = neighbor

        self.v_avg = 0

        relative_pos = [goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]]
        # relative_dis = ((goal_pos[0] - start_pos[0]) ** 2 + (goal_pos[1] - start_pos[1]) ** 2) ** 0.5
        relative_dis = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])
        return np.array([relative_dis / (t + 0.001)] + self.rts_nums)


class MapEnv:
    """
    """

    def __init__(self, mapdata: dict, traj: pd.DataFrame,
                 test_mode=False, testid_start=0, test_num=8,
                 use_real_map=False, realmap_row=326, realmap_col=364,
                 is_lower=False, dummy_mode=None):
        '''
        这里的traj用的是重新排版后的，每一条数据包含了起点和终点，这样就不需要写很多特判逻辑
        '''

        self.traj = traj
        self.step_cnt = 0
        self.map_row = realmap_row
        self.map_col = realmap_col

        if use_real_map:
            self.mapdata = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)
        self.is_lower = is_lower  # when is_lower is True, the env is used for lower model, and mode info is not used
        self.dummy_mode = dummy_mode

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num
        self.distance_hold = 0 if test_mode else 0
        self.traj_cnt = 0  # traj_CNT 是当前训练轨迹的严格索引

    def reset(self):
        # reset env by using next two traj record
        # for example, 1st interation, start = traj[0], goal = traj[1]; 2nd interation, start = traj[1], goal = traj[2]...
        self.step_cnt = 0

        if self.isTest:
            mod = self.test_num
            start_id = self.testid_start
        else:
            mod = len(self.traj)
            start_id = 0
        # print('reset map env', 'traj_cnt', self.traj_cnt, 'start_id', start_id, 'mod', mod)
        locx_start = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locx_o'])
        locy_start = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locy_o'])
        locx_end = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locx_d'])
        locy_end = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locy_d'])

        # when test lower_model, using serval traj records
        self.traj_cnt += 1

        self.mode = self.traj.loc[start_id + self.traj_cnt % mod, 'mode'] if not self.is_lower else self.dummy_mode
        # delta is the relative position of the start_position and 0,0; delta only change when start_position change(when reset)
        # neighbor is the 8 elements list of the grid not including itself, 0-8 are the neighbors from 1,0 to 1,-1
        self.neighbor = np.array(get_neighbor(self.mapdata[self.mode], locx_start, locy_start, size=3))
        self.delta = np.array([locx_start, locy_start])
        self.state = np.array([0, 0])
        self.goal = np.array([locx_end - locx_start, locy_end - locy_start])
        # max step is the mahattan distance between start and goal add 10
        self.max_step = np.abs(locx_start - locx_end) + np.abs(locy_start - locy_end) + 10
        return np.hstack((self.state, self.goal, self.neighbor))

    def step(self, action: int):  # todo 这里要改
        # agent will move to 8 directions,action is tuple of (dx,dy)
        reward = 0
        self.step_cnt += 1
        d = dxdy_dict[action]

        # update state of position
        self.state += np.array(d)
        # Python
        dist = float(np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]))
        denominator = np.abs(self.goal[0]) + np.abs(self.goal[1])

        if denominator != 0:
            dist /= denominator
        else:
            dist = float('inf')  # Assign a large value or handle appropriately
        # not in the available neighbor
        if self.neighbor[action] != 0:  # when size = 3
            # if self.neighbor[(2+dx)*3 + (2+dy)]!=0: # when size=5
            #     dx,dy = d[0], d[1]
            reward += 0.1  # reward += 0.3

        # update neighbor
        self.neighbor = np.array(
            get_neighbor(self.mapdata[self.mode], self.state[0] + self.delta[0], self.state[1] + self.delta[1], size=3))

        # to encourage the agent travel in the shortest path
        reward -= dist if dist > self.distance_hold else 0  # 惩罚到终点的距离
        # reward -= 1 if np.abs(self.state[0] - self.goal[0]) + np.abs(self.state[1] - self.goal[1]) > self.distance_hold else 0
        if np.abs(self.state[0] - self.goal[0]) + np.abs(
                self.state[1] - self.goal[1]) == self.distance_hold or self.step_cnt == self.max_step:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map
        return np.hstack((self.state, self.goal, self.neighbor)), reward, done


class ODMapEnv:
    """
    训练两个网络两个智能体，一个从起点出发，一个从终点出发，till met or max step
    """

    def __init__(self, mapdata: dict, traj: pd.DataFrame,
                 test_mode=False, testid_start=0, test_num=8,
                 use_real_map=False, realmap_row=326, realmap_col=364,
                 is_lower=False, dummy_mode=None):
        '''
        这里的traj用的是重新排版后的，每一条数据包含了起点和终点，这样就不需要写很多特判逻辑
        '''

        self.traj = traj
        self.step_cnt = 0
        self.map_row = realmap_row
        self.map_col = realmap_col

        if use_real_map:
            self.mapdata = mapdata_to_modelmatrix(mapdata, realmap_row, realmap_col)
        self.is_lower = is_lower  # when is_lower is True, the env is used for lower model, and mode info is not used
        self.dummy_mode = dummy_mode

        # test mode
        self.isTest = test_mode
        self.testid_start = testid_start
        self.test_num = test_num
        self.distance_hold = 0 if test_mode else 1
        self.traj_cnt = 0  # traj_CNT 是当前训练轨迹的严格索引

    def reset(self):
        """
        # reset env by using next two traj record
        # for example, 1st interation, start = traj[0], goal = traj[1]; 2nd interation, start = traj[1], goal = traj[2]...
        # 考虑到两个智能体，所以返回state是一个dict，d[start]和d[end]分别为原先结构

        :return:  包含从起点和终点出发的智能体的state的字典
        """

        self.step_cnt = 0
        if self.isTest:
            mod = self.test_num
            start_id = self.testid_start
        else:
            mod = len(self.traj)
            start_id = 0

        # t==t 时刻绝对坐标，只能在reset时更新为对应csv里的记录 || 在step时被action更新
        self.locx_start = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locx_o'])
        self.locy_start = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locy_o'])
        self.locx_end = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locx_d'])
        self.locy_end = float(self.traj.loc[start_id + self.traj_cnt % mod, 'locy_d'])

        # when test lower_model, using serval traj records
        self.traj_cnt += 1

        self.mode = self.traj.loc[self.traj_cnt % mod, 'mode'] if not self.is_lower else self.dummy_mode

        # delta is the relative position of the start_position and 0,0; delta only change when start_position change(when reset)
        # neighbor is the 8 elements list of the grid not including itself, 0-8 are the neighbors from 1,0 to 1,-1
        self.neighbor_start = np.array(get_neighbor(self.mapdata[self.mode], self.locx_start, self.locy_start, size=3))
        self.neighbor_end = np.array(get_neighbor(self.mapdata[self.mode], self.locx_end, self.locy_end, size=3))

        # 计算t==0时刻相对坐标
        ego_pos = np.array([0, 0])
        goal_pos_start = np.array([self.locx_end - self.locx_start, self.locy_end - self.locy_start])
        goal_pos_end = np.array([self.locx_start - self.locx_end, self.locy_start - self.locy_end])

        # max step is the mahattan distance between start and goal add 10
        self.max_step = np.abs(self.locx_start - self.locx_end) + np.abs(self.locy_start - self.locy_end) + 10

        self.state_dual = {
            'start': np.hstack((ego_pos, goal_pos_start, self.neighbor_start)),
            'end': np.hstack((ego_pos, goal_pos_end, self.neighbor_end))
        }

        return self.state_dual

    def step_2agent(self, action_start: int, action_end: int):  #
        """
        接受两个智能体动作后更新环境，如果要多agents，传参actionlist就行

        :param action_start: 从起点出发的agent的动作
        :param action_end: 从终点出发的agent的动作
        :return: state_dual, reward_dual, done # state reward 是各自的，done是共有的
        """
        # agent will move to 8 directions,action is tuple of (dx,dy)
        self.step_cnt += 1

        r_start, r_end = 0, 0
        d_start, d_end = dxdy_dict[action_start], dxdy_dict[action_end]

        # 更新 t==t 时刻绝对坐标
        self.locx_start += d_start[0]
        self.locy_start += d_start[1]
        self.locx_end += d_start[0]
        self.locy_end += d_start[1]

        # 更新state
        for k in self.state_dual.keys():
            if k == 'start':
                self.state_dual[k][:2] += np.array([d_start[0], d_start[1]])
                self.state_dual[k][2:4] += np.array([d_end[0], d_end[1]])
                self.state_dual[k][4:] = np.array(get_neighbor(self.mapdata[self.mode],
                                                               self.locx_start,
                                                               self.locy_start, size=3))
            elif k == 'end':
                self.state_dual[k][:2] += np.array([d_end[0], d_end[1]])
                self.state_dual[k][2:4] += np.array([d_start[0], d_start[1]])
                self.state_dual[k][4:] = np.array(get_neighbor(self.mapdata[self.mode],
                                                               self.locx_end,
                                                               self.locy_end, size=3))

        # not in the available neighbor
        if self.neighbor_start[action_start] != 0:  # when size = 3
            r_start += 0.0
        if self.neighbor_end[action_end] != 0:
            r_end += 0.0

        # update neighbor
        self.neighbor_start = self.state_dual['start'][4:]
        self.neighbor_end = self.state_dual['end'][4:]

        # to encourage the agent travel in the shortest path
        if np.abs(self.locx_start - self.locx_end) + np.abs(self.locy_start - self.locy_end) > self.distance_hold:
            r_start -= 1
            r_end -= 1

        if np.abs(self.locx_start - self.locx_end) + np.abs(
                self.locy_start - self.locy_end) <= self.distance_hold or self.step_cnt == self.max_step:
            done = True
        else:
            done = False

        # to avoid the repeated state and encourage the agent explore by real-map
        return self.state_dual, r_start, r_end, done
