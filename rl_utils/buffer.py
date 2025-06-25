import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections

dxdy_dict = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}

class TrainTraj:
    """
    in this code, traj means the list of [locx, locy,t]; while traintraj means list of [s,a,r,s',g] in RL
    this class is to save traintraj
    """
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.done = []
        self.length = 0

    def store_step(self, state, action, reward, max_step ,done):
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)
        self.env_max_step = max_step
        self.states.append(state)
        self.length += 1

class TrainTrajwithMapinfo(TrainTraj):
    def __init__(self, init_state, init_position):
        super(TrainTrajwithMapinfo, self).__init__(init_state)
        self.cur_positions = [init_position]

    def store_step_withmapinfo(self, state, action, reward, max_step,cur_position, done):
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)
        self.env_max_step = max_step
        self.states.append(state)
        self.cur_positions.append(cur_position)
        self.length += 1

class Buffer:
    """
    replay buffer for DDPG with HER
    """
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add_traj(self, traj):
        self.buffer.append(traj)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, dis_threshold=1, her_ratio=0.8):
        batch = dict(state=[],
                     action=[],
                     next_state=[],
                     reward=[],
                     done=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer,1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state] # state: dim = 12, [curx, cury, goalx, goaly, neighbor x 8]
            next_state = traj.states[step_state+1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.done[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step_state+1, traj.length+1)
                goal = traj.states[step_goal][:2]
                dis = np.abs(goal[0] - state[0]) + np.abs(goal[1] - state[1])
                reward = 0
                reward += 0.3 if state[action+4]!=0 else 0
                reward -= 1 if dis > dis_threshold else 0
                done = False if dis > dis_threshold else True
                state = np.hstack((state[:2], goal, state[4:]))
                next_state = np.hstack((next_state[:2], goal, next_state[4:]))

            batch['state'].append(state)
            batch['next_state'].append(next_state)
            batch['action'].append(action)
            batch['reward'].append(reward)
            batch['done'].append(done)

        batch['state'] = np.array(batch['state'])
        batch['next_state'] = np.array(batch['next_state'])
        batch['action'] = np.array(batch['action'])

        return batch

    def sample_with_mapinfo(self, batch_size, use_her, dis_threshold=0, her_ratio=0.8):
        batch = dict(state=[],
                     action=[],
                     next_state=[],
                     reward=[],
                     current_position = [],
                     next_position = [],
                     done=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer,1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state] # state: dim = 12, [curx, cury, goalx, goaly, neighbor x 8]
            next_state = traj.states[step_state+1]
            next_position = traj.cur_positions[step_state + 1]

            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            current_position = traj.cur_positions[step_state]
            done = traj.done[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step_state+1, traj.length+1)
                goal = traj.states[step_goal][:2]
                dis = np.abs(goal[0] - state[0]) + np.abs(goal[1] - state[1])
                reward = 0
                dx,dy=  dxdy_dict[action]
                d = (2+dx)*5 + (2+dy)
                reward += 0.3  if state[d+4]!=0 else 0
                reward -= 1 if dis > dis_threshold else 0
                done = False if dis > dis_threshold else True
                state = np.hstack((state[:2], goal, state[4:]))
                next_state = np.hstack((next_state[:2], goal, next_state[4:]))

            batch['state'].append(state)
            batch['next_state'].append(next_state)
            batch['action'].append(action)
            batch['reward'].append(reward)
            batch['current_position'].append(current_position)
            batch['next_position'].append(next_position)
            batch['done'].append(done)

        batch['state'] = np.array(batch['state'])
        batch['next_state'] = np.array(batch['next_state'])
        batch['action'] = np.array(batch['action'])
        batch['current_position'] = np.array(batch['current_position'])
        batch['next_position'] = np.array(batch['next_position'])

        return batch
