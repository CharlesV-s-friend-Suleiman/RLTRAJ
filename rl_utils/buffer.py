import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections



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

    def store_step(self, state, action, reward, down):
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(down)
        self.states.append(state)
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

    def sample(self, batch_size, use_her, dis_threshold=0, her_ratio=0.8):
        batch = dict(state=[],
                     action=[],
                     next_state=[],
                     reward=[],
                     done=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer,1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.states[step_state+1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.done[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step_state+1, traj.length+1)
                goal = traj.states[step_goal][:2]
                dis = np.abs(goal[0] - state[0]) + np.abs(goal[1] - state[1])
                reward = -1  if dis > dis_threshold else 0
                done = False if dis > dis_threshold else True
                state = np.hstack((state[:2], goal))
                next_state = np.hstack((next_state[:2], goal))

            batch['state'].append(state)
            batch['next_state'].append(next_state)
            batch['action'].append(action)
            batch['reward'].append(reward)
            batch['done'].append(done)

        batch['state'] = np.array(batch['state'])
        batch['next_state'] = np.array(batch['next_state'])
        batch['action'] = np.array(batch['action'])

        return batch

