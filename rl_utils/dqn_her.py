import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections

"""
using DQN and HER to recover traj between 2 traj record: start and goal"""
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
    def __init__(self, mapdata:dict, traj:pd.DataFrame):
        self.mapdata = mapdata
        self.traj = traj
        self.traj_cnt = 0
        self.hashmap = set()

    def reset(self):
        # reset env by using next two traj record
        # for example, 1st interation, start = traj[0], goal = traj[1]; 2nd interation, start = traj[1], goal = traj[2]...
        self.traj_cnt += 1 \
            if self.traj.loc[self.traj_cnt, 'mode'] != self.traj.loc[self.traj_cnt+1, 'mode'] else 0 # if the mode is different, then reset the env
        locx_start = float(self.traj.loc[self.traj_cnt, 'locx'])
        locy_start = float(self.traj.loc[self.traj_cnt, 'locy'])
        locx_end = float(self.traj.loc[self.traj_cnt + 1, 'locx'])
        locy_end = float(self.traj.loc[self.traj_cnt + 1, 'locy'])

        self.mode = self.traj.loc[self.traj_cnt, 'mode']
        self.traj_cnt += 1
        self.state = np.array([locx_start, locy_start])
        self.goal = np.array([locx_end,locy_end])
        self.hashmap = set()
        self.dis = np.sqrt(np.sum(np.square(self.state - self.goal)))
        self.expected_length = np.sqrt(np.sum(np.square(self.state - self.goal)))
        self.current_length = 0
        return np.hstack((self.state, self.goal))

    def step(self, action:int):
        # agent will move to 8 directions,action is tuple of (dx,dy)
        reward = 0
        dxdy_dict = {0:(-1,1), 1:(0,1), 2:(1,1), 3:(-1,0), 4:(1,0), 5:(-1,-1), 6:(0,-1), 7:(1,-1)}
        d = dxdy_dict[action]
        done = False

        self.current_length += 1
        self.state[0] += d[0]
        self.state[1] += d[1]

        # to encourage the agent travel in the shortest path
        dis = np.sqrt(np.sum(np.square(self.state - self.goal)))
        if dis < self.dis:
            reward += 2
            self.dis = dis

        # to avoid the repeated state and encourage the agent explore by real-map

        if dis <= 2:
            reward += 10
            done = True
        if self.current_length >= self.expected_length:
            done = True

        return np.hstack((self.state, self.goal)), reward, done


class DQN:
    # DQN foe discrete action space
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        # set the Q-net
        self.qnet = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # set the target Q-net
        self.target_qnet = Qnet(state_dim, hidden_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

        self.gamma = gamma # discount factor
        self.epsilon = epsilon # epsilon-greedy
        self.target_update = target_update
        self.device = device
        self.cnt = 0

    def take_action(self,state): # use epsilon-greedy to take action
        if np.random.random()< self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.qnet(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['state'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.qnet(states).gather(1, actions)

        max_next_q_values = self.target_qnet(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.cnt % self.target_update == 0:
            self.target_qnet.load_state_dict(
                self.qnet.state_dict())  #
        self.cnt += 1

class Qnet(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim ,action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # relu activation function
        return self.fc2(x)

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

    def sample(self, batch_size, use_her, dis_threshold=0.15, her_ratio=0.8):
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
                dis = np.sqrt(np.sum(np.square(next_state[:2] - goal)))
                reward = -1.0 if dis > dis_threshold else 0
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

