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

        self.traj_cnt = testid_start if self.isTest else 0 # TRAJCNT is the index of the traj record + 2

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
        locx_start = float(self.traj.loc[self.traj_cnt%mod, 'locx'])
        locy_start = float(self.traj.loc[self.traj_cnt%mod, 'locy'])
        locx_end = float(self.traj.loc[self.traj_cnt%mod + 1 , 'locx'])
        locy_end = float(self.traj.loc[self.traj_cnt%mod + 1, 'locy'])

        # when test model, using serval traj records
        self.traj_cnt += 1

        self.mode = self.traj.loc[self.traj_cnt%8, 'mode']
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


class DQN:
    # DQN foe discrete action space
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 lr,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type):
        self.dqn_type = dqn_type
        self.action_dim = action_dim

        if dqn_type == "dueling":
            # set the Q-net
            self.qnet = VAnet(state_dim, hidden_dim, action_dim).to(device)
            # set the target Q-net
            self.target_qnet = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else:
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

    def take_action(self,state, episode): # use epsilon-greedy to take action
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

        # double DQN next 2 lines
        max_action = self.qnet(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_qnet(next_states).gather(1, max_action)

        # DQN
        #max_next_q_values = self.target_qnet(next_states).max(1)[0].view(
        #    -1, 1)
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

class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fcA = torch.nn.Linear(hidden_dim, action_dim)
        self.fcV = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fcA(F.relu(self.fc1(x)))
        V = self.fcV(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q

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

