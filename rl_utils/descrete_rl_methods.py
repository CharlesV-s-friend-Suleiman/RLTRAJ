import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections

# DQN method
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

# SAC method to balance the exploration and exploitation

