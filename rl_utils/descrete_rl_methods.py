import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import collections

# DQN method, Q-net for DQN and VA-net for duelDQN, DQN method is unstable and the return will wander
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
class Policy(torch.nn.Module): #TODO: add the action information of real-map road network
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # use softmax to get the probability of each action
        return F.softmax(self.fc2(x), dim=1)


class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr,
                 alpha_lr, target_entropy,gamma, tau, device):

        # policy net
        self.actor = Policy(state_dim, hidden_dim, action_dim).to(device)
        # actor and critic net using VAnet
        self.critic1 = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = VAnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = VAnet(state_dim, hidden_dim, action_dim).to(device)

        # set the optimizer & initialize the target net
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # set the hyperparameters & device
        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action_prob = self.actor(state)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()

        return action.item()

    def calculate_target(self, rewards, next_states, done):
        next_probs = self.actor(next_states)
        next_logprobs = self.target_critic1(next_states+ 1e-8) # add a small value to avoid NAN
        ent = -torch.sum(next_probs * next_logprobs, dim=1).unsqueeze(1)
        q1_value = self.target_critic1(next_states)
        q2_value = self.target_critic2(next_states)
        min_qvalue = torch.sum(next_probs*(torch.min(q1_value, q2_value))
                               ,dim=1).unsqueeze(1)
        next_value = min_qvalue + self.log_alpha.exp() * ent
        td_target = rewards + self.gamma * (1 - done) * next_value

        return td_target

    def soft_update(self,net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

        # update the critic net
        td_target = self.calculate_target(rewards, next_states, dones)
        critic_q1_values = self.critic1(states).gather(1, actions)
        critic_q2_values = self.critic2(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_q1_values, td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(critic_q2_values, td_target.detach()))

        # optimize the critic net
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # update the policy net
        probs = self.actor(states)
        logprobs = self.critic1(states + 1e-8)
        ent = -torch.sum(probs * logprobs, dim=1).unsqueeze(1)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        min_qvalue = torch.sum(probs*(torch.min(q1_value, q2_value)),dim=1).unsqueeze(1)
        actor_loss = torch.mean( - self.log_alpha.exp() * ent - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the alpha
        alpha_loss = torch.mean( (ent - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # soft update the target net
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

