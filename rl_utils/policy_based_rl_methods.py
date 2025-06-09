"""
Policy based algprothms, for compared methods and for multi-agent methods
PG, AC, SAC

2 reference; 1 is using PG and multi=stage training; 1 is using a multi-head attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from rl_utils.tools import mapdata_to_modelmatrix
from rl_utils.tools import ConvNet, sense_map

map_row = 529
map_col = 564
dxdy_dict = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}

class Qnet(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim ,action_dim):
        super(Qnet, self).__init__()
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) # 4 modes
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim//4)
        self.fc3 = torch.nn.Linear(hidden_dim//4, action_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))  # relu activation function
        # return self.fc2(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x)

class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc0 = torch.nn.Linear(state_dim, hidden_dim) # 4 modes
        self.fcV = torch.nn.Linear(hidden_dim, 1)
        self.fcA = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        A = F.elu(self.fc0(x))
        A = self.fcA(A)

        V = F.elu(self.fc0(x))
        V = self.fcV(V)

        Q = V + A - A.mean(1).view(-1, 1)

        return Q

class simpled_VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(simpled_VAnet, self).__init__()
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc0 = torch.nn.Linear(state_dim, hidden_dim) # 4 modes
        self.fcA = torch.nn.Linear(hidden_dim, action_dim)
        self.fcV = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc0(x)
        A = self.fcA(F.relu(x))
        V = self.fcV(F.relu(x))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q

# SAC method to balance the exploration and exploitation
class Policy(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim//4)
        self.fc3 = torch.nn.Linear(hidden_dim//4, action_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        # use softmax to get the probability of each action
        return F.softmax(self.fc3(x), dim=1)


class PolicyWithConv(nn.Module):
    def __init__(self,state_dim, hidden_dim, action_dim, conv_dim):
        super(PolicyWithConv, self).__init__()
        self.conv_net = ConvNet(input_channel=1, output_dim=conv_dim)
        self.fc1 = nn.Linear(state_dim+conv_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, map_info):
        """
        :param state: start and end position
        :param map_info: 5x5 matrix neighbor info
        :return:
        """

        conv_output = self.conv_net(map_info.unsqueeze(1))
        combined_input = torch.cat([state, conv_output], dim=1)
        x = F.relu(self.fc1(combined_input))
        x = F.softmax(self.fc2(x), dim=1)

        return x


class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr,
                 alpha_lr, target_entropy,gamma, tau, device,with_conv = False,
                 using_realmap=False, mapdata=None):

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
        self.visited_states = set()  # Set to store visited states


    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action_prob = self.actor(state)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()

        # Avoid revisiting states
        # for _ in range(action_prob.shape[1]):
        #     if tuple(state[:2]) in self.visited_states:
        #         action_prob[0, action] = -float('inf')

        self.visited_states.add(tuple(state[:2]))
        return action.item()

    def calculate_target(self, rewards, next_states, done):
        next_probs = self.actor(next_states)
        next_logprobs = torch.log(next_probs + 1e-8)# add a small value to avoid NAN
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
            F.smooth_l1_loss(critic_q1_values, td_target.detach()))
        critic_2_loss = torch.mean(
            F.smooth_l1_loss(critic_q2_values, td_target.detach()))

        # optimize the critic net
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # update the policy net

        probs = self.actor(states)
        logprobs = torch.log(probs + 1e-8)
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

        return float(critic_1_loss),float(actor_loss)


class SACWithConv(SAC):
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr,
                 alpha_lr, target_entropy,gamma, tau, device,
                 using_realmap=False, mapdata=None,mode='GSD'):
        super(SACWithConv, self).__init__(state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr,
                 alpha_lr, target_entropy,gamma, tau, device,
                 using_realmap=False, mapdata = None)
        self.mode = mode
        self.mapdata = mapdata
        self.mode_mapdata = self.mapdata[self.mode]
        self.actor = PolicyWithConv(state_dim, hidden_dim, action_dim, 5*5).to(device)

    def set_mode(self, mode):
        self.mode = mode
        self.mode_mapdata = self.mapdata[self.mode]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        current_positions = torch.tensor(transition_dict['current_position'], dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)

        # update the critic net
        td_target = self.calculate_target_with_conv(rewards, next_states, transition_dict['next_position'],dones)
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
        sensation_matrix = sense_map(self.mode_mapdata,current_positions,grid=5)
        probs = self.actor(states, sensation_matrix)
        logprobs = torch.log(probs + 1e-8)
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

        return float(critic_1_loss),float(actor_loss)


    def calculate_target_with_conv(self, rewards, next_states, next_positions,done):
        sensation_matrix = sense_map(self.mode_mapdata,next_positions, grid=5)
        next_probs = self.actor(next_states, sensation_matrix)
        next_logprobs = torch.log(next_probs + 1e-8)
        ent = -torch.sum(next_probs * next_logprobs, dim=1).unsqueeze(1)
        q1_value = self.target_critic1(next_states)
        q2_value = self.target_critic2(next_states)
        min_qvalue = torch.sum(next_probs*(torch.min(q1_value, q2_value))
                               ,dim=1).unsqueeze(1)
        next_value = min_qvalue + self.log_alpha.exp() * ent
        td_target = rewards + self.gamma * (1 - done) * next_value

        return td_target


    def take_action_with_conv(self, state, position):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        position = torch.tensor([position], dtype=torch.float).to(self.device)
        sensation_matrix = sense_map(self.mode_mapdata, position, grid=5)
        action_prob = self.actor(state, sensation_matrix)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()

        return action.item()