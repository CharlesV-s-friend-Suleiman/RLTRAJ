"""Description: This file is used to recover the trajectory from the HER samples
@Author: yangXiao
the pseudocode of DQN/SAC with HER
1. initialize replay buffer R
2. initialize Q-network Q
3. initialize target Q-network Q'== Q
4. for episode = 1, E do:
    get initial state s
    for t = 1, T do:
        select action a = argmax(Q(s,a))
        execute a in the env, get reward r and next state s'
        store <s,a,r,s'> in R
        if enough samples in R, using HER to sample a batch of samples:
            for each sample, if the goal is reached, set reward = 0, done = True
            else, set reward = -1, done = False
            update Q-network
    end for
    update target Q-network
end for
"""

import pickle
import torch
import matplotlib.pyplot as plt
from sympy.abc import alpha
from tqdm import tqdm
import pandas as pd
import numpy as np
from rl_utils.buffer import Buffer, TrainTraj
from rl_utils.env import MapEnv
from rl_utils.descrete_rl_methods import DQN, SAC

# load the mapdata and traj, set the buffer
buffer_size = 10000

with open ('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/artificial_traj_mixed.csv', )
buffer = Buffer(buffer_size)
return_list = []

# set the hyperparameters for all methods
gamma = .98
minimal_size = 1000
batch_size = 128
device = torch.device("cuda")
hidden_dim = 128
env = MapEnv(mapdata, traj, train_num=1260)

# set the device & hyperparameters for DQN
lr = 0.001
num_episodes = 10000
num_train = 20
epsilon = .02
target_update = 10

# set the device & hyperparameters for SAC
actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
tau = 0.005
target_entropy = -1

np.random.seed(42)
torch.manual_seed(42)
# start training

def train(agent, env, episodes, agent_type, use_her, **kwargs):
    ep = 0
    for i in range(10):

        with tqdm(total=int(num_episodes/10), desc = 'Iteration {}'.format(i)) as pbar:

            for e in range(int(num_episodes/10)):
                ep += 1
                state = env.reset()
                traj = TrainTraj(state)
                episode_return = 0
                done = False

                # sample trajectory
                while not done:
                    action = agent.take_action(state) # epsilon-greedy with decay
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(state, action, reward, done)
                buffer.add_traj(traj)
                return_list.append(episode_return)

                # use HER to sample a batch of samples
                if buffer.size() >= minimal_size:
                    for _ in range(num_train):
                        transition_dict = buffer.sample(batch_size, use_her=use_her)
                        agent.update(transition_dict)

                if (e+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + e + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # plot the return
    averge_return_per10 = []
    for i in range(0, len(return_list), 10):
        averge_return_per10.append(np.mean(return_list[i:i+10]))
    plt.plot([i*10 for i in range(len(averge_return_per10))],averge_return_per10)
    plt.xlabel('Episodes')
    plt.ylabel('Returns per 10 episodes')
    plt.title('{} with HER on {}'.format(agent_type,'RealMap'))
    plt.show()

    if agent_type == 'DQN':
        torch.save(agent.target_qnet.state_dict(), 'model/{}_{}_eps_in{}.pth'.format(agent_type, episodes,'realmap'))
    if agent_type == 'SAC':
        torch.save(agent.actor.state_dict(), 'model/{}_{}_eps_in{}.pth'.format(agent_type, episodes,'realmap'))
    print('Model saved successfully!')

    return None


DQN_agent = DQN(4, hidden_dim, 8, lr, gamma, epsilon, target_update, device, "dueling")
SAC_agent = SAC(4, hidden_dim, 8,
                actor_lr = alpha_lr, critic_lr=critic_lr,alpha_lr=alpha_lr,
                target_entropy= target_entropy, gamma = gamma, tau=tau,device = device)

#train(DQN_agent, env, num_episodes, 'DQN', use_her=True)
train(SAC_agent, env, num_episodes, 'SAC', use_her=True)