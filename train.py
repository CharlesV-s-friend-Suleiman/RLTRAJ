"""Description: This file is used to recover the trajectory from the HER samples
@Author: yangXiao
the pseudocode of DQN with HER
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
"""

import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from rl_utils.dqn_her import Buffer, DQN, MapEnv, TrainTraj

# set the device & hyperparameters
lr = 0.003
num_episodes = 5200
num_train = 20
hidden_dim = 128
gamma = .98
epsilon = .01
target_update = 10
buffer_size = 10000
minimal_size = 256
batch_size = 64
device = torch.device("cuda")

# load the mapdata and traj, set the env, buffer, agent
with open ('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/artificial_traj_mixed.csv', )
env = MapEnv(mapdata, traj)
buffer = Buffer(buffer_size)
agent = DQN(4, hidden_dim, 8, lr, gamma, epsilon, target_update, device, "dueling")
return_list = []

np.random.seed(42)
torch.manual_seed(42)
# start training
for i in range(10):

    with tqdm(total=int(num_episodes/10), desc = 'Iteration {}'.format(i)) as pbar:
        for e in range(int(num_episodes/10)):
            state = env.reset()
            traj = TrainTraj(state)
            episode_return = 0
            done = False

            # sample trajectory
            while not done:
                action = agent.take_action(state)
                state, reward, done = env.step(action)
                episode_return += reward
                traj.store_step(state, action, reward, done)
            buffer.add_traj(traj)
            return_list.append(episode_return)

            # use HER to sample a batch of samples
            if buffer.size() >= minimal_size:
                for _ in range(num_train):
                    transition_dict = buffer.sample(batch_size, use_her=True)
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
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN with HER on {}'.format('RealMap'))
plt.show()

# save the model
torch.save(agent.target_qnet.state_dict(), 'model/dqn_her.pth')
print('Model saved successfully!')

