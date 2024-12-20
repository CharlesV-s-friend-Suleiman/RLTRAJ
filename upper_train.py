"""
Description: This script is used to train the upper model using DQN.
@Author: yangXiao; Email: yx21@seu.edu.cn
the pseudocode of DQN for upper model is as follows:
Initialize replay memory R
Initialize pretrained lower model F(traj,mode) -> routes
Initialize Q-network Q
Initialize target Q-network Q'== Q

"""

import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

from rl_utils.buffer import Buffer, TrainTraj
from rl_utils.env import UpperEnv
from rl_utils.descrete_rl_methods import DQN

# load map data and traj data
buffer_size = 20000
minimal_size = 2048
buffer = Buffer(buffer_size)
with open ('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
trajdata = pd.read_csv('data/train10000.csv')

# load hyperparameters for upper model
state_dim = 5
action_dim = 20
hidden_dim = 128
lr = 0.003
gamma = 0.9
batch_size = 256
target_update = 50
num_episodes = 20000
epsilon = 0.1
num_train = 20

# load lower model config
lower_model_config = {
    'state_dim': 12,
    'hidden_dim': 128,
    'action_dim': 8,
    'model_path': './lower_model/DQN_15000_eps_inrealmap_1110_sota.pth',
}

env = UpperEnv(mapdata,trajdata,trainid_start=0, train_num=9000,m=4,
               use_real_map=True,realmap_col=364,realmap_row=326,
               lower_model_config=lower_model_config)
upper_agent  = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,target_update,device='cuda',
                   dqn_type="dueling", using_realmap=True, mapdata=mapdata)

def train_uppermodel(agent, env, episodes,agent_type, use_her=False):
    return_list = []
    losses = []
    for i in range(10):
        with tqdm(total = int(episodes/10), desc='Iteration %d' % i) as pbar:
            for e in range(int(episodes/10)):
                s = env.reset()
                traj = TrainTraj(s)
                done = False
                episode_return = 0

                # sample
                while not done:
                    a = agent.take_action(s)
                    s, r, done = env.step_with20action(a)
                    episode_return += r
                    traj.store_step(s,a,r,None,done) # max_step is not used in upper model
                buffer.add_traj(traj)
                return_list.append(episode_return)

                # batch update with adam
                if buffer.size() > minimal_size:
                    episode_loss = []
                    for _ in range(num_train):
                        loss = 0
                        trainsition_dict = buffer.sample(batch_size, use_her)
                        loss += agent.update(trainsition_dict)
                        episode_loss.append(loss/num_train)
                    losses.append(np.mean(episode_loss))

                # update gui
                if (e + 1)% 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (episodes / 10 * i + e + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # plot the return and loss
    averge_return_per10 = []
    for i in range(0, len(return_list), 10):
        averge_return_per10.append(np.mean(return_list[i:i + 10]))

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Returns per 10 episodes', color=color)
    ax1.plot([i * 10 for i in range(len(averge_return_per10))], averge_return_per10, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1

    ax2.plot(range(minimal_size, minimal_size + len(losses)), losses, color=color, label='Average Q-Loss per 10 episodes')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('{} Travel Mode Choice on {}'.format(agent_type, 'RealMap'))
    plt.size = (30, 20)
    plt.savefig('upper_model/{}_{}_eps_in{}_{}.png'.format(agent_type, episodes, 'realmap',
                                                    str(datetime.datetime.now().month) + str( datetime.datetime.now().day)))
    plt.show()

    torch.save(agent.target_qnet.state_dict(),
                'upper_model/{}_{}_eps_in{}_{}.pth'.format(agent_type, episodes, 'realmap',
                                                    str(datetime.datetime.now().month) + str( datetime.datetime.now().day)))
    print('Model saved successfully!')

    return None

#train_uppermodel(upper_agent, env, num_episodes, 'DQN', use_her=False)
train_uppermodel(agent = upper_agent, env = env, episodes=num_episodes,
      agent_type = 'DQN', use_her=False)

