"""
Description: This file is used to recover the trajectory from the HER samples
@Author: yangXiao; Email: yx21@seu.edu.cn
the pseudocode of DQN/SAC with HER for lower model
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
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from rl_utils.buffer import Buffer, TrainTraj
from rl_utils.env import MapEnv
from rl_utils.descrete_rl_methods import DQN, SAC

# load the mapdata and traj, set the buffer
buffer_size = 15000

with open ('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
shuffle_traj = pd.read_csv('data/train10000.csv')

buffer = Buffer(buffer_size)
return_list = []

# set the hyperparameters for all methods
gamma = .96
minimal_size = 2048
batch_size = 256
device = torch.device("cuda")
hidden_dim = 128
env = MapEnv(mapdata, shuffle_traj, train_num=9990,trainid_start=0, use_real_map=True, realmap_row=326, realmap_col=364)

#trainwithTGTS_env=MapEnv(mapdata,traj, train_num=2000, trainid_start=3276, use_real_map=True,realmap_row=326, realmap_col=364)
#trainwithTG_env=MapEnv(mapdata,traj, train_num=800, trainid_start=4437, use_real_map=True,realmap_row=326, realmap_col=364)


# set the device & hyperparameters for DQN
lr = 0.001
num_episodes = 15000
num_train = 20
epsilon = .05
target_update = 50

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
    return_list = []
    losses = []
    critic_losses = []
    actor_losses = []

    for i in range(10):
        with tqdm(total=int(episodes / 10), desc='Iteration {}'.format(i)) as pbar:
            for e in range(int(episodes / 10)):
                ep += 1
                state = env.reset()
                agent.visited_states.clear()  # Clear the visited states set
                traj = TrainTraj(state)
                episode_return = 0
                done = False

                # sample trajectory
                while not done:
                    mapinfo = env.delta
                    env_max_step = env.max_step
                    action = agent.take_action(state)  # epsilon-greedy with decay
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(state, action, reward, env_max_step, done)
                buffer.add_traj(traj)
                return_list.append(episode_return)

                # use HER to sample a batch of samples
                if buffer.size() >= minimal_size:
                    episode_losses = []
                    episode_critic_losses = []
                    episode_actor_losses = []
                    for _ in range(num_train):
                        loss = 0
                        critic_loss = 0
                        actor_loss = 0
                        transition_dict = buffer.sample(batch_size, use_her=use_her)
                        if agent_type == 'SAC':
                            critic_loss, actor_loss = agent.update(transition_dict)
                            episode_critic_losses.append(critic_loss)
                            episode_actor_losses.append(actor_loss)
                        elif agent_type == 'DQN':
                            loss += agent.update(transition_dict)
                            episode_losses.append(loss / num_train)

                    if agent_type == 'SAC':
                        critic_losses.append(np.mean(episode_critic_losses))
                        actor_losses.append(np.mean(episode_actor_losses))
                    elif agent_type == 'DQN':
                        losses.append(np.mean(episode_losses))

                if (e + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (episodes / 10 * i + e + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    # plot the return and losses
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

    if agent_type == 'SAC':
        ax2.plot(range(minimal_size, minimal_size + len(critic_losses)), critic_losses, color=color, label='Critic Loss')
        ax2.plot(range(minimal_size, minimal_size + len(actor_losses)), actor_losses, color='tab:green', label='Actor Loss')
    elif agent_type == 'DQN':
        ax2.plot(range(minimal_size, minimal_size + len(losses)), losses, color=color, label='Average Q-Loss per 10 episodes')

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('{} with HER on {}'.format(agent_type, 'RealMap'))
    plt.show()

    if agent_type == 'DQN':
        torch.save(agent.target_qnet.state_dict(),
                   'lower_model/{}_{}_eps_in{}_{}.pth'.format(agent_type, episodes, 'realmap',
                                                        str(datetime.datetime.now().month) + str(
                                                            datetime.datetime.now().day)))
    if agent_type == 'SAC':
        torch.save(agent.actor.state_dict(),
                   'lower_model/{}_{}_eps_in{}_{}.pth'.format(agent_type, episodes, 'realmap',
                                                        str(datetime.datetime.now().month) + str(
                                                            datetime.datetime.now().day)))
    print('Model saved successfully!')

    return None
# def train(agent, env, episodes, agent_type, use_her, **kwargs):
#     ep = 0
#     return_list = []
#     losses = []
#     critic_losses = []
#     actor_losses = []
#
#     for i in range(10):
#         with tqdm(total=int(episodes/10), desc='Iteration {}'.format(i)) as pbar:
#             for e in range(int(episodes/10)):
#                 ep += 1
#                 state = env.reset()
#                 agent.visited_states.clear()  # Clear the visited states set
#                 traj = TrainTraj(state)
#                 episode_return = 0
#                 done = False
#
#                 # sample trajectory
#                 while not done:
#                     mapinfo = env.delta
#                     action = agent.take_action(state, mapinfo)  # epsilon-greedy with decay
#                     state, reward, done = env.step(action)
#                     episode_return += reward
#                     traj.store_step(state, action, reward, done)
#                 buffer.add_traj(traj)
#                 return_list.append(episode_return)
#
#                 # use HER to sample a batch of samples
#                 if buffer.size() >= minimal_size:
#                     for _ in range(num_train):
#                         loss = 0
#                         critic_loss = 0
#                         actor_loss = 0
#                         transition_dict = buffer.sample(batch_size, use_her=use_her)
#                         if agent_type == 'SAC':
#                             critic_loss, actor_loss = agent.update(transition_dict)
#                             critic_losses.append(critic_loss)
#                             actor_losses.append(actor_loss)
#                         elif agent_type == 'DQN':
#                             loss += agent.update(transition_dict)
#                             losses.append(loss/num_train)
#
#                 if (e + 1) % 10 == 0:
#                     pbar.set_postfix({
#                         'episode': '%d' % (episodes / 10 * i + e + 1),
#                         'return': '%.3f' % np.mean(return_list[-10:])
#                     })
#                 pbar.update(1)
#
#     # plot the return and losses
#     averge_return_per10 = []
#     for i in range(0, len(return_list), 10):
#         averge_return_per10.append(np.mean(return_list[i:i + 10]))
#
#     fig, ax1 = plt.subplots()
#
#     color = 'tab:blue'
#     ax1.set_xlabel('Episodes')
#     ax1.set_ylabel('Returns per 10 episodes', color=color)
#     ax1.plot([i * 10 for i in range(len(averge_return_per10))], averge_return_per10, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
#
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#     color = 'tab:red'
#     ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
#
#     if agent_type == 'SAC':
#         ax2.plot(range(len(critic_losses)), critic_losses, color=color, label='Critic Loss')
#         ax2.plot(range(len(actor_losses)), actor_losses, color='tab:green', label='Actor Loss')
#     elif agent_type == 'DQN':
#         ax2.plot(range(len(losses)), losses, color=color, label='Loss')
#
#     ax2.tick_params(axis='y', labelcolor=color)
#
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.title('{} with HER on {}'.format(agent_type, 'RealMap'))
#     plt.show()
#
#     if agent_type == 'DQN':
#         torch.save(agent.target_qnet.state_dict(),
#                    'lower_model/{}_{}_eps_in{}_{}.pth'.format(agent_type, episodes, 'realmap',
#                                                         str(datetime.datetime.now().month) + str(
#                                                             datetime.datetime.now().day)))
#     if agent_type == 'SAC':
#         torch.save(agent.actor.state_dict(),
#                    'lower_model/{}_{}_eps_in{}_{}.pth'.format(agent_type, episodes, 'realmap',
#                                                         str(datetime.datetime.now().month) + str(
#                                                             datetime.datetime.now().day)))
#     print('Model saved successfully!')
#
#     return None

### main function ###

DQN_agent = DQN(12, hidden_dim, 8, lr, gamma, epsilon, target_update, device,
                "dueling",using_realmap=True, mapdata=mapdata)

SAC_agent = SAC(12, hidden_dim, 8,
                actor_lr = alpha_lr, critic_lr=critic_lr,alpha_lr=alpha_lr,
                target_entropy= target_entropy, gamma = gamma, tau=tau,device = device,
                using_realmap=True, mapdata=mapdata)

train(DQN_agent, env, num_episodes, 'DQN', use_her=True)
#train(SAC_agent, env, num_episodes, 'SAC', use_her=True)

### main function ###