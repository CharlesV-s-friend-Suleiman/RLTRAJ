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
from rl_utils.buffer import Buffer, TrainTraj, TrainTrajwithMapinfo
from rl_utils.env import MapEnv
from rl_utils.value_based_rl_methods import DQN
from rl_utils.policy_based_rl_methods import SAC, SACWithConv

# load the mapdata and traj, set the buffer
buffer_size = 4096
map_row = 529
map_col = 564
with open ('data/GridModesAdjacentRealworld.pkl','rb') as f:
    mapdata = pickle.load(f)
shuffle_traj = pd.read_csv('data/data_lower_train_random.csv')

buffer = Buffer(buffer_size)
return_list = []

# set the hyperparameters for all methods
gamma = .98
minimal_size = 0
batch_size = 256
device = torch.device("cuda")
hidden_dim = 64
env = MapEnv(mapdata, shuffle_traj, use_real_map=True, realmap_row=map_row, realmap_col=map_col)

# set the device & hyperparameters for DQN
lr = 0.001
num_episodes = 30000
num_train = 20
epsilon = .05
target_update = 50

# set the device & hyperparameters for SAC
actor_lr = 1e-4
critic_lr = 1e-3
alpha_lr = 1e-3
tau = 0.005
target_entropy = -1

np.random.seed(42)
torch.manual_seed(42)

# start training
def train(agent, env, episodes, agent_type, use_her, with_conv, **kwargs):
    ep = 0
    return_list = []
    losses = []
    critic_losses = []
    actor_losses = []

    import csv

    # 初始化 CSV 数据列表
    csv_data = []

    for i in range(10):
        with tqdm(total=int(episodes / 10), desc='Iteration {}'.format(i)) as pbar:
            for e in range(int(episodes / 10)):
                ep += 1
                state = env.reset()
                agent.visited_states.clear()  # 清空访问过的状态
                traj = TrainTrajwithMapinfo(state, state[:2] + env.delta) if with_conv else TrainTraj(state)
                episode_return = 0
                done = False

                # 采样轨迹
                while not done:
                    env_max_step = env.max_step
                    if with_conv:
                        agent.set_mode(env.mode)
                        action = agent.take_action_with_conv(state, state[:2] + env.delta)
                    else:
                        action = agent.take_action(state)  # epsilon-greedy with decay
                    state, reward, done = env.step(action)
                    episode_return += reward
                    if with_conv:
                        cur_pos = state[:2] + env.delta  # [x y]
                        traj.store_step_withmapinfo(state, action, reward, env_max_step, cur_pos, done)
                    else:
                        traj.store_step(state, action, reward, env_max_step, done)
                buffer.add_traj(traj)
                return_list.append(episode_return)

                # 准备 CSV 数据
                if agent_type == 'SAC':
                    row = [ep, episode_return, None, None]  # 默认损失值为 None
                elif agent_type == 'DQN':
                    row = [ep, episode_return, None]  # 默认损失值为 None
                if buffer.size() >= minimal_size:
                    episode_losses = []
                    episode_critic_losses = []
                    episode_actor_losses = []
                    for _ in range(num_train):
                        loss = 0
                        if with_conv:
                            transition_dict = buffer.sample_with_mapinfo(batch_size, use_her=use_her)
                        else:
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
                        row[2] = np.mean(episode_critic_losses)  # 更新 critic_loss
                        row[3] = np.mean(episode_actor_losses)  # 更新 actor_loss
                    elif agent_type == 'DQN':
                        losses.append(np.mean(episode_losses))
                        row[2] = np.mean(episode_losses)  # 更新 DQN 损失

                if (e + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (episodes / 10 * i + e + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })

                csv_data.append(row)  # 添加行到 CSV 数据
                pbar.update(1)

    # 写入 CSV 文件
    csv_file_name = 'lower_model/{}_{}_eps_in{}_{}_training_results.csv'.format(agent_type, episodes, 'realmap',
                                                        str(datetime.datetime.now().month) + str(
                                                            datetime.datetime.now().day))
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        if agent_type == 'SAC':
            writer.writerow(['Episode', 'Return', 'Critic Loss', 'Actor Loss'])  # 写入表头
        elif agent_type == 'DQN':
            writer.writerow(['Episode', 'Return', 'Loss'])  # 写入表头
        writer.writerows(csv_data)
    print(f"结果已保存到 {csv_file_name}")

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
    plt.savefig('lower_model/{}_{}_eps_in{}_{}_Reward & Loss.png'.format(agent_type, episodes, 'realmap',
                                                        str(datetime.datetime.now().month) + str(
                                                            datetime.datetime.now().day)))
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
#
DQN_agent = DQN(12, hidden_dim, 8, lr, gamma, epsilon, target_update, device,
                "dueling",using_realmap=True)

# # normal sac
# SAC_agent = SAC(12, hidden_dim, 8,
#                 actor_lr = actor_lr, critic_lr=critic_lr,alpha_lr=alpha_lr,
#                 target_entropy= target_entropy, gamma = gamma, tau=tau,device = device,
#                 using_realmap=True,mapdata =env.mapdata)

# SAC_agent = SACWithConv(12, hidden_dim, 8,
#                 actor_lr = actor_lr, critic_lr=critic_lr,alpha_lr=alpha_lr,
#                 target_entropy= target_entropy, gamma = gamma, tau=tau,device = device,
#                 using_realmap=True,mapdata =env.mapdata)

train(DQN_agent, env, num_episodes, 'DQN', use_her=True, with_conv = False)
# train(SAC_agent, env, num_episodes, 'SAC', use_her=True, with_conv = False)

### main function ###
# Function to save training configuration
def save_training_config(file_name, config):
    with open(file_name, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

# Define the configuration parameters
config = {
    'learning_rate': lr,
    'gamma': gamma,
    'batch_size': batch_size,
    'num_episodes': num_episodes,
    'epsilon': epsilon,
    'target_update': target_update,
    'actor_lr': actor_lr,
    'critic_lr': critic_lr,
    'alpha_lr': alpha_lr,
    'tau': tau,
    'target_entropy': target_entropy,
    'buffer_size': buffer_size,
    'minimal_size': minimal_size,
    'num_train': num_train,
    'hidden_dim': hidden_dim,
    'device': device,
    'map_row': map_row,
    'map_col': map_col
}

# Save the configuration to a text file
model_name = 'lower_model/{}_eps_in{}_{}.pth'.format(num_episodes, 'realmap',
                                                        str(datetime.datetime.now().month) + str(
                                                            datetime.datetime.now().day))
config_file_name = model_name.replace('.pth', '.txt')
save_training_config(config_file_name, config)