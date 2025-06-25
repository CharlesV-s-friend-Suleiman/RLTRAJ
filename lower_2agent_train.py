"""
Description:
1. 训练下层模型，两个agent分别从起点和终点出发，直到相遇或达到最大步数
2. 采用了新的数据结构,力求可读可维护性高，之前直接从原始数据csv上进行训练和测试的几个环境类 UpperEnv MapEnv等难以维护,希望zwg参考新类把它们也修改一下

技术对比：
训练时：
1. 两个agent共用一个buffer，在更新网络参数时采样一次更新（即一个网络，act两次）
2. 分开两个actor网络，（critic网络是一样的因为它们干得好不好都是一个标准评价），buffer中采样两次分别训练，保证策略差异性
测试时：
1. 两个agent分别从起点和终点出发，直到相遇或达到最大步数

@Author: yangXiao; Email: yx21@seu.edu.cn
the pseudocode of OD-lower-model (sac method)

"""
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from rl_utils.buffer import Buffer, TrainTraj, TrainTrajwithMapinfo
from rl_utils.env import MapEnv, ODMapEnv
from rl_utils.value_based_rl_methods import DQN
from rl_utils.policy_based_rl_methods import SAC, SACWithConv
from matplotlib.gridspec import GridSpec

# load the mapdata and traj, set the buffer
buffer_size = 2**14
map_row = 529
map_col = 564
with open ('data/GridModesAdjacentRealworld.pkl','rb') as f:
    mapdata = pickle.load(f)
shuffle_traj = pd.read_csv('data/data_lower_train_mixed_0.csv')

buffer = Buffer(buffer_size)
return_list = []

# set the hyperparameters for all methods
gamma = .98
minimal_size = 4096
batch_size = 512
device = torch.device("cuda")
# todo : finish env
env = ODMapEnv(mapdata = mapdata,
               traj = shuffle_traj,
               testid_start=0,
               test_mode=False,
               use_real_map=True,
               realmap_row=map_row,
               realmap_col=map_col)
# set the device & hyperparameters for DQN
lr = 0.001
num_episodes = 20000
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
def train_2agent(agent1, agent2, env, episodes, agent_type, use_her, **kwargs):
    ep = 0
    return_list = [[],[]]
    critic_losses = [[],[]]
    actor_losses = [[],[]]

    for i in range(10):
        with tqdm(total=int(episodes / 10), desc='Iteration {}'.format(i)) as pbar:
            for e in range(int(episodes / 10)):
                ep += 1
                state_dual = env.reset()
                traj_start = TrainTraj(state_dual['start'])
                traj_end = TrainTraj(state_dual['end'])

                episode_return_1 = 0
                episode_return_2 = 0

                done = False

                # sample trajectory
                while not done:
                    state_start = state_dual['start']
                    state_end = state_dual['end']

                    env_max_step = env.max_step
                    # agent1 agent2分别更新各自的策略网络&价值网络，其实价值网络应该合用，但是先开发再优化吧
                    action1 = agent1.take_action(state_start)  # epsilon-greedy with decay
                    action2 = agent2.take_action(state_end)

                    state_dual, r_start, r_end, done = env.step_2agent(action1, action2)
                    episode_return_1 += r_start
                    episode_return_2 += r_end
                    # 同质性轨迹一次采样两条
                    traj_start.store_step(state_dual['start'], action1, r_start, env_max_step, done)
                    traj_end.store_step(state_dual['end'], action2, r_end, env_max_step, done)

                buffer.add_traj(traj_start)
                buffer.add_traj(traj_end)
                return_list[0].append(episode_return_1)
                return_list[1].append(episode_return_2)


                # use HER to sample a batch of samples
                if buffer.size() >= minimal_size:
                    episode_critic_losses = [[],[]]
                    episode_actor_losses = [[],[]]
                    for _ in range(num_train):
                        for j, agent in enumerate([agent1, agent2]):
                            transition_dict = buffer.sample(batch_size, use_her=use_her)
                            critic_loss, actor_loss = agent.update(transition_dict)
                            episode_critic_losses[j].append(critic_loss)
                            episode_actor_losses[j].append(actor_loss)
                    for j in range(2):
                        critic_losses[j].append(np.mean(episode_critic_losses[j]))
                        actor_losses[j].append(np.mean(episode_actor_losses[j]))

                if (e + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (episodes / 10 * i + e + 1),
                        'return': '%.3f' % np.mean(return_list[0][-10:])
                    })
                pbar.update(1)

    # plot the return and losses
    averge_return_per10 = [[],[]]
    for i in range(0, len(return_list), 10):
        averge_return_per10[0].append(np.mean([0][i:i + 10]))
        averge_return_per10[1].append(np.mean([1][i:i + 10]))

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1])  # Create a grid with two equal subplots

    # Left subplot
    ax1 = fig.add_subplot(gs[0])
    color = 'tab:blue'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Returns(pre10) for agent1', color=color)
    ax1.plot([i * 10 for i in range(len(averge_return_per10[0]))], averge_return_per10[0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Loss for agent1', color=color)
    ax2.plot(range(minimal_size, minimal_size + len(critic_losses[0])), critic_losses[0], color=color, label='Critic Loss')
    ax2.plot(range(minimal_size, minimal_size + len(actor_losses[0])), actor_losses[0], color='tab:green', label='Actor Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Right subplot
    ax3 = fig.add_subplot(gs[1])
    color = 'tab:blue'
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Returns(pre10) for agent2 ', color=color)
    ax3.plot([i * 10 for i in range(len(averge_return_per10[1]))], averge_return_per10[1], color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax4.set_ylabel('Loss', color=color)
    ax4.plot(range(minimal_size, minimal_size + len(critic_losses[1])), critic_losses[1], color=color, label='Critic Loss')
    ax4.plot(range(minimal_size, minimal_size + len(actor_losses[1])), actor_losses[1], color='tab:green', label='Actor Loss')
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.suptitle('{} with HER on {}'.format(agent_type, 'RealMap'))  # Add a common title
    plt.show()

    torch.save(agent1.actor.state_dict(),
                   'lower_model/{}_{}_eps_in{}_{}.pth'.format(agent_type, episodes, 'realmap',
                                                        str(datetime.datetime.now().month) + str(
                                                            datetime.datetime.now().day)))
    torch.save(agent2.actor.state_dict(),
                   'lower_model/{}_{}_eps_in{}_{}.pth'.format(agent_type, episodes, 'realmap',
                                                        str(datetime.datetime.now().month) + str(
                                                            datetime.datetime.now().day)))
    print('Model saved successfully!')

    return None

# normal sac
SAC_agent1 = SAC(12, 64, 8,
                actor_lr = alpha_lr, critic_lr=critic_lr,alpha_lr=alpha_lr,
                target_entropy= target_entropy, gamma = gamma, tau=tau,device = device,
                using_realmap=True,mapdata =env.mapdata)
SAC_agent2 = SAC(12, 64, 8,
                actor_lr = alpha_lr, critic_lr=critic_lr,alpha_lr=alpha_lr,
                target_entropy= target_entropy, gamma = gamma, tau=tau,device = device,
                using_realmap=True,mapdata =env.mapdata)



train_2agent(SAC_agent1, SAC_agent1, env, num_episodes, 'SAC', use_her=True)

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