import torch
import numpy as np
import pickle
import pandas as pd

from rl_utils.env import MapEnv, dxdy_dict
from rl_utils.policy_based_rl_methods import Policy
import matplotlib.pyplot as plt

map_row = 529
map_col = 564
state_dim = 12
hidden_dim = 64
action_dim = 8

actor_net = Policy(state_dim, hidden_dim, action_dim)
actor_net.load_state_dict(torch.load('lower_model/gaiReward_SAC_10000_eps_inrealmap_627.pth'))
actor_net.eval()

with open('data/GridModesAdjacentRealworld.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/data_train_upper_250624.csv')

env = MapEnv(mapdata, traj, test_mode=True, testid_start=0, test_num=12000,
             use_real_map=True, realmap_row=map_row, realmap_col=map_col)

all_trajs = []
results = []
match_rates = []
modes = [] # store the mode of each traj
num_finish = 0

for i in range(12000):
    state = env.reset()
    done = False
    total_reward = 0
    distance_to_start = state[:2] # always (0, 0)
    end_to_start = state[2:4]
    path = []
    actual_pos = []

    step_cnt = 0
    actions = []
    state_set = set()
    state_set.add(tuple(state[:2]))
    while not done:
        step_cnt += 1

        #选取动作避免重复
        action_scores = actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        sorted_actions = torch.sort(action_scores, descending=True).indices.squeeze().tolist()
        for action in sorted_actions:
            if len(actions) == 0:
                action = sorted_actions[0]
            elif not (dxdy_dict[action][0] + dxdy_dict[actions[-1]][0] == 0 and dxdy_dict[action][1] +
                      dxdy_dict[actions[-1]][1] == 0):
                break
            else:
                action = sorted_actions[0]  # Fallback to the best action if no valid action is found
        # action = int(actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax())

        next_state, reward, done = env.step(action)
        actions.append(action)
        state = next_state
        state_set.add(tuple(state[:2]))
        total_reward += reward
        path.append(state[:2])
        actual_pos.append(state[:2]+ env.delta)
    all_trajs.append(actual_pos)

    total_step = 0
    match_step = 0
    # caculate match rate
    for j, coord in enumerate(actual_pos[1:]):
        x,y = int(coord[0]), int(coord[1])
        if env.mapdata[env.mode][x][y] == 1:
            match_step += 1
        total_step += 1

    mahattan_dis = abs(end_to_start[0]) + abs(end_to_start[1])
    if step_cnt <= mahattan_dis:
        match_rate = match_step / (total_step +0.0001)
        match_rates.append(match_rate)
        modes.append(env.mode)
        num_finish += 1
    else:
        print('not finished traj id:',i)
    results.append((distance_to_start, end_to_start, total_reward, path, step_cnt, actions))
    # Plot the path and the start and end points
    path = np.array(path)



print('finish rate:',num_finish/len(results), 'TOTAL:',len(results),'FINISH:',num_finish)
print('average match rate:',np.mean(match_rates))

# 画出四个模式的匹配率分布情况，安0.1分箱；同时输出每个模式的匹配率平均值
mode_match_rates = {mode: [] for mode in ['TG', 'TS','GSD','GG']}

# 将 match_rates 按照 modes 分类
for rate, mode in zip(match_rates, modes):
    if mode in mode_match_rates:
        mode_match_rates[mode].append(rate)

# 创建子图
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()

# 绘制每种模式的分布
for i, (mode, rates) in enumerate(mode_match_rates.items()):
    axes[i].hist(rates, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[i].set_title(f'Match Rates Distribution for Mode: {mode}')
    axes[i].set_xlabel('Match Rate')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True)

# 调整布局
plt.tight_layout()
plt.show()
