import torch
import numpy as np
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rl_utils.env import MapEnv, dxdy_dict
from rl_utils.value_based_rl_methods import VAnet
from rl_utils.policy_based_rl_methods import Policy, PolicyWithConv
from rl_utils.tools import sense_map

test_type = 'SAC'  # or 'SAC'
map_row = 529
map_col = 564
state_dim = 12
hidden_dim = 64
action_dim = 8
qnet = VAnet(state_dim, hidden_dim, action_dim)
# qnet.load_state_dict(torch.load('lower_model/DQN_30000_eps_inrealmap_628.pth'))

test_with_conv = False
# sac without conv
actor_net = Policy(state_dim, hidden_dim, action_dim)
actor_net.load_state_dict(torch.load('lower_model/gaiReward_SAC_10000_eps_inrealmap_627.pth'))

# # sac with conv
# actor_net = PolicyWithConv(state_dim, hidden_dim, action_dim,5*5)
# actor_net.load_state_dict(torch.load('lower_model/SAC_3000_eps_inrealmap_626.pth'))

qnet.eval()
actor_net.eval()

mode_v_dict = {'TG': 300, 'GG': 120, 'GSD': 60, 'TS': 150}
modelist = ['GSD', 'GG', 'TS', 'TG']

# test TG:461/13,857/20,1128/16; GG: 4712/10,50/20; GSD:295/11,428/20 ; TS: 178/17,2577/15
testid_start = 10
num_tests = 14
with open('data/GridModesAdjacentRealworld.pkl', 'rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/data_train_upper_250624.csv')
trajmode = traj.loc[testid_start, 'mode']

env = MapEnv(mapdata, traj, test_mode=True, testid_start=testid_start, test_num=num_tests,
             use_real_map=True, realmap_row=map_row, realmap_col=map_col)

# Define the number of tests
results = []
# backimg_path = 'figur/{}_back.jpg'.format(trajmode) # 326x364 map
backimg_path = 'figur/{}_plot_js.png'.format(trajmode)  # 529x564 map

background_img = mpimg.imread(backimg_path)
x_min, y_min = 11111, 11111
x_max, y_max = -1, -1

all_trajs = []
for i in range(num_tests):
    state = env.reset()
    done = False
    total_reward = 0
    distance_to_start = state[:2]  # always (0, 0)
    end_to_start = state[2:]
    path = []
    actual_pos = []

    step_cnt = 0
    actions = []
    state_set = set()
    state_set.add(tuple(state[:2]))
    while not done:
        step_cnt += 1
        if test_type == 'DQN':  # TODO: DQN的方法需要补起来，孟，张 补一下
            q_values = qnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
            action = sorted_actions[0]
            if tuple(state[:2] + dxdy_dict[action]) in state_set:
                done = True
            else:
                state_set.add(tuple(state[:2] + dxdy_dict[action]))
            for j in range(len(sorted_actions)):
                tmp_action = sorted_actions[j]
                if tuple(state[:2] + dxdy_dict[tmp_action]) not in state_set:
                    action = tmp_action
                    break

        # if test_type == 'SAC': #Z or other policy-based methods
        #     if test_with_conv:
        #         # state tensor size is (1, state_dim); mapinfo tensor size is
        #         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        #         sensation_mtx = sense_map(env.mapdata[trajmode], [state[:2]+ env.delta], grid=5)
        #         sensation_mtx = sensation_mtx.to('cpu')
        #         # state tensor (1, state_dim) ; sensation_mtx tensor (1, grid, grid)
        #         action = int(actor_net(state_tensor, sensation_mtx).argmax())
        #     else:
        #         action = int(actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax())

        # 改:避免重复道路
        if test_type == 'SAC':  # Z or other policy-based methods
            if test_with_conv:
                # state tensor size is (1, state_dim); mapinfo tensor size is
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                sensation_mtx = sense_map(env.mapdata[trajmode], [state[:2] + env.delta], grid=5)
                sensation_mtx = sensation_mtx.to('cpu')
                # state tensor (1, state_dim); sensation_mtx tensor (1, grid, grid)
                action_scores = actor_net(state_tensor, sensation_mtx)
            else:
                action_scores = actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))

            sorted_actions = torch.sort(action_scores, descending=True).indices.squeeze().tolist()

            for action in sorted_actions:
                if len(actions) == 0:
                    action = sorted_actions[0]
                elif not (dxdy_dict[action][0] + dxdy_dict[actions[-1]][0] == 0 and dxdy_dict[action][1] + dxdy_dict[actions[-1]][1] == 0):
                    break
                else:
                    action = sorted_actions[0]  # Fallback to the best action if no valid action is found

        next_state, reward, done = env.step(action)
        actions.append(action)
        state = next_state
        state_set.add(tuple(state[:2]))
        total_reward += reward
        path.append(state[:2])
        actual_pos.append(state[:2] + env.delta)
    all_trajs.append(actual_pos)

    results.append((distance_to_start, end_to_start, total_reward, path, step_cnt, actions))
    # Plot the path and the start and end points
    path = np.array(path)
    print(path)

    if traj.loc[i + testid_start, 'ID'] == traj.loc[i + testid_start + 1, 'ID']:
        plt.plot(path[:, 0] + traj.loc[i + testid_start, 'locx_o'], path[:, 1] + traj.loc[i + testid_start, 'locy_o'],
                 marker='o', label=f'Test {i + 1} Path', markersize=1, linewidth=2)
        plt.scatter(distance_to_start[0] + traj.loc[i + testid_start, 'locx_o'],
                    distance_to_start[1] + traj.loc[i + testid_start, 'locy_o'], marker='o', color='green',
                    label=f'Test {i + 1} Start')
        plt.scatter(end_to_start[0] + traj.loc[i + testid_start, 'locx_o'],
                    end_to_start[1] + traj.loc[i + testid_start, 'locy_o'], marker='x', color='red',
                    label=f'Test {i + 1} End')
        x_min = min(x_min, distance_to_start[0] + traj.loc[i + testid_start, 'locx_o']) - 2
        y_min = min(y_min, distance_to_start[1] + traj.loc[i + testid_start, 'locy_o']) - 2
        x_max = max(x_max, distance_to_start[0] + traj.loc[i + testid_start, 'locx_o']) + 2
        y_max = max(y_max, distance_to_start[1] + traj.loc[i + testid_start, 'locy_o']) + 2

        t_upper = traj.loc[i + testid_start + 1, 'time'] - traj.loc[i + testid_start, 'time']
        # compute t_lower and t_upper
        upper_mode = traj.loc[i + testid_start, 'mode']
        t_lower = 0
        v_expected = mode_v_dict[upper_mode] / 60
        v_rural = 0.6  # 36km/h

        for j, coord in enumerate(actual_pos[1:]):
            x, y = int(coord[0]), int(coord[1])
            if env.mapdata[upper_mode][x][y] == 0:
                t_lower += ((actual_pos[j][0] - actual_pos[j - 1][0]) ** 2 + (
                        actual_pos[j][1] - actual_pos[j - 1][1]) ** 2) ** 0.5 / v_expected
                # print('lower path is blocked, and travel in rural area', x, y)
            else:
                t_lower += ((actual_pos[j][0] - actual_pos[j - 1][0]) ** 2 + (
                        actual_pos[j][1] - actual_pos[j - 1][1]) ** 2) ** 0.5 / v_expected
        print('Iter', i, 't_upper:', t_upper, 't_lower:', t_lower)
    print('rts_predicted is :', actual_pos)

height, width = background_img.shape[0], background_img.shape[1]
ratio = ((height / map_row) * (width / map_col)) ** 0.5
x_min_idx = int(x_min * ratio)
x_max_idx = int(x_max * ratio)
y_min_idx = int(y_min * ratio)
y_max_idx = int(y_max * ratio)

# Print the calculated indices for debugging
print(f"x_min_idx: {x_min_idx}, x_max_idx: {x_max_idx}")
print(f"y_min_idx: {y_min_idx}, y_max_idx: {y_max_idx}")

# Ensure indices are within the valid range
x_min_idx = max(0, x_min_idx)
x_max_idx = min(background_img.shape[0], x_max_idx)
y_min_idx = max(0, y_min_idx)
y_max_idx = min(background_img.shape[1], y_max_idx)

# Slice the background image
sliced_background_img = background_img[height - y_max_idx: height - y_min_idx, x_min_idx:x_max_idx]
# Display the background image with the same limits
plt.imshow(sliced_background_img, extent=[x_min, x_max, y_min, y_max], aspect='equal', alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('Agent Paths in Environment')
plt.show()

# Print total rewards for each test
for i, (distance_to_start, end_to_start, total_reward, path, cnt, action) in enumerate(results):
    print(f'Test {i + 1} - Total reward: {total_reward} - Total step: {cnt}')
    print(f'Path: {path}')
    print(f'Distance: {distance_to_start} - End: {end_to_start}')
    print(f'Actions: {action}')

# test for all tarj
