import torch
import numpy as np
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from rl_utils.env import MapEnv
from rl_utils.descrete_rl_methods import VAnet, Qnet, Policy


state_dim = 12
hidden_dim = 128
action_dim = 8
qnet = VAnet(state_dim, hidden_dim, action_dim)
qnet.load_state_dict(torch.load('model/DQN_15000_eps_inrealmap_117_128x128_sota.pth'))

actor_net = Policy(state_dim, hidden_dim, action_dim)
#actor_net.load_state_dict(torch.load('model/SAC_10000_eps_inrealmap.pth'))

qnet.eval()
actor_net.eval()

# test TG:4845,3;GG: 0, 8 or 9 5; GSD:3088,10 ; TS: 3736,.]5
testid_start =4845
num_tests =3
with open('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/artificial_traj_mixed.csv')
trajmode = traj.loc[testid_start, 'mode']

env = MapEnv(mapdata, traj, test_mode=True, testid_start=testid_start, test_num=num_tests,
             use_real_map=True, realmap_row=326, realmap_col=364)
dxdy_dict = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}

# Define the number of tests
results = []
backimg_path = 'figur/{}_back.jpg'.format(trajmode)
background_img = mpimg.imread(backimg_path)
x_min, y_min =  11111,11111
x_max, y_max = -1,-1
for i in range(num_tests):
    state = env.reset()
    done = False
    total_reward = 0
    distance_to_start = state[:2] # always (0, 0)
    end_to_start = state[2:]
    path = []

    step_cnt = -0
    actions = []
    state_set = set()
    state_set.add(tuple(state[:2]))
    while not done:
        step_cnt += 1
        q_values = qnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        sorted_actions = torch.sort(q_values, descending=True).indices.squeeze().tolist()
        # action = sorted_actions[0]
        # if tuple(state[:2] + dxdy_dict[action]) in state_set:
        #     done = True
        # else:
        #     state_set.add(tuple(state[:2] + dxdy_dict[action]))
        for j in range(len(sorted_actions)):
            tmp_action = sorted_actions[j]
            if tuple(state[:2] + dxdy_dict[tmp_action]) not in state_set:
                action = tmp_action
                break
        next_state, reward, done = env.step(action)
        actions.append(action)
        state = next_state
        print(tuple(state[:2]))
        state_set.add(tuple(state[:2]))
        total_reward += reward
        path.append(state[:2])

    results.append((distance_to_start, end_to_start, total_reward, path, step_cnt, actions))
    # Plot the path and the start and end points
    path = np.array(path)
    if traj.loc[i+testid_start, 'ID'] == traj.loc[i+testid_start + 1, 'ID']:
        plt.plot(path[:, 0] + traj.loc[i+testid_start, 'locx'], path[:, 1] + traj.loc[i+testid_start, 'locy'], marker = 'o',label=f'Test {i+1} Path',markersize = 1, linewidth = 2)
        plt.scatter(distance_to_start[0] + traj.loc[i+testid_start, 'locx'], distance_to_start[1] + traj.loc[i+testid_start, 'locy'], marker='o', color='green',label=f'Test {i+1} Start')
        plt.scatter(end_to_start[0] + traj.loc[i+testid_start, 'locx'], end_to_start[1] + traj.loc[i+testid_start, 'locy'], marker='x', color='red', label=f'Test {i+1} End')
        x_min = min(x_min, distance_to_start[0] + traj.loc[i+testid_start, 'locx'])-2
        y_min = min(y_min, distance_to_start[1] + traj.loc[i+testid_start, 'locy'])-2
        x_max = max(x_max, distance_to_start[0] + traj.loc[i+testid_start, 'locx'])+2
        y_max = max(y_max, distance_to_start[1] + traj.loc[i+testid_start, 'locy'])+2

height, width = background_img.shape[0], background_img.shape[1]
ratio = ((height / 326) *( width / 364))**0.5
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
sliced_background_img = background_img[height-y_max_idx: height-y_min_idx,x_min_idx:x_max_idx]
# Display the background image with the same limits
plt.imshow(sliced_background_img, extent=[x_min, x_max, y_min, y_max],aspect='equal', alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('Agent Paths in Environment')
plt.show(figsize=(20, 16))

# Print total rewards for each test
for i, (_, _, total_reward, _, cnt, action) in enumerate(results):
    print(f'Test {i+1} - Total reward: {total_reward} - Total step: {cnt}')
    print(f'Actions: {action}')