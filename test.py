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
qnet.load_state_dict(torch.load('model/DQN_10000_eps_inrealmap_1024_4_1_nhash.pth'))

actor_net = Policy(state_dim, hidden_dim, action_dim)
actor_net.load_state_dict(torch.load('model/SAC_10000_eps_inrealmap_1022.pth'))

qnet.eval()
actor_net.eval()

testid_start = 3689
num_tests = 5


with open('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/artificial_traj_mixed.csv')
trajmode = traj.loc[testid_start, 'mode']

env = MapEnv(mapdata, traj, test_mode=True, testid_start=testid_start, test_num=num_tests,
             use_real_map=True, realmap_row=326, realmap_col=364)

# Define the number of tests
results = []
backimg_path = 'figur/{}_plot_with_mapdata.png'.format(trajmode)
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

    while not done:
        #action = actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item() # SAC
        action = qnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item() # DQN
        state, reward, done = env.step(action)
        #print(env.traj_cnt,  i,state, reward) # print the state and reward for debugging
        total_reward += reward
        path.append(state[:2])

    results.append((distance_to_start, end_to_start, total_reward, path))
    # Plot the path and the start and end points
    path = np.array(path)
    if traj.loc[i+testid_start, 'ID'] == traj.loc[i+testid_start + 1, 'ID']:
        plt.plot(path[:, 0] + traj.loc[i+testid_start, 'locx'], path[:, 1] + traj.loc[i+testid_start, 'locy'], marker = 'o',label=f'Test {i+1} Path',markersize = 1, linewidth = 1)
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
plt.imshow(sliced_background_img, extent=[x_min, x_max, y_min, y_max],aspect='equal', alpha=1)

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('Agent Paths in Environment')
plt.show(figsize=(20, 16))

# Print total rewards for each test
for i, (_, _, total_reward, _) in enumerate(results):
    print(f'Test {i+1} - Total reward: {total_reward}')