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
qnet.load_state_dict(torch.load('model/DQN_10000_eps_inrealmap_1023.pth'))

actor_net = Policy(state_dim, hidden_dim, action_dim)
actor_net.load_state_dict(torch.load('model/SAC_10000_eps_inrealmap_1022.pth'))

qnet.eval()
actor_net.eval()

testid_start = 0
num_tests = 8

with open('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/artificial_traj_mixed.csv')
env = MapEnv(mapdata, traj, test_mode=True, testid_start=testid_start, test_num=num_tests,
             use_real_map=True, realmap_row=326, realmap_col=364)

# Define the number of tests
results = []
background_img = mpimg.imread('figur/GG_plot_with_mapdata.png')
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
        x_min = min(x_min, distance_to_start[0] + traj.loc[i+testid_start, 'locx'])
        y_min = min(y_min, distance_to_start[1] + traj.loc[i+testid_start, 'locy'])
        x_max = max(x_max, distance_to_start[0] + traj.loc[i+testid_start, 'locx'])
        y_max = max(y_max, distance_to_start[1] + traj.loc[i+testid_start, 'locy'])


# bg img size is 868x969; x,y size is 326x364 ;when transiting, x_new = x*2.662, y_new = y*2.662
sliced_background_img = background_img[int(x_min*2.662): int(x_max*2.662),int(y_min*2.662):int(y_max*2.662)]
# Display the background image with the same limits
#plt.imshow(sliced_background_img,extent=[x_min,y_min,x_max,y_max],aspect='equal', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.title('Agent Paths in Environment')
plt.show(figsize=(20, 16))

# Print total rewards for each test
for i, (_, _, total_reward, _) in enumerate(results):
    print(f'Test {i+1} - Total reward: {total_reward}')