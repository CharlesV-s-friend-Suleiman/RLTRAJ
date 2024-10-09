import torch
import numpy as np
import pickle
from rl_utils.dqn_her import Qnet, MapEnv, VAnet
import pandas as pd
import matplotlib.pyplot as plt

state_dim = 4
hidden_dim = 128
action_dim = 8
qnet = VAnet(state_dim, hidden_dim, action_dim)
qnet.load_state_dict(torch.load('model/dqn_her.pth'))
qnet.eval()

with open('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/artificial_traj_mixed.csv')
env = MapEnv(mapdata, traj)
num_tests = 8

# Define the number of tests
results = []

for i in range(num_tests):
    state = env.reset(train_multi_point =True)
    done = False
    total_reward = 0
    distance_to_start = state[:2] # always (0, 0)
    end_to_start = state[2:]
    path = []

    while not done:
        action = qnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
        state, reward, done = env.step(action)
        total_reward += reward
        path.append(state[:2])

    results.append((distance_to_start, end_to_start, total_reward, path))
    # Plot the path and the start and end points
    path = np.array(path)
    plt.plot(path[:, 0] + traj.loc[i, 'locx'], path[:, 1] + traj.loc[i, 'locy'], marker = 'o',label=f'Test {i+1} Path',markersize = 1, linewidth = 1)
    plt.scatter(distance_to_start[0] + traj.loc[i, 'locx'], distance_to_start[1] + traj.loc[i, 'locy'], marker='o', color='green',label=f'Test {i+1} Start')
    plt.scatter(end_to_start[0] + traj.loc[i, 'locx'], end_to_start[1] + traj.loc[i, 'locy'], marker='x', color='red', label=f'Test {i+1} End')

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.gca().set_aspect('equal')

plt.title('Agent Paths in Environment')
plt.show()

# Print total rewards for each test
for i, (_, _, total_reward, _) in enumerate(results):
    print(f'Test {i+1} - Total reward: {total_reward}')