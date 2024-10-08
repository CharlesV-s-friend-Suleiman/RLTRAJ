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
    state = env.reset()
    done = False
    total_reward = 0
    start = state[:2]
    end = state[2:]
    path = []

    while not done:
        action = qnet(torch.tensor(state, dtype=torch.float32)).argmax().item()
        state, reward, done = env.step(action)
        total_reward += reward
        path.append(state[:2])

    results.append((start, end, total_reward, path))

# Plot all paths on the same graph
for i, (start, end, total_reward, path) in enumerate(results):
    path = np.array(path)
    plt.scatter(start[0], start[1], color='green', s=100, label=f'Start {i+1}')
    plt.scatter(end[0], end[1], color='red', s=100, label=f'End {i+1}')
    plt.plot(path[:, 0], path[:, 1], marker='o', markersize=2, label=f'Path {i+1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Agent Paths in Environment')
plt.show()

# Print total rewards for each test
for i, (_, _, total_reward, _) in enumerate(results):
    print(f'Test {i+1} - Total reward: {total_reward}')