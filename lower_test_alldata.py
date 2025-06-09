import torch
import numpy as np
import pickle
import pandas as pd

from rl_utils.env import MapEnv, dxdy_dict
from rl_utils.policy_based_rl_methods import Policy

map_row = 529
map_col = 564
state_dim = 12
hidden_dim = 128
action_dim = 8

actor_net = Policy(state_dim, hidden_dim, action_dim)
actor_net.load_state_dict(torch.load('lower_model/SAC_30000_eps_inrealmap_320.pth'))
actor_net.eval()

with open('data/GridModesAdjacentRealworld.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/datainrealworld.csv')

env = MapEnv(mapdata, traj, test_mode=True, testid_start=0, test_num=5000,
             use_real_map=True, realmap_row=map_row, realmap_col=map_col)

all_trajs = []
results = []
match_rates = []
num_finish = 0
for i in range(5000):
    state = env.reset()
    done = False
    total_reward = 0
    distance_to_start = state[:2] # always (0, 0)
    end_to_start = state[2:]
    path = []
    actual_pos = []

    step_cnt = 0
    actions = []
    state_set = set()
    state_set.add(tuple(state[:2]))
    while not done:
        step_cnt += 1
        action = int(actor_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax())
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
    # caculate amtch rate
    for j, coord in enumerate(actual_pos[1:]):
        x,y = int(coord[0]), int(coord[1])
        if env.mapdata[env.mode][x][y] == 1:
            match_step += 1
        total_step += 1
    match_rate = match_step/(total_step+0.1) if step_cnt<=env.max_step else 0
    match_rates.append(match_rate)

    results.append((distance_to_start, end_to_start, total_reward, path, step_cnt, actions))
    # Plot the path and the start and end points
    path = np.array(path)
    if step_cnt<env.max_step:
        num_finish += 1
    else:
        print('not finished traj id:',i)




print('finish rate:',num_finish/len(results), 'TOTAL:',len(results),'FINISH:',num_finish)
print('average match rate:',np.mean(match_rates))

# to test the match rate

