import torch
import numpy as np
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rl_utils.env import MapEnv, dxdy_dict, UpperEnv
from rl_utils.descrete_rl_methods import VAnet, Qnet, Policy


state_dim = 5
action_dim = 4
hidden_dim = 128

# num_tests is the end idx in train10000.csv
num_test = 10000
num_trajs = 500
action_dict = {0:'GSD',1:'GG',2:'TS',3:'TG'}

uppermodel_path = 'upper_model/DQN_20000_eps_inrealmap_129.pth'
qnet = VAnet(state_dim, hidden_dim, action_dim)
qnet.load_state_dict(torch.load(uppermodel_path))
qnet.eval()

lower_config = {
    'state_dim': 12,
    'hidden_dim': 128,
    'action_dim': 8,
    'model_path': 'lower_model/DQN_15000_eps_inrealmap_1110_sota.pth',
}

with open('data/GridModesAdjacentRes.pkl','rb') as f:
    mapdata = pickle.load(f)
traj = pd.read_csv('data/train10000.csv')

upper_env = UpperEnv(mapdata, traj, test_mode=True, testid_start=-1, test_num=num_test,
             use_real_map=True, realmap_row=326, realmap_col=364, lower_model_config=lower_config)

test_results = []
# compute the accuracy of prediction
accuracy_dict = {'GSD':0,'GG':0,'TS':0,'TG':0}
t_upper_dict = {'GSD':[],'GG':[],'TS':[],'TG':[]}
t_lower_dict = {'GSD':[],'GG':[],'TS':[],'TG':[]}
totalamount_dict = {'GSD':0,'GG':0,'TS':0,'TG':0}
# compute the match rate of lower mode for  each mode
matchrated_dict = {'GSD':0,'GG':0,'TS':0,'TG':0}
totalroad_dict = {'GSD':0,'GG':0,'TS':0,'TG':0}

for i in range(num_trajs):
    s = upper_env.reset()
    done = False
    while not done:
        q_values = qnet(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
        # q-max action
        action = torch.argmax(q_values).item()

        # result evaluation
        current_idx = (upper_env.traj_idx + upper_env.step_cnt)
        current_record_idx = traj.loc[current_idx, 'ID']
        x,y = traj.loc[current_idx, 'locx'], traj.loc[current_idx, 'locy']
        print(current_idx,'stepcnt',upper_env.step_cnt,'stepstart',upper_env.traj_idx,current_record_idx,x,y, 'predict',action_dict[action],'true',traj.loc[current_idx,'mode'])
        totalamount_dict[traj.loc[current_idx,'mode']] += 1
        accuracy_dict[traj.loc[current_idx,'mode']] += int(action_dict[action]==traj.loc[current_idx,'mode'])

        test_results.append(int(action_dict[action]==traj.loc[current_idx,'mode']))
        s, reward, done = upper_env.step(action)
        total, match = upper_env.is_match_compute_tuple
        totalroad_dict[action_dict[action]] += total
        matchrated_dict[action_dict[action]] += match

        print(action_dict[action],' mode match rate:', match/(total+0.1),match,total)
        t_upper_dict[traj.loc[current_idx,'mode']].append(upper_env.t_upper)
        t_lower_dict[action_dict[action]].append(upper_env.t_lower)



print('the accuracy of prediction is :', np.average(test_results), uppermodel_path)
print('accuracy for each mode is: ')
for key in accuracy_dict.keys():
    print(key, accuracy_dict[key]/totalamount_dict[key])
print('match rate for each mode is: ')
for key in matchrated_dict.keys():
    print(key, matchrated_dict[key]/totalroad_dict[key])

# plot t_lower and t_upper distribution of each mode in different colors
colors = {'GSD': 'blue', 'GG': 'green', 'TS': 'red', 'TG': 'purple'}
fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", sharey="all", figsize=(10, 8))

def cap_values(data, cap):
    return [min(x, cap) for x in data]

cap_value = 25
for label in t_upper_dict.keys():
    capped_upper = cap_values(t_upper_dict[label], cap_value)
    capped_lower = cap_values(t_lower_dict[label], cap_value)

    ax1.hist(capped_upper, bins=100, alpha=0.5, color=colors[label], linestyle='dashed', label=f'{label} t_upper')
    ax2.hist(capped_lower, bins=100, alpha=0.5, color=colors[label], linestyle='solid', label=f'{label} t_lower')

ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of t_upper')
ax1.legend()

ax2.set_xlabel('t values')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of t_lower')
ax2.legend()

plt.tight_layout()
plt.show()