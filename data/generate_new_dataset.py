"""
从原始数据集csv中分别为上层和下层生成新的数据集
data_raw: col = [id, locx, locy, time, mode]

data_lower: 对相邻的同id的进行差分,得到每一条od数据 col = [id, locx_o, locy_o, locx_d, locy_d, mode(mode与o一致)
data_upper: 对data_lower 按id进行聚合

"""
import numpy as np
import pandas as pd
data_raw = pd.read_csv('realworldTraj15000.csv')
# Generate data_lower
data_lower = []
for _, group in data_raw.groupby('ID'):
    for i in range(len(group) - 1):
        row_o = group.iloc[i]
        row_d = group.iloc[i + 1]
        distance = np.sqrt((row_d['locx'] - row_o['locx'])**2 + (row_d['locy'] - row_o['locy'])**2)

        data_lower.append({
            'ID': row_o['ID'],
            'locx_o': row_o['locx'],
            'locy_o': row_o['locy'],
            'locx_d': row_d['locx'],
            'locy_d': row_d['locy'],
            'mode': row_o['mode'],
            'time': row_d['time'] - row_o['time'],
            'distance': distance
        })

data_lower = pd.DataFrame(data_lower)
data_lower.to_csv('data_train_upper_250624.csv')
# # Sort by distance in ascending order
# data_lower = data_lower.sort_values(by='distance', ascending=False)
# data_lower = data_lower[data_lower['distance'] > 0]  # Remove rows with zero distance

# # Save the new dataset
# data_lower.to_csv('data_lower_train_sort_ascend_0.csv', index=True)
#
# # random shuffle the data_lower
# data_lower2 = data_lower.sample(frac=1).reset_index(drop=True)
# data_lower2.to_csv('data_lower_train_random.csv', index=False)
#
# # merge
# data_lower3 = pd.concat([data_lower, data_lower2], ignore_index=True)
# data_lower3.to_csv('data_lower_train_mixed_0.csv', index=False)