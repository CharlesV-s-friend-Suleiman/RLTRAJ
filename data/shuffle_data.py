import pandas as pd
import numpy as np
df = pd.read_csv('artificial_traj_mixed.csv')

# shuffle the data by ID
# v1, shuffle by id aim to fix the problem: model overfit to 4 distributions temporally.


# 按照 group_column 分组
groups = df.groupby('ID')

# 获取分组键
group_keys = list(groups.groups.keys())

# 打乱组与组之间的顺序
np.random.shuffle(group_keys)

# 创建一个新的 DataFrame，按打乱后的顺序拼接组
new_df = pd.concat([groups.get_group(key) for key in group_keys])

# 重置索引
new_df = new_df.reset_index(drop=True)

new_df.to_csv('artificial_traj_mixed_shuffled.csv')