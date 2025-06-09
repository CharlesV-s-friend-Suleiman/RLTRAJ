import pandas as pd
import numpy as np
from scipy.stats import norm

import pickle
dataset_name = 'datainrealworld'
# Assuming df is your DataFrame
df = pd.read_csv('data/'+dataset_name +'.csv')

# Compute the differences
df['delta_x'] = df.groupby('ID')['locx'].diff()
df['delta_y'] = df.groupby('ID')['locy'].diff()
df['delta_t'] = df.groupby('ID')['time'].diff()

df = df[abs(df['delta_x'])+abs(df['delta_y']) != 0]


distance_method = 'euclidean'  # or 'euclidean'
# Calculate the velocity using manhattan distance
if distance_method == 'manhattan':
    df['velocity'] = (np.abs(df['delta_x']) + np.abs(df['delta_y'])) / df['delta_t'] * 60
# Calculate the velocity using euclidean distance
elif distance_method == 'euclidean':
    df['velocity'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2) / df['delta_t'] * 60

# Output rows where delta_t is zero
delta_t_zero_rows = df[df['velocity']<10]
print(delta_t_zero_rows)


# Drop NaN values which are the result of the diff() operation
df = df.dropna(subset=['velocity'])

# Ensure there are no infinite values
df = df[np.isfinite(df['velocity'])]

# Separate the velocities by mode
modes = df['mode'].unique()
processed_data = {}

for mode in modes:
    mode_data = df[df['mode'] == mode]['velocity'].to_numpy()
    threshold = np.percentile(mode_data, 5)
    filtered_data = mode_data[mode_data > threshold]
    mean = np.mean(filtered_data)
    std = np.std(filtered_data)
    processed_data[mode] = {
        'mean': mean,
        'std': std
    }

# Print the results
for mode, data in processed_data.items():
    print(f'Mode: {mode}')
    print(f'Mean: {data["mean"]}')
    print(f'Std: {data["std"]}')

    print()

print(processed_data)

with open('data/vdistribution'+dataset_name +'.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

