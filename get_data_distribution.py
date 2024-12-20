import pandas as pd
import numpy as np
from scipy.stats import norm

import pickle

# Assuming df is your DataFrame
df = pd.read_csv('data/train10000.csv')

# Compute the differences
df['delta_x'] = df.groupby('ID')['locx'].diff()
df['delta_y'] = df.groupby('ID')['locy'].diff()
df['delta_t'] = df.groupby('ID')['time'].diff()

distance_method = 'euclidean'  # or 'euclidean'
# Calculate the velocity using manhattan distance
if distance_method == 'manhattan':
    df['velocity'] = (np.abs(df['delta_x']) + np.abs(df['delta_y'])) / df['delta_t'] * 60
# Calculate the velocity using euclidean distance
elif distance_method == 'euclidean':
    df['velocity'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2) / df['delta_t'] * 60

# Drop NaN values which are the result of the diff() operation
df = df.dropna(subset=['velocity'])

# Separate the velocities by mode
modes = df['mode'].unique()
processed_data = {mode: df[df['mode'] == mode]['velocity'].to_numpy() for mode in modes}

for mode in processed_data:
    print(f'Mode: {mode}')
    print(f'Mean: {np.mean(processed_data[mode])}')
    print(f'Std: {np.std(processed_data[mode])}')
    print(f'Min: {np.min(processed_data[mode])}')
    print(f'Max: {np.max(processed_data[mode])}')
    print()

with open('data/vdistribution.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

# Given value x
x = 200

# Calculate the confidence levels
confidence_levels = {}
for mode, data in processed_data.items():
    mean = np.mean(data)
    std = np.std(data)
    z_score = (x - mean) / std
    confidence_level = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    confidence_levels[mode] = confidence_level

# Find the maximum confidence level and corresponding mode
max_confidence_level = max(confidence_levels.values())
max_confidence_mode = max(confidence_levels, key=confidence_levels.get)

# Print the maximum confidence level and corresponding mode
print(f"Maximum confidence level for x = {x} is {max_confidence_level} for mode {max_confidence_mode}")