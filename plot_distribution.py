import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the data
df = pd.read_csv('data/artificial_traj_mixed_train.csv')

# Define colors for each mode
mode_colors = {
    'GG': 'red',
    'GSD': 'blue',
    'TG': 'green',
    'TS': 'purple'
}

# Initialize the plot with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# Define a mapping from mode to subplot index
mode_to_ax = {
    'GG': axs[0, 0],
    'GSD': axs[0, 1],
    'TG': axs[1, 0],
    'TS': axs[1, 1]
}

# Dictionary to count occurrences of (dx, dy) pairs for each mode
count_dict = {mode: defaultdict(int) for mode in mode_colors.keys()}

# Iterate through the dataframe to count occurrences
for i in range(350):
    if df.loc[i, 'ID'] == df.loc[i + 1, 'ID']:
        dx = df.loc[i + 1, 'locx'] - df.loc[i, 'locx']
        dy = df.loc[i + 1, 'locy'] - df.loc[i, 'locy']
        mode = df.loc[i, 'mode']
        count_dict[mode][(dx, dy)] += 1

# Plot the points with sizes based on occurrences
for mode, ax in mode_to_ax.items():
    for (dx, dy), count in count_dict[mode].items():
        ax.scatter(dx, dy, color=mode_colors[mode], s=count * 10)  # Adjust size multiplier as needed

# Set plot labels and titles for each subplot
for mode, ax in mode_to_ax.items():
    ax.set_xlabel('Delta X')
    ax.set_ylabel('Delta Y')
    ax.set_title(f'Coordinate Difference for Mode {mode}')
    ax.grid(True)

# Set the main title
fig.suptitle('Coordinate Difference Distribution for Different Travel Modes')

# Show plot
plt.show()