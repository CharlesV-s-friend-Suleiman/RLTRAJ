"""
This file aim to plot the distribution of distance between 2 records of 4 different mode in 1 figur

"""

import matplotlib.pyplot as plt
import pandas as pd

raw_df = pd.read_csv("data/train10000.csv")
colors = {'GSD': 'red', 'GG': 'yellow', 'TS': 'green', 'TG': 'blue'}
fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", sharey="all", figsize=(10, 8))

