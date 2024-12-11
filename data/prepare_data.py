import pandas as pd

df = pd.read_csv('test_traj_lower.csv')

def modetoint(mode):
    if mode == 'static':
        return 0
    elif mode == 'TG':
        return 1
    elif mode == 'GSD':
        return 2
    elif mode == 'GG':
        return 3
    elif mode == 'TS':
        return 6

df['mode'] = df['mode'].apply(modetoint)
df.to_csv('test_traj_lower.csv', index=False)