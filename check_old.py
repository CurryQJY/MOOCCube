import pandas as pd
import glob
print('--- Start ---')
for f in glob.glob('*.csv'):
    try:
        df = pd.read_csv(f)
        if 'hot_R@10' in df.columns and 'Count_hot' in df.columns:
            df = df[df['Period'] >= 3]
            hc = df['Count_hot'].sum()
            cc = df['Count_cold'].sum()
            if hc == 0 or cc == 0: continue
            hr10 = (df['hot_R@10']*df['Count_hot']).sum() / hc
            cr10 = (df['cold_R@10']*df['Count_cold']).sum() / cc
            print(f'FILE: {f}')
            print(f'     Hot R@10: {hr10:.4f}')
            print(f'     Cold R@10: {cr10:.4f}')
        elif 'R@10' in df.columns and 'Count' in df.columns:
            df = df[df['Period'] >= 3]
            tc = df['Count'].sum()
            r10 = (df['R@10']*df['Count']).sum() / tc
            print(f'FILE (overall): {f}')
            print(f'     Overall R@10: {r10:.4f}')
    except Exception as e:
        pass
print('--- End ---')
