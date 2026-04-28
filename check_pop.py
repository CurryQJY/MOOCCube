import pandas as pd
import numpy as np

df = pd.read_pickle("processed_data_hin/stream_data.pkl")
df['i_idx'] = df['i_idx'] + 1

df = df.sample(frac=1.0, random_state=2025).reset_index(drop=True)
n = len(df)
train_df = df.iloc[:int(n*0.8)].copy()
val_df = df.iloc[int(n*0.8):int(n*0.9)].copy()
test_df = df.iloc[int(n*0.9):].copy()

i_counts = train_df['i_idx'].value_counts().reset_index()
i_counts.columns = ['i_idx', 'pop']
train_df = pd.merge(train_df, i_counts, on='i_idx', how='left')
val_df = pd.merge(val_df, i_counts, on='i_idx', how='left')
test_df = pd.merge(test_df, i_counts, on='i_idx', how='left')

val_df['pop'] = val_df['pop'].fillna(0).astype(int)
test_df['pop'] = test_df['pop'].fillna(0).astype(int)

# Check pop < 5 in test_df
print(f"Test size: {len(test_df)}")
print(f"Cold items in test_df (pop < 5): {(test_df['pop'] < 5).sum()}")
print(f"Hot items in test_df (pop >= 5): {(test_df['pop'] >= 5).sum()}")
print(f"Distribution of pop in test_df:\n{test_df['pop'].value_counts().head(10)}")

