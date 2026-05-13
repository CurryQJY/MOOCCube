import pandas as pd

def print_metrics(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Period' in df.columns:
            df = df[df['Period'] >= 3] # Warmup
        hot_count = df['Count_hot'].sum() if 'Count_hot' in df.columns else 1
        cold_count = df['Count_cold'].sum() if 'Count_cold' in df.columns else 1
        
        metrics_to_calc = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
        
        print("--- Cold Metrics ---")
        for base in metrics_to_calc:
            m = f'cold_{base}'
            if m in df.columns and 'Count_cold' in df.columns:
                val = (df[m] * df['Count_cold']).sum() / cold_count if cold_count > 0 else 0
                print(f'{m}: {val:.4f}')
            else:
                print(f'{m}: N/A')
            
        print("--- Hot Metrics ---")
        for base in metrics_to_calc:
            m = f'hot_{base}'
            if m in df.columns and 'Count_hot' in df.columns:
                val = (df[m] * df['Count_hot']).sum() / hot_count if hot_count > 0 else 0
                print(f'{m}: {val:.4f}')
            else:
                print(f'{m}: N/A')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

print('\n=== PAM Enhanced (基准上限) ===')
print_metrics('mooc_metrics_pam_final.csv')

print('\n=== 原始带 MAML 的 USIM系列 ===')
print_metrics('metrics_rl_usim.csv')

print('\n=== Pure RL-USIM (无LLM) ===')
print_metrics('mooc_metrics_pure_usim.csv')

print('\n=== Pure RL-USIM + LLM ===')
print_metrics('mooc_metrics_final.csv')
