import pandas as pd

def print_metrics(file_path):
    df = pd.read_csv(file_path)
    df = df[df['Period'] >= 3] # Warmup
    hot_count = df['Count_hot'].sum()
    cold_count = df['Count_cold'].sum()
    
    metrics_to_calc = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']
    
    print("--- Cold Metrics ---")
    for base in metrics_to_calc:
        m = f'cold_{base}'
        val = (df[m] * df['Count_cold']).sum() / cold_count if cold_count > 0 else 0
        print(f'{m}: {val:.4f}')
        
    print("--- Hot Metrics ---")
    for base in metrics_to_calc:
        m = f'hot_{base}'
        val = (df[m] * df['Count_hot']).sum() / hot_count if hot_count > 0 else 0
        print(f'{m}: {val:.4f}')

print('\n=== 1. PAM Enhanced ===')
print_metrics('mooc_metrics_pam_final_hin_full.csv')

print('\n=== 2. PAM LLM Enhanced ===')
print_metrics('mooc_metrics_pam_llm_final_hin_full.csv')
