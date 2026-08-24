import os, re
files = ['cgrc_static_hin.py', 'drop_static_hin.py', 'gar_static_hin.py', 'marec_static_hin.py']
print(f"{'file':<24} {'epochs':<8} {'mini_batch':<11} {'uses_split_df':<14} {'has_fair_ver':<13}")
for f in files:
    if not os.path.exists(f):
        print(f'{f}: MISSING')
        continue
    fair = f.replace('.py', '_fair.py')
    has_fair = os.path.exists(fair)
    with open(f, encoding='utf-8') as fp:
        src = fp.read()
    m = re.search(r'STATIC_EPOCHS[^,]*"\s*,\s*"(\d+)"', src)
    epochs = m.group(1) if m else 'N/A'
    has_minibatch = bool(re.search(r'for\s+start\s+in\s+range|for\s+batch\s+in|enumerate\(.*loader', src))
    has_split_env = 'USIM_STATIC_SPLIT_DIR' in src or 'static_split_df' in src
    print(f'{f:<24} {epochs:<8} {str(has_minibatch):<11} {str(has_split_env):<14} {str(has_fair):<13}')
