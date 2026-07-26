"""Aggregate and print all *_result.json files into a unified table."""
import json, glob
import pandas as pd

metrics = ['R@5', 'R@10', 'R@20', 'N@5', 'N@10', 'N@20']

def load_all_results():
    """Load all result jsons, normalizing two formats into unified structure."""
    files = sorted(glob.glob('*_result*.json'))
    results = {}

    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        if isinstance(data, list):
            data = data[0]

        name = f.replace('_result.json', '')

        # Check format: nested dict (sample_cold: {R@5: ...}) vs flat (samp_cold_R@5: ...)
        if 'sample_cold' in data and isinstance(data.get('sample_cold'), dict):
            # Already in nested format (popularity, bpr, lightgcn, hhcor, light_path, pam)
            results[name] = data
        else:
            # Flat format (sasrec, drop, gar, lightgcl)
            entry = {'sample_cold': {}, 'sample_hot': {}, 'full_cold': {}, 'full_hot': {}}
            for m in metrics:
                for prefix, key in [('samp_cold_', 'sample_cold'), ('samp_hot_', 'sample_hot'),
                                    ('full_cold_', 'full_cold'), ('full_hot_', 'full_hot')]:
                    flat_key = prefix + m
                    if flat_key in data:
                        entry[key][m] = data[flat_key]
            results[name] = entry

    # Add USIM (our method) from final_report CSV
    usim_csv = 'final_report_usim_feedback_fast3_content_delta.csv'
    try:
        df = pd.read_csv(usim_csv)
        entry = {'sample_cold': {}, 'sample_hot': {}, 'full_cold': {}, 'full_hot': {}}
        for _, row in df.iterrows():
            m = row['metric']
            if not pd.isna(row.get('full_cold')):
                entry['full_cold'][m] = row['full_cold']
            if not pd.isna(row.get('full_hot')):
                entry['full_hot'][m] = row['full_hot']
            if not pd.isna(row.get('sampled_cold')):
                entry['sample_cold'][m] = row['sampled_cold']
            if not pd.isna(row.get('sampled_hot')):
                entry['sample_hot'][m] = row['sampled_hot']
        results['usim_full (Ours)'] = entry
    except Exception:
        pass

    return results


def print_table(title, results, key):
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    header = f"{'Method':<28}" + "".join(f"{m:<10}" for m in metrics)
    print(header)
    print("-" * 88)

    # Sort: static methods first, then stream, then ours last
    def sort_key(name):
        if 'Ours' in name:
            return (2, name)
        elif 'static' in name:
            return (0, name)
        else:
            return (1, name)

    for name in sorted(results.keys(), key=sort_key):
        d = results[name]
        block = d.get(key, {})
        if not block:
            continue
        row = f"{name:<28}"
        for m in metrics:
            val = block.get(m, 0.0)
            row += f"{val:<10.4f}"
        print(row)


def main():
    results = load_all_results()

    print("\n" + "#" * 95)
    print("#  COMPLETE RESULTS TABLE — MOOCCube Baselines + Ours")
    print("#" * 95)

    print_table("Full Ranking — Cold Start Users", results, "full_cold")
    print_table("Full Ranking — Hot Users", results, "full_hot")
    print_table("Sampled (1+200) — Cold Start Users", results, "sample_cold")
    print_table("Sampled (1+200) — Hot Users", results, "sample_hot")


if __name__ == "__main__":
    main()
