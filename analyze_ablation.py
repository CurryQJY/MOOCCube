import csv, os

OUT_DIR = 'outputs/usim_feedback_fast3_course_ablation'
SRC_CSV = os.path.join(OUT_DIR, 'summary_course_ablation.csv')

rows = []
with open(SRC_CSV, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

metrics = [
    ('sampled_cold_r5','R@5'), ('sampled_cold_r10','R@10'), ('sampled_cold_r20','R@20'),
    ('sampled_cold_n5','N@5'), ('sampled_cold_n10','N@10'), ('sampled_cold_n20','N@20'),
    ('sampled_hot_r5','R@5'), ('sampled_hot_r10','R@10'), ('sampled_hot_r20','R@20'),
    ('sampled_hot_n5','N@5'), ('sampled_hot_n10','N@10'), ('sampled_hot_n20','N@20'),
    ('full_cold_r5','R@5'), ('full_cold_r10','R@10'), ('full_cold_r20','R@20'),
    ('full_cold_n5','N@5'), ('full_cold_n10','N@10'), ('full_cold_n20','N@20'),
    ('full_hot_r5','R@5'), ('full_hot_r10','R@10'), ('full_hot_r20','R@20'),
    ('full_hot_n5','N@5'), ('full_hot_n10','N@10'), ('full_hot_n20','N@20'),
]

base = rows[0]  # fast3_ref

groups = [
    ('Sampled Cold-Start', [m for m in metrics if m[0].startswith('sampled_cold')]),
    ('Sampled Hot',        [m for m in metrics if m[0].startswith('sampled_hot')]),
    ('Full Cold-Start',    [m for m in metrics if m[0].startswith('full_cold')]),
    ('Full Hot',           [m for m in metrics if m[0].startswith('full_hot')]),
]

exp_labels = {
    'fast3_ref':              'FAST3 (baseline)',
    'plus_prereq_aux':        '+ Prereq Aux Loss',
    'plus_prereq_reward':     '+ Prereq Reward',
    'plus_concept_reward':    '+ Concept Reward',
    'plus_difficulty_reward':  '+ Difficulty Reward',
    'plus_redundant_penalty': '+ Redundant Penalty',
    'plus_course_reward_all': '+ All Rewards (no sampling)',
    'plus_course_sampling':   '+ Course Sampling',
    'plus_course_rerank':     '+ Course Rerank',
    'plus_all_course':        '+ All Course (no rerank)',
    'plus_all_course_rerank': '+ All Course + Rerank',
}

# ── helper: find best per column (excluding baseline) ──
def find_best(gmetrics):
    best = {}
    for key, _ in gmetrics:
        bv, bn = -1, ''
        for r in rows[1:]:
            v = float(r[key])
            if v > bv:
                bv, bn = v, r['experiment']
        best[key] = bn
    return best

# ── 1. absolute values CSV ──
abs_csv = os.path.join(OUT_DIR, 'ablation_absolute.csv')
with open(abs_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    for gname, gmetrics in groups:
        w.writerow([gname] + [m[1] for m in gmetrics])
        for r in rows:
            label = exp_labels.get(r['experiment'], r['experiment'])
            w.writerow([label] + [f"{float(r[k]):.4f}" for k, _ in gmetrics])
        w.writerow([])
print(f"[saved] {abs_csv}")

# ── 2. relative change CSV ──
rel_csv = os.path.join(OUT_DIR, 'ablation_relative.csv')
with open(rel_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    for gname, gmetrics in groups:
        w.writerow([gname] + [m[1] for m in gmetrics])
        for r in rows:
            label = exp_labels.get(r['experiment'], r['experiment'])
            vals = []
            for key, _ in gmetrics:
                v, b = float(r[key]), float(base[key])
                delta = (v - b) / b * 100 if b != 0 else 0
                vals.append(f"{delta:+.2f}%")
            w.writerow([label] + vals)
        w.writerow([])
print(f"[saved] {rel_csv}")

# ── 3. Markdown report ──
md_path = os.path.join(OUT_DIR, 'ablation_report.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('# Course-Side Ablation Study Results\n\n')
    f.write(f'- **Baseline**: `fast3_ref` (all course signals disabled)\n')
    f.write(f'- **Cold-start users**: {rows[0]["sampled_cold_count"]}\n')
    f.write(f'- **Hot users**: {rows[0]["sampled_hot_count"]}\n\n')

    # absolute tables
    f.write('## Absolute Performance\n\n')
    for gname, gmetrics in groups:
        f.write(f'### {gname}\n\n')
        hdr = '| Experiment | ' + ' | '.join(m[1] for m in gmetrics) + ' |\n'
        sep = '|---|' + '|'.join(['---:'] * len(gmetrics)) + '|\n'
        f.write(hdr)
        f.write(sep)
        best = find_best(gmetrics)
        for r in rows:
            label = exp_labels.get(r['experiment'], r['experiment'])
            cells = []
            for key, _ in gmetrics:
                val = f"{float(r[key]):.4f}"
                if r['experiment'] != 'fast3_ref' and best.get(key) == r['experiment']:
                    val = f"**{val}**"
                cells.append(val)
            f.write(f'| {label} | ' + ' | '.join(cells) + ' |\n')
        f.write('\n')

    # relative tables
    f.write('## Relative Change vs Baseline (%)\n\n')
    for gname, gmetrics in groups:
        f.write(f'### {gname}\n\n')
        hdr = '| Experiment | ' + ' | '.join(m[1] for m in gmetrics) + ' |\n'
        sep = '|---|' + '|'.join(['---:'] * len(gmetrics)) + '|\n'
        f.write(hdr)
        f.write(sep)
        for r in rows[1:]:
            label = exp_labels.get(r['experiment'], r['experiment'])
            cells = []
            for key, _ in gmetrics:
                v, b = float(r[key]), float(base[key])
                delta = (v - b) / b * 100 if b != 0 else 0
                cells.append(f"{delta:+.2f}%")
            f.write(f'| {label} | ' + ' | '.join(cells) + ' |\n')
        f.write('\n')

    # best per metric
    f.write('## Best Experiment per Metric\n\n')
    f.write('| Metric Group | Metric | Best Experiment | Value | Baseline | Change |\n')
    f.write('|---|---|---|---:|---:|---:|\n')
    for gname, gmetrics in groups:
        for key, label in gmetrics:
            bv, bn = -1, ''
            for r in rows:
                v = float(r[key])
                if v > bv:
                    bv, bn = v, r['experiment']
            bval = float(base[key])
            delta = (bv - bval) / bval * 100
            bl = exp_labels.get(bn, bn)
            f.write(f'| {gname} | {label} | {bl} | {bv:.4f} | {bval:.4f} | {delta:+.2f}% |\n')
    f.write('\n')

    # key findings
    f.write('## Key Findings\n\n')
    f.write('1. **Concept Reward** is the single best component for cold-start users '
            '(SC-R@5 +5.51%, FC-R@20 +5.98%).\n')
    f.write('2. **Prereq Reward** is the single best for hot users '
            '(SH-N@5 +7.68%, FH-R@5 +22.93%, FH-N@5 +23.57%).\n')
    f.write('3. **Difficulty Reward** also significantly boosts hot users '
            '(FH-R@5 +15.27%) but hurts cold-start R@20 (-4.26%).\n')
    f.write('4. **All Course combined** (`plus_all_course`) achieves the best aggregate '
            'cold-start performance (R@10 +4.16%, N@10 +3.75%) but hurts hot users.\n')
    f.write('5. **Prereq Aux Loss** alone is slightly harmful; **Course Rerank** has near-zero impact.\n')
    f.write('6. There is a clear **cold-hot trade-off**: combining all signals helps cold-start '
            'but degrades hot-user metrics.\n')

print(f"[saved] {md_path}")
print("\nDone. Generated 3 files:")
