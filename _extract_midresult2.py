"""Extract intermediate USIM training results from train.log (UTF-16 encoded)."""
from pathlib import Path
import re
import math

text = Path('results_mooccubex/train.log').read_bytes().decode('utf-16', errors='ignore')

# Find each Period block: matches "Period N (current=X, accumulated=Y)"
period_blocks = list(re.finditer(
    r">>>\s*Period\s+(\d+)\s+\(current=(\d+),\s+accumulated=(\d+)\)\s*<<<"
    r"(.*?)(?=>>>\s*Period\s+\d+|\Z)",
    text, re.DOTALL
))

# Patterns inside each block
sampled_pat = re.compile(
    r"Sampled Cold=([\d.]+)\s+Hot=([\d.]+)\s+\|\s+Full Cold=([\d.]+)\s+Hot=([\d.]+)"
)
es_pat = re.compile(
    r"\[EARLYSTOP\]\s+Epoch\s+(\d+):\s+Full Cold N@10=([\d.]+),\s+Full Cold R@10=([\d.]+),\s+Full Hot R@10=([\d.]+)\s+\|\s+(update|wait\([\d/]+\))"
)
restore_pat = re.compile(
    r"\[EARLYSTOP\]\s+Restore best epoch=(\d+)\s+\(Full Cold N@10=([\d.]+),\s+R@10=([\d.]+),\s+Full Hot R@10=([\d.]+)\)"
)

print(f"Found {len(period_blocks)} period blocks\n")

# Header
print(f"{'Period':<7}{'NewSamp':<10}{'CumulSmp':<11}"
      f"{'EvSmpC@10':<11}{'EvSmpH@10':<11}"
      f"{'EvFullC@10':<12}{'EvFullH@10':<12}"
      f"{'BestN@10':<10}{'BestR@10':<10}{'BestHR@10':<11}{'Status':<10}")
print("-" * 132)

results = []
for m in period_blocks:
    period_num = int(m.group(1))
    new_samp = int(m.group(2))
    cumul = int(m.group(3))
    block = m.group(4)

    smp = sampled_pat.search(block)
    if smp:
        sc, sh, fc, fh = (float(smp.group(i)) for i in range(1, 5))
    else:
        sc = sh = fc = fh = float('nan')

    # Best epoch (use restore if exists, otherwise last update)
    best_n = best_r = best_h = float('nan')
    rest = restore_pat.search(block)
    status = "running"
    if rest:
        best_n = float(rest.group(2))
        best_r = float(rest.group(3))
        best_h = float(rest.group(4))
        status = "done"
    else:
        for em in es_pat.finditer(block):
            if em.group(5) == "update":
                best_n = float(em.group(2))
                best_r = float(em.group(3))
                best_h = float(em.group(4))
        if any(es_pat.finditer(block)):
            status = "in_progress"

    results.append((period_num, new_samp, cumul, sc, sh, fc, fh, best_n, best_r, best_h, status))

for r in results:
    p, ns, cm, sc, sh, fc, fh, bn, br, bh, st = r
    sc_s = f"{sc:.4f}" if not math.isnan(sc) else "—"
    sh_s = f"{sh:.4f}" if not math.isnan(sh) else "—"
    fc_s = f"{fc:.4f}" if not math.isnan(fc) else "—"
    fh_s = f"{fh:.4f}" if not math.isnan(fh) else "—"
    bn_s = f"{bn:.4f}" if not math.isnan(bn) else "—"
    br_s = f"{br:.4f}" if not math.isnan(br) else "—"
    bh_s = f"{bh:.4f}" if not math.isnan(bh) else "—"
    print(f"{p:<7}{ns:<10}{cm:<11}{sc_s:<11}{sh_s:<11}{fc_s:<12}{fh_s:<12}{bn_s:<10}{br_s:<10}{bh_s:<11}{st:<10}")

# Compute weighted-average over evaluation periods only (exclude warmup)
print(f"\n[Weighted by per-period new sample count, excluding warmup periods]")
def w_avg(values, weights):
    valid = [(v, w) for v, w in zip(values, weights) if not math.isnan(v) and w > 0]
    if not valid: return float('nan')
    s = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / max(s, 1)

ws = [r[1] for r in results]
print(f"  EvalSampled Cold @10: {w_avg([r[3] for r in results], ws):.4f}")
print(f"  EvalSampled Hot  @10: {w_avg([r[4] for r in results], ws):.4f}")
print(f"  EvalFull    Cold @10: {w_avg([r[5] for r in results], ws):.4f}")
print(f"  EvalFull    Hot  @10: {w_avg([r[6] for r in results], ws):.4f}")
print()
print(f"  Train-best Full Cold N@10: {w_avg([r[7] for r in results], ws):.4f}")
print(f"  Train-best Full Cold R@10: {w_avg([r[8] for r in results], ws):.4f}")
print(f"  Train-best Full Hot  R@10: {w_avg([r[9] for r in results], ws):.4f}")

# Latest period info
if results:
    last = results[-1]
    print(f"\n[Latest period]: P{last[0]} | new={last[1]} | cumul={last[2]} | status={last[10]}")
