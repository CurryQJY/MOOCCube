# GAR ColdRec Source-Default Three-Seed Report

- Seeds: 2025, 2026, 2027
- Runs: 3
- ColdRec commit: `18efd24`
- Epoch ceiling: 500 (early stopping enabled)
- Protocol: strict full-catalog course-cold, train-only history

## Per-Seed Cold Results

| Seed | Best epoch | Interaction R@10 | Interaction N@10 | Course R@10 | Course N@10 |
|---:|---:|---:|---:|---:|---:|
| 2025 | 9 | 0.343297 | 0.197336 | 0.257811 | 0.126199 |
| 2026 | 7 | 0.541806 | 0.355564 | 0.240918 | 0.126109 |
| 2027 | 17 | 0.506246 | 0.324979 | 0.280325 | 0.145175 |

## Three-Seed Cold Summary

| Metric family | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Interaction macro | 0.350847 +/- 0.099355 | 0.463783 +/- 0.105848 | 0.256050 +/- 0.080686 | 0.292626 +/- 0.083929 |
| Course macro | 0.151950 +/- 0.005468 | 0.259684 +/- 0.019770 | 0.098119 +/- 0.007526 | 0.132494 +/- 0.010982 |
