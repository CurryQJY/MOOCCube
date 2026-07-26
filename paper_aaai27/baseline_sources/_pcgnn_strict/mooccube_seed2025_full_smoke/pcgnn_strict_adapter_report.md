# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 512 |
| Validation sequence examples | 256 |
| Test sequence examples | 256 |
| Epochs | 2 |
| Last loss | 6.5150 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 1 |
| Best validation score | 0.0064 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2025_full_smoke\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.0020242914979757085,
    "R@10": 0.020242914979757085,
    "R@20": 0.038461538461538464,
    "N@5": 0.0007831028486529183,
    "N@10": 0.006425185963719737,
    "N@20": 0.011217225174076199
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 26,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 256,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.0,
    "R@10": 0.003968253968253968,
    "R@20": 0.057539682539682536,
    "N@5": 0.0,
    "N@10": 0.0011708230594878355,
    "N@20": 0.014894178005273775
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 42,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 256,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.