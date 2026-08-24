# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 64 |
| Validation sequence examples | 64 |
| Test sequence examples | 64 |
| Epochs | 1 |
| Last loss | 6.5467 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 1 |
| Best validation score | 0.0000 |
| Stopped early | False |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2025_full_warning_check\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.0,
    "R@10": 0.0,
    "R@20": 0.0,
    "N@5": 0.0,
    "N@10": 0.0,
    "N@20": 0.0
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 19,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 64,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.0,
    "R@10": 0.041666666666666664,
    "R@20": 0.041666666666666664,
    "N@5": 0.0,
    "N@10": 0.013144369866072031,
    "N@20": 0.013144369866072031
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 24,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 64,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.