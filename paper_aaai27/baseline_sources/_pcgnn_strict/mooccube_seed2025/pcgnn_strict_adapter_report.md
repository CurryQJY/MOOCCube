# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 2048 |
| Validation sequence examples | 1024 |
| Test sequence examples | 1024 |
| Epochs | 1 |
| Last loss | 6.5007 |

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
  "count_full_cold_item_macro": 33,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 1024,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.006896551724137932,
    "R@10": 0.006896551724137932,
    "R@20": 0.01839080459770115,
    "N@5": 0.003509595037262066,
    "N@10": 0.003509595037262066,
    "N@20": 0.006326337231579992
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 58,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 1024,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.