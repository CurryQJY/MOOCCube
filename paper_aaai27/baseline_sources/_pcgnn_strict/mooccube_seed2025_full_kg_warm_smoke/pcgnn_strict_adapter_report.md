# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 2048 |
| Validation sequence examples | 1024 |
| Test sequence examples | 1024 |
| Epochs | 5 |
| Last loss | 7.1709 |
| Last RS loss | 5.2043 |
| Last KG loss | 1.9665 |
| KG triples | 5115 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 596 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 1 |
| Best validation score | 0.0000 |
| Stopped early | False |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2025_full_kg_warm_smoke\checkpoints\best_model.pt |

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
    "R@5": 0.001724137931034483,
    "R@10": 0.020689655172413793,
    "R@20": 0.043521421107628,
    "N@5": 0.0006669875986802441,
    "N@10": 0.00715644482243786,
    "N@20": 0.01272524841018688
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
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.