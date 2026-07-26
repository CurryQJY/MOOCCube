# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 64 |
| Validation sequence examples | 64 |
| Test sequence examples | 64 |
| Epochs | 1 |
| Last loss | 8.3759 |
| Last RS loss | 6.3861 |
| Last KG loss | 1.9898 |
| KG triples | 5115 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 596 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 1 |
| Best validation score | 0.0180 |
| Stopped early | False |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\_device_parity_cpu\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.021052631578947368,
    "R@10": 0.05263157894736842,
    "R@20": 0.05263157894736842,
    "N@5": 0.008144269625990349,
    "N@10": 0.01797939425487243,
    "N@20": 0.01797939425487243
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
    "R@10": 0.0,
    "R@20": 0.125,
    "N@5": 0.0,
    "N@10": 0.0,
    "N@20": 0.032708340573541946
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
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.