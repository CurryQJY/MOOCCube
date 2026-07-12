# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 233334 |
| Validation sequence examples | 18909 |
| Test sequence examples | 37848 |
| Epochs | 20 |
| Last loss | 6.1861 |
| Last RS loss | 6.1703 |
| Last KG loss | 0.0158 |
| KG triples | 41414 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 6967 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 11 |
| Best validation score | 0.0127 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\coco_seed2025_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.015267778081761923,
    "R@10": 0.026196984393490264,
    "R@20": 0.04841224864386492,
    "N@5": 0.00924295415281987,
    "N@10": 0.012723364035913407,
    "N@20": 0.018264796738286616
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 409,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 18909,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.014980261347497269,
    "R@10": 0.027732647571148974,
    "R@20": 0.04617360950046342,
    "N@5": 0.009030346723152755,
    "N@10": 0.013099216852171016,
    "N@20": 0.01771123188702089
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 820,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 37848,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.