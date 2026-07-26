# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 1339076 |
| Validation sequence examples | 109636 |
| Test sequence examples | 219270 |
| Epochs | 20 |
| Last loss | 3.4372 |
| Last RS loss | 3.4362 |
| Last KG loss | 0.0010 |
| KG triples | 4532 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 616 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 9 |
| Best validation score | 0.0335 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\junyi_seed2026_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.03525461990584557,
    "R@10": 0.07698845933778758,
    "R@20": 0.15193256034899993,
    "N@5": 0.020197350828961107,
    "N@10": 0.03352065004174039,
    "N@20": 0.052347017479807044
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 35,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 109636,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.05279864739401549,
    "R@10": 0.10247295695720071,
    "R@20": 0.1733169180327174,
    "N@5": 0.030097039906547697,
    "N@10": 0.04598597226791548,
    "N@20": 0.06377830532821768
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 71,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 219270,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.