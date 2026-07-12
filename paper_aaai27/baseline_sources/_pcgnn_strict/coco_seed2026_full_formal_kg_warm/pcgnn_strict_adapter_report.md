# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 233326 |
| Validation sequence examples | 18919 |
| Test sequence examples | 37848 |
| Epochs | 20 |
| Last loss | 6.0940 |
| Last RS loss | 6.0798 |
| Last KG loss | 0.0142 |
| KG triples | 41414 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 6967 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 20 |
| Best validation score | 0.0132 |
| Stopped early | False |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\coco_seed2026_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.01491114029282902,
    "R@10": 0.026533896982276588,
    "R@20": 0.04665686634015092,
    "N@5": 0.009453716513016916,
    "N@10": 0.013166697464347628,
    "N@20": 0.018187065762572716
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 410,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 18919,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.012995853169187053,
    "R@10": 0.0255497894653098,
    "R@20": 0.048351821247903756,
    "N@5": 0.007520444213140964,
    "N@10": 0.011491910383969007,
    "N@20": 0.01719629204347135
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 819,
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