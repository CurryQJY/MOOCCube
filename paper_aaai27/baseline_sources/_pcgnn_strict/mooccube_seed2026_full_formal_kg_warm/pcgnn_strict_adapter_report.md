# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 266368 |
| Validation sequence examples | 32988 |
| Test sequence examples | 66749 |
| Epochs | 20 |
| Last loss | 3.9974 |
| Last RS loss | 3.9910 |
| Last KG loss | 0.0064 |
| KG triples | 5115 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 596 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 13 |
| Best validation score | 0.0306 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2026_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.03915366131417576,
    "R@10": 0.062461145256530654,
    "R@20": 0.09519210855756084,
    "N@5": 0.022981940927344707,
    "N@10": 0.030580574062001018,
    "N@20": 0.038819383313296374
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 34,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 32988,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.03206210607662479,
    "R@10": 0.05436008596318311,
    "R@20": 0.1024040058943937,
    "N@5": 0.019696842030441898,
    "N@10": 0.02684256037918414,
    "N@20": 0.03875188723752332
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 68,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 66749,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.