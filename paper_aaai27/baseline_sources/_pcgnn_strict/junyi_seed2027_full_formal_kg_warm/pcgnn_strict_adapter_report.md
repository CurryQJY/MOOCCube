# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 1339148 |
| Validation sequence examples | 109537 |
| Test sequence examples | 219206 |
| Epochs | 20 |
| Last loss | 3.4238 |
| Last RS loss | 3.4226 |
| Last KG loss | 0.0012 |
| KG triples | 4532 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 615 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 7 |
| Best validation score | 0.0361 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\junyi_seed2027_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.04170622912446861,
    "R@10": 0.07912517042723993,
    "R@20": 0.14443006463456515,
    "N@5": 0.023989844419354082,
    "N@10": 0.036094312486868646,
    "N@20": 0.0524821531459495
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 36,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 109537,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.04283308620385778,
    "R@10": 0.07880763326159881,
    "R@20": 0.14638125153729492,
    "N@5": 0.024910178550486002,
    "N@10": 0.03640083315471509,
    "N@20": 0.05330010681867861
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 71,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 219206,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.