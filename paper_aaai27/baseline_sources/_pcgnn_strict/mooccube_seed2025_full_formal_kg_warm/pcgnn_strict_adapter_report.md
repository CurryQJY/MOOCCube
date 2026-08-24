# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 267210 |
| Validation sequence examples | 32464 |
| Test sequence examples | 65605 |
| Epochs | 20 |
| Last loss | 3.8693 |
| Last RS loss | 3.8613 |
| Last KG loss | 0.0080 |
| KG triples | 5115 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 596 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 10 |
| Best validation score | 0.0526 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2025_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.05660935089478371,
    "R@10": 0.09705533244918368,
    "R@20": 0.1403490711122434,
    "N@5": 0.039339732374689886,
    "N@10": 0.05259846350937727,
    "N@20": 0.06326142472034664
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 34,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 32464,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.025134703151686796,
    "R@10": 0.06225125154268782,
    "R@20": 0.11185974721302742,
    "N@5": 0.015253324591537295,
    "N@10": 0.026940295853395413,
    "N@20": 0.039417761215994045
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 68,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 65605,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.