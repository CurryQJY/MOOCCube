# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 268813 |
| Validation sequence examples | 32370 |
| Test sequence examples | 62280 |
| Epochs | 20 |
| Last loss | 3.9808 |
| Last RS loss | 3.9740 |
| Last KG loss | 0.0068 |
| KG triples | 5115 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 596 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 12 |
| Best validation score | 0.0466 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2027_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.05507753689860571,
    "R@10": 0.08026700607721882,
    "R@20": 0.14020033462061765,
    "N@5": 0.03872529057670615,
    "N@10": 0.046613359189110616,
    "N@20": 0.06150203696608009
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 34,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 32370,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.015036364994402238,
    "R@10": 0.03931020062854771,
    "R@20": 0.08481759132359083,
    "N@5": 0.008423276826784943,
    "N@10": 0.016297364448429857,
    "N@20": 0.027606660139626923
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 68,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 62280,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.