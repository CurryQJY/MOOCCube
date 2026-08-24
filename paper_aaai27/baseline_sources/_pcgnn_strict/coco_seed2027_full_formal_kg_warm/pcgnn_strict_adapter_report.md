# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 233302 |
| Validation sequence examples | 18939 |
| Test sequence examples | 37858 |
| Epochs | 20 |
| Last loss | 6.3318 |
| Last RS loss | 6.3159 |
| Last KG loss | 0.0159 |
| KG triples | 41414 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 6966 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 10 |
| Best validation score | 0.0128 |
| Stopped early | True |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\coco_seed2027_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.014682015817036174,
    "R@10": 0.026421412280901294,
    "R@20": 0.044304768456265214,
    "N@5": 0.00902949323397425,
    "N@10": 0.01277154127194267,
    "N@20": 0.017239697901203513
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 410,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 18939,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.012662045963369953,
    "R@10": 0.02285109581030551,
    "R@20": 0.04138488693315352,
    "N@5": 0.007674399007025554,
    "N@10": 0.010925932812262824,
    "N@20": 0.015588349091504777
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 820,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 37858,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.