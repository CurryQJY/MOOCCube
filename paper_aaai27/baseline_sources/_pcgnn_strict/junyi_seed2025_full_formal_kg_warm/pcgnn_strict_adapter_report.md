# PCGNN Strict Adapter Report

| Field | Value |
|---|---:|
| Train sequence examples | 1339698 |
| Validation sequence examples | 109293 |
| Test sequence examples | 218398 |
| Epochs | 20 |
| Last loss | 3.2846 |
| Last RS loss | 3.2838 |
| Last KG loss | 0.0008 |
| KG triples | 4532 |
| KG loss weight | 1.0 |
| RS candidate mode | warm |
| RS candidate items | 616 |
| Validation metric | full_cold_item_macro.N@10 |
| Best epoch | 16 |
| Best validation score | 0.0476 |
| Stopped early | False |
| Best checkpoint | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_pcgnn_strict\junyi_seed2025_full_formal_kg_warm\checkpoints\best_model.pt |

## Validation

```json
{
  "full_cold_item_macro": {
    "R@5": 0.05211099464885227,
    "R@10": 0.10628393053840539,
    "R@20": 0.20995118149574013,
    "N@5": 0.030283323474207702,
    "N@10": 0.047609105457880836,
    "N@20": 0.07371508763765407
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 35,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 109293,
  "rows_full_hot": 0
}
```

## Test

```json
{
  "full_cold_item_macro": {
    "R@5": 0.04039911032071352,
    "R@10": 0.08790424170875298,
    "R@20": 0.15691456805687934,
    "N@5": 0.023163722390013325,
    "N@10": 0.03834302024784767,
    "N@20": 0.05554557407757177
  },
  "full_hot_item_macro": {},
  "count_full_cold_item_macro": 71,
  "count_full_hot_item_macro": 0,
  "rows_full_cold": 218398,
  "rows_full_hot": 0
}
```

## Notes

- This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.
- Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.
- The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.
- Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.
- Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.