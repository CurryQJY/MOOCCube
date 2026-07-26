# Priority Baseline Concrete Experiment Analysis

## Summary

| Priority | Candidate | Concrete experiment | Status | Main finding |
|---:|---|---|---|---|
| 1 | PCGNN | one-step loss + strict-history full-sort smoke | ok | loss=6.5364, eval_cases=32, median_rank=325.5 |
| 2 | UPGPR | strict split path-reachability proxy | ok | val_reach=0.750, test_reach=0.778 |
| 3 | MSEC-Rec | matrix density and dependency gate | blocked_by_dependency | inputs exist but DGL is missing; sampled evaluator still needs replacement |
| 4 | KGAN | official loader on adapted smoke data | ok_loader_only | train=[2884, 3], test=[328, 3], random split remains |

## Details

### PCGNN

```json
{
  "status": "ok",
  "train_sequence_examples": 32,
  "eval_sequence_examples": 32,
  "loss_after_one_step": 6.536442279815674,
  "score_shape": [
    32,
    699
  ],
  "sample_recall_at_10": 0.0,
  "sample_ndcg_at_10": 0.0,
  "median_target_rank": 325.5,
  "interpretation": "Forward/loss/full-sort smoke only; metrics are not publishable because this is one mini-batch on a capped smoke dataset."
}
```

### UPGPR

```json
{
  "status": "ok",
  "train_pairs": 2000,
  "validation": {
    "eval_pairs": 500,
    "with_train_history": 4,
    "target_reachable": 3,
    "target_reachable_rate": 0.75
  },
  "test": {
    "eval_pairs": 500,
    "with_train_history": 9,
    "target_reachable": 7,
    "target_reachable_rate": 0.7777777777777778
  },
  "interpretation": "Coverage measures whether a target cold item shares concept/teacher/school with any train-history item; it is an upper-bound proxy for path-based candidate reachability, not trained UPGPR accuracy."
}
```

### MSEC-Rec

```json
{
  "status": "blocked_by_dependency",
  "dependency": "dgl missing in current py.bat environment",
  "matrices": {
    "train_uc.npy": {
      "shape": [
        2976,
        698
      ],
      "nonzero": 2000,
      "density": 0.0009628123363219028
    },
    "val_uc.npy": {
      "shape": [
        2976,
        698
      ],
      "nonzero": 500,
      "density": 0.0002407030840804757
    },
    "train_uv.npy": {
      "shape": [
        2976,
        5180
      ],
      "nonzero": 14689,
      "density": 0.0009528609831029186
    },
    "ck.npy": {
      "shape": [
        698,
        9712
      ],
      "nonzero": 5422,
      "density": 0.000799825814400287
    },
    "course_video.npy": {
      "shape": [
        698,
        5180
      ],
      "nonzero": 2094,
      "density": 0.0005791505791505791
    },
    "video_concept.npy": {
      "shape": [
        5180,
        9712
      ],
      "nonzero": 23310,
      "density": 0.0004633443163097199
    }
  },
  "interpretation": "MSEC matrix inputs are structurally available; official graph/model execution requires DGL and evaluator replacement."
}
```

### KGAN

```json
{
  "status": "ok_loader_only",
  "train_shape": [
    2884,
    3
  ],
  "test_shape": [
    328,
    3
  ],
  "n_entity": 27743,
  "n_relation": 25,
  "relation_set": 25,
  "aggregate_users": 1598,
  "interpretation": "Official loader accepts adapted data but randomly splits ratings_final.txt, so it is not strict-protocol-safe without a custom loader/evaluator."
}
```
