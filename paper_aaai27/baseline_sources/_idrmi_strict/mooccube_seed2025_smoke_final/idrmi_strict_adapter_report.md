# IDRMI Strict Adapter Report

- status: `smoke_passed`
- seed: `2025`
- device: `cuda`
- best epoch: `1`
- positive train edges: `464314`
- candidate courses: `698`

## Metrics

- validation: `{"N@10": 0.0, "N@20": 0.013823659738799662, "N@5": 0.0, "R@10": 0.0, "R@20": 0.05263157894736842, "R@5": 0.0}`
- test: `{"N@10": 0.0, "N@20": 0.0024101897204141577, "N@5": 0.0, "R@10": 0.0, "R@20": 0.010416666666666666, "R@5": 0.0}`
- validation score std: `0.03078987`
- test score std: `0.02937475`

## Gates

- cuda_selected: `True`
- nonempty_adjacency: `True`
- warm_only_negatives: `True`
- finite_loss: `True`
- nonconstant_validation_scores: `True`
- nonconstant_test_scores: `True`
- validation_has_cold_courses: `True`
- test_has_cold_courses: `True`
- no_positive_cold_train_edges: `True`
