# IDRMI Strict Adapter Report

- status: `smoke_passed`
- seed: `2025`
- device: `cuda`
- best epoch: `3`
- positive train edges: `464314`
- candidate courses: `698`

## Metrics

- validation: `{"N@10": 0.002464569349888506, "N@20": 0.0029773321176295537, "N@5": 0.002464569349888506, "R@10": 0.00390625, "R@20": 0.0057444852941176475, "R@5": 0.00390625}`
- test: `{"N@10": 0.00041954256359635396, "N@20": 0.0007823872660636979, "N@5": 0.0, "R@10": 0.001451378809869376, "R@20": 0.002902757619738752, "R@5": 0.0}`
- validation score std: `0.03234908`
- test score std: `0.03186282`

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
