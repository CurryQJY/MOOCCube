# IDRMI Strict Adapter Report

- status: `smoke_passed`
- seed: `2025`
- device: `cuda`
- best epoch: `2`
- positive train edges: `464314`
- candidate courses: `698`

## Metrics

- validation: `{"N@10": 0.0014833456606240228, "N@20": 0.0016679165511466459, "N@5": 0.001225307842189813, "R@10": 0.0035841498539889356, "R@20": 0.004313467617738601, "R@5": 0.0028025154284821497}`
- test: `{"N@10": 0.0018207354803199383, "N@20": 0.007553541963262728, "N@5": 0.001381135975158887, "R@10": 0.0036923853427691597, "R@20": 0.026323415449138514, "R@5": 0.0023201898299486474}`
- validation score std: `0.03633763`
- test score std: `0.03588275`

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
