# KGRec Strict-Cold Quick Screening Design

## Objective

Diagnose why all three formal KGRec runs select epoch 1 by adding missing
training diagnostics and running a bounded learning-rate screen on the MOOCCube
seed-2025 strict item-cold split.

The existing seed-2025/2026/2027 formal outputs remain unchanged. All screening
runs use new output directories and are treated as diagnostic experiments, not
paper results.

## Runner Diagnostics

Extend `paper_aaai27/scripts/run_kgrec_strict_seed.py` with two observable
behaviors:

1. Evaluate validation before the first optimizer update and write an epoch-0
   progress row. Epoch 0 participates in best-checkpoint selection so the run
   can detect whether training immediately damages strict-cold ranking.
2. Average the model-provided `rec_loss`, `mae_loss`, and `cl_loss` values over
   the optimizer steps in each epoch and store them in the corresponding
   progress row.

The epoch-0 row has no training loss or training batches. It records validation
metrics, validation score, best-checkpoint state, and early-stopping state. The
existing epoch numbering remains unchanged: the first trained epoch is epoch 1.

## Test Design

Add focused unit tests in `tests/test_kgrec_strict_runner.py` for:

- deterministic averaging of named loss components across batches;
- the epoch-0 progress-row shape and absence of training-only values;
- best-checkpoint comparison behavior when epoch 0 is the current best.

Tests use pure helper functions rather than constructing the full GPU model.
The new tests must fail before implementation and pass after the minimal runner
changes. The complete KGRec test set must also remain green.

## Screening Matrix

Use the seed-2025 atomic dataset and strict validation/test protocol for three
serial CUDA runs:

| Run | Learning rate | Epochs | Patience | Batch size |
|---|---:|---:|---:|---:|
| baseline-diagnostic | 1e-4 | 10 | 4 | 4096 |
| lr-5e-5 | 5e-5 | 10 | 4 | 4096 |
| lr-1e-5 | 1e-5 | 10 | 4 | 4096 |

All other KGRec hyperparameters remain at the current adapted defaults. Each
run writes to a separate directory under
`paper_aaai27/baseline_sources/_kgrec_strict/diagnostic_lr_screen_seed2025/`.

Runs execute serially because the workstation has one CUDA device and another
project process may share GPU memory.

## Selection Rules

Configuration selection uses validation data only. The primary measure is
strict cold item-macro NDCG@10. Secondary evidence is:

- whether the best checkpoint occurs after epoch 1;
- whether cold Recall@10 and NDCG@10 remain stable after their peak;
- whether total loss and component losses are finite and interpretable;
- whether epoch 0 is worse than at least one trained checkpoint.

The runner may still produce its normal test report, but test metrics are not
used to choose among the three configurations.

## Verification and Outputs

Before reporting a result:

- all KGRec unit tests must pass;
- each screening report must have `status: complete` and `device: cuda`;
- each run must contain a nonempty progress file and best checkpoint;
- progress rows must include epoch 0 and named loss components for trained
  epochs;
- no KGRec process may remain after the serial screen completes.

The final comparison reports the validation trajectory and loss components for
all three learning rates, identifies the preferred configuration, and states
whether the epoch-1 pattern is resolved or merely delayed.
