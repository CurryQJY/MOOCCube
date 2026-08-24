# Main-Checkpoint Actor Inference A/B Design

## Goal

Measure whether applying the trained Actor policy to strict-cold course representations at evaluation time improves the current main-table full-ranking result. The comparison must isolate the inference path: both arms use the same frozen checkpoint, split, candidate catalog, train-history mask, and metric implementation.

## Checkpoint scope

Use the recovered main-table checkpoints under:

`checkpoints/recovery_validation/main_table_51ea12fc_candidate/strict_item_cold_balanced_thr1_seed_<seed>`

Seed 2025 is the exact recovered main-table reproduction and is the first diagnostic. Seed 2026 is also complete. Seed 2027 is currently still training and must not be read until a finished checkpoint is written. The final aggregate comparison uses seeds 2025, 2026, and 2027; partial results must be labeled explicitly.

## Evaluation arms

### A. Static inference control

Preserve the current evaluation path:

1. Mask the ID branch for strict-cold courses.
2. Build the static course vector with the frozen content/fusion encoder.
3. Build the projected user vector.
4. Score the full catalog by normalized dot product.
5. Mask the user's training history and compute Recall/NDCG.

### B. Deterministic Actor refinement

Change only the cold-course item bank:

1. Start from the same strict-cold static course vector as arm A.
2. For each of the five simulator steps, retrieve and sample candidates exactly as configured by the checkpoint, while resetting a fixed evaluation RNG so candidate construction is reproducible.
3. Select the action by Actor-logit argmax, not stochastic sampling.
4. Apply the trained state transition and configured residual scale.
5. Cache the refined cold-course vectors, then run the same full-ranking evaluator as arm A.

The Critic may be called only for audit/value reporting. It must not select actions or transform the course vector because the trained architecture defines it as a value estimator, not an inference encoder.

## Leakage and fairness constraints

- Do not use test labels, held-out interactions, target behavioral embeddings, or test-set rewards in refinement.
- Candidate-user histories and educational artifacts must match the main-table manifest.
- Warm-course vectors, user vectors, score temperature, seen-item masking, and candidate catalog must be identical between arms.
- Load the checkpoint read-only and write results to a new output directory.
- Reject a run when the split/config fingerprint does not match the checkpoint.

## Outputs

For each seed, save:

- static and Actor-refined cold item-macro Recall/NDCG at 5, 10, and 20;
- interaction-weighted cold metrics as diagnostics;
- per-cold-course metric differences;
- Actor/state-transition call counts;
- representation displacement statistics (cosine similarity and L2 distance before/after refinement);
- item-bank construction time and full-ranking evaluation time;
- a machine-readable comparison CSV and a concise text summary.

## Tests

Tests are written before the evaluation implementation and must demonstrate:

1. the static arm calls neither Actor nor simulator;
2. the refined arm calls the Actor and state transition for strict-cold courses;
3. repeated refined evaluation is bit-identical;
4. no held-out label or target behavioral embedding enters refinement;
5. warm-course vectors remain unchanged;
6. both arms use the same evaluator and target-course restoration after history masking;
7. checkpoint and split/config provenance mismatches are rejected.

## Decision rule

Report effect sizes rather than declaring success from one seed. Actor inference is supported only if the three-seed cold item-macro comparison is consistently non-negative and improves the primary NDCG metrics without relying on test-time selection. A seed-2025 improvement alone is diagnostic and cannot replace the three-seed result.

## Non-goals

- No retraining or hyperparameter search on the test set.
- No modification of the main training source or existing result files.
- No new Critic-guided search, beam search, or reward re-optimization.
- No paper-table replacement until the complete three-seed comparison is available.
