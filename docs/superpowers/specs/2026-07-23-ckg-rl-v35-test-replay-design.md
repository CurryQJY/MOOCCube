# CKG-RL V3.5 Frozen Test Replay Design

## Objective

Evaluate the already-selected V3.5 seed-2025 checkpoint once on the strict
cold test split without retraining, changing a model parameter, or selecting a
new epoch.  The replay is diagnostic because the seed-2025 test split was
inspected by earlier exploratory routes; it is not fresh confirmatory evidence.

## Frozen Source Contract

The replay source is:

```text
outputs/ckg_rl_usim_v35_action_distill/seed2025/action_distill_manifest.json
checkpoints/ckg_rl_usim_v35_action_distill/seed2025/{teacher,generator,policy}.pt
```

The manifest fixes epoch 15 and contains the SHA-256 digest of every stage
checkpoint.  Before any test row is loaded, the replay must:

1. Read the source manifest.
2. Verify `route=ckg_rl_usim_v35_action_distill`, `test_loaded=false`, and the
   selected epoch is non-negative.
3. Hash the three source checkpoint files and require exact agreement with the
   manifest.
4. Instantiate and load the teacher, vector generator, rank panels, course
   bias, and selected action-distilled actor from those checkpoints.

Any missing, stage-mismatched, or hash-mismatched checkpoint aborts the replay.

## Test Evaluation

Only after frozen-source validation, the replay reconstructs the same train-only
history and pseudo-item partition, loads `static_test.pkl` exactly once, and
calls the existing target-free full-ranking evaluator.  It evaluates hot and
strict-cold rows with the item bank generated from the frozen policy at the
already-selected epoch.  No optimizer, gradient, P_val diagnostic, C_val
metric, or checkpoint selection exists in this script.

The source P-only output directory is never edited.  Results are written to
`outputs/ckg_rl_usim_v35_action_distill/test_replay_seed2025/` with a manifest
that marks `diagnostic_only=true`, `test_loaded=true`, and records the source
hashes and selected epoch.

## Required Artifacts

- `test_replay_manifest.json`
- `test_metrics.json`
- `test_per_item_hot.csv`
- `test_per_item_cold.csv`

The replay result must preserve `policy_mode=action_distill_rollout` rather
than exposing the inherited V3.2 label `ppo_rollout`.
