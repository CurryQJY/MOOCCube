# Stage B Multi-Seed Replication Design

## Goal

Replicate the completed seed-2025 frozen-Hot masked pseudo-cold adapter screen
on strict split seeds 2026 and 2027, validation only. This is a robustness
check, not a hyperparameter search and not a test-set evaluation.

## Fixed Protocol

For every seed, retain the same registered mechanism:

- fresh CGRC Hot preflight for 15 epochs with the existing pre-registered Hot
  floors;
- select only its own completed Hot checkpoint selected by the unchanged Hot
  validation rule; its epoch is recorded rather than assumed to be 15;
- exactly 102 pseudo-cold items from 596 warm items per adapter epoch;
- shared content-only 64/64 residual MLP, zero final layer, 15 adapter epochs,
  Adam `lr=1e-3`, no weight decay or soft delta penalty;
- 32 warm-only negatives per removed positive, original-history exclusion,
  course-balanced loss, and the same four `0.003` retention guards;
- epoch-0 parity and full-catalog, item-macro validation evaluation only.

The trust radius remains the registered fixed value `0.24929234` from the
completed seed-2025 screen. Each seed's q75 of normalized warm Hot/content
displacement is recomputed and recorded only as an audit statistic; it does not
alter tau or gate the run. This avoids turning multi-seed replication into a
per-seed hyperparameter calibration.

## Isolation And Provenance

Each seed gets unique Hot output/checkpoint/log roots and unique Stage B
output/checkpoint/log roots. Both launchers require fresh roots, use native
stderr-safe process handling, and hash inputs/source/protected files before
and after execution. The Stage B launcher additionally binds the Hot manifest,
result, selected checkpoint SHA256, split/data/source hashes, selected epoch,
and the fixed tau plus audited q75.

Neither phase reads a test split. The replication Hot preflight manually reads
only meta, content, static train, and static validation; the adapter has the
same restricted input set.

The Hot contract is a JSON object embedded in that seed's `preflight_result`
under `selected_checkpoint_contract`:

```json
{
  "schema_version": 1,
  "seed": 2026,
  "epoch": 12,
  "relative_path": "epoch_012.pt",
  "sha256": "<64 lowercase hex characters>",
  "architecture": {"emb_dim": 64, "mlp_hidden": 64, "layers_full": 2},
  "fixed_trust_tau": 0.24929234,
  "warm_q75_audit": 0.0
}
```

The concrete epoch and q75 vary by seed; all other fields are fixed contract
rules. The adapter rejects any mismatch among the result, checkpoint payload,
current checkpoint hash, seed, split/data/source provenance, or fixed tau.

## Decision Rule

For each seed select the validation epoch with the highest Cold N@10 among
epochs passing all Hot/Overall guards, breaking ties with Cold R@10 and later
epoch. The seed passes only when selected Cold N@10 improves by at least
`0.003` from its own epoch-0 row without a Cold R@10 decrease.

After both runs, report all three seeds separately plus mean and standard
deviation of selected Cold/Hot/Overall metrics and their deltas. Do not run the
test set or merge this path into the main-table implementation in this phase.
