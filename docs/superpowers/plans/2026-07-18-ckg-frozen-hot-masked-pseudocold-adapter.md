# Frozen Hot Masked Pseudo-Cold Adapter Plan

## Objective

Implement an isolated seed-2025 validation runner for a frozen CKG Hot Expert
and a shared content adapter. Keep it separate from all main-table code and do
not add a launcher before runner review.

## Completed Contracts

- [x] Replace the ambiguous ratio with exact catalog invariants: 698 total,
  596 train-warm, 102 train-zero, and exactly 102 selected warm IDs per epoch.
  The logged warm-pool ratio is `102/596`; it is not a percentage mask claim.
- [x] Use a differentiable tangent-space exponential map with a small numerical
  inward margin, so the final output remains within chordal distance
  `0.24929234` in float32. The final shift is returned as the delta; no soft
  regularizer is permitted.
- [x] Lock adapter dimensions 64/64, 15 epochs, batch 4096, 32 warm-only
  negatives per positive, temperature 0.5, Adam `lr=1e-3`, zero weight decay,
  parity tolerance `1e-5`, and retention tolerance `0.003`.
- [x] Add fail-closed checks for completed Hot artifacts, the registered
  epoch-15 checkpoint SHA256/config/payload, and manifest-bound meta/content/
  train/validation/source hashes.
- [x] Manually load only meta/content/static train/static validation inputs and
  rebuild validation popularity from train.
- [x] Add epoch-0 parity and runtime-baseline selection helpers.

## Runner Flow

1. Validate the static split, Hot result/manifest/checkpoint, and input/source
   hashes before creating output artifacts.
2. Freeze `CGRCNet`, form normalized full-graph Hot and content banks, and
   confirm the warm q75 calibration within `1e-4`.
3. Evaluate and export epoch 0; reject a parity deviation above `1e-5`.
4. For each epoch, deterministically select 102 warm items, delete all of their
   graph edges, assert zero selected columns, recompute frozen masked banks,
   and update only the shared adapter.
5. Use removed edges as positives and warm-only candidates absent from original
   train history as negatives. Require all 32 negatives for every removed
   positive, then weight each edge by `1/|E_i|` and average by selected course,
   so interaction-heavy courses cannot dominate.
6. Evaluate full-catalog validation with all train-zero IDs routed through the
   adapter; export item-macro Cold/Hot, weighted Overall, row audit data, and
   an adapter checkpoint.
7. Select from epoch 0 through epoch 15 with four retention guards. Always
   write a result, including a gate-failed artifact if selection has no eligible
   row.

## Verification

- Focused pytest covers config locks, final-space trust cap, deterministic fixed
  selection, graph masking, warm-only negatives, mixed-bank routing, Hot
  preflight/checkpoint/hash failures, epoch-0 parity, selection, and manual
  validation-only input loading.
- Compile the isolated runner and run the focused regression suite with the
  Hot-preflight tests.
- Before any formal run, review source and data provenance again; launcher,
  test evaluation, and main-table code remain out of scope.
