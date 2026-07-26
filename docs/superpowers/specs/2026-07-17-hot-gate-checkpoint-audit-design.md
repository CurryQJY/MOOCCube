# Hot-Gate Checkpoint Audit Design

## Goal

Run a low-cost seed-2025 audit that isolates restoration of the Hot ID/content gate and tests whether early checkpoints preserve Hot performance without sacrificing Cold performance.

## Scope and invariants

- Do not modify the protected main-table files.
- Keep the existing TDInit anchor, ContentDelta, USIM, and auxiliary-loss settings unless required to remove the prior Hot-only normalization confound.
- Remove only the Hot-specific pre-simulation L2 normalization so Hot and Cold enter simulation with the same scale convention.
- Train for at most 8 epochs with seed 2025 and save/evaluate every epoch.

## Selection and diagnostics

Report Cold, Hot, and item-count-weighted Overall metrics for every validation checkpoint. Select three candidates: best Cold, best Hot, and best normalized balanced score. Record gate weights and Cold/Hot ContentDelta norm/clipping statistics. Do not use test metrics for checkpoint selection.

## Success criterion

The experiment is informative if it determines whether an early Hot-gate checkpoint improves Hot R@10/N@10 while retaining a meaningful Cold gain. If no checkpoint satisfies that trade-off, abandon the gate-only route rather than launch a multi-seed run.
