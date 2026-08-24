# Frozen Hot Expert With Masked Pseudo-Cold Adapter

## Scope

This is a single seed-2025, validation-only Stage B preflight. It asks whether
a shared content-only adapter can improve strict-Cold ranking while retaining a
frozen CKG Hot Expert and its mixed-bank Overall quality. It is not a main-table
experiment and has no test evaluation, CBI, simulator, PPO, reward, score
fusion, item-ID target, or graph edge write-back.

## Registered Data And Provenance

The runner manually reads only `meta.json`, `content_emb.pt`,
`static_train.pkl`, and `static_val.pkl`. It does not invoke data/split helpers
that read extra files. Validation popularity is reconstructed exclusively from
the static training rows.

Before any output is created, the runner requires:

- completed Hot manifest and result, both with gate status `completed`;
- a passed Hot result selecting epoch 15;
- the exact Hot checkpoint SHA256
  `a41c466d8244fa08e043cfd8dc0289e3f99f5dd5af351f4b891d62780a2c258f`;
- checkpoint payload fields `epoch`, `model_state`, and architecture config
  `emb_dim=64`, `mlp_hidden=64`, `layers_full=2`;
- matching preflight before/after hashes for the allowed meta/content/train/val
  inputs and the CGRC/data/evaluator sources used by this runner.

## Fixed Pseudo-Cold Protocol

The 698-course catalog must contain exactly 596 train-warm and 102 train-zero
courses. At each epoch, a deterministic generator seeded from `(2025, epoch)`
selects exactly 102 distinct IDs from the 596 warm IDs. Therefore the effective
warm-pool selection ratio is `102 / 596 = 0.17114094`; this protocol does not
claim a 15-percent warm mask.

All edges incident to the selected set `S` are removed from the student graph.
The runner asserts that every selected graph column has zero nonzeros and writes
the selected count, the catalog counts, the warm-pool ratio, and a sorted-ID
SHA256 to each epoch record.

## Model And Trust Region

Load `epoch_015.pt` into a fresh `CGRCNet`, freeze every parameter, and set it
to evaluation mode. Define the content base as

`c_i = normalize(frozen_hot.item_x()[i])`.

The only trainable module is a shared MLP with `LayerNorm(64) -> Linear(64,64)
-> GELU -> Linear(64,64)`. Its final linear layer is zero initialized and it has
no item-ID input or embedding. The raw update is projected to the tangent plane
of `c_i` and mapped onto the unit sphere with a capped exponential map. This
preserves a nonzero derivative to the final layer while retaining the exact
epoch-0 content-base output. With

`theta_max = 2 asin(tau / 2)` and `tau = 0.24929234`,

the final returned vector `a_i` always obeys

`||a_i - c_i||_2 <= tau`.

The angular cap is evaluated a few floating-point units inward before final
normalization, so float32 rounding cannot cross the registered chordal bound.

The returned delta is exactly `a_i - c_i`. There is no soft delta regularizer:
the hard final-space chordal cap is the only update constraint.

## Masked Ranking Objective

For an epoch's selected warm courses, recompute frozen user and non-selected
Hot-item banks on the edge-deleted graph. Selected items use the adapter; their
removed original training edges are the only positives. A positive edge
`(u, i)` receives 32 negatives sampled only from train-warm item IDs absent
from user `u`'s complete original training history. True train-zero catalog IDs
cannot be training negatives.

The per-course-balanced loss is

`L = (1 / |S|) sum_{i in S} (1 / |E_i|) sum_{(u,i) in E_i} CE(pos(u,i), neg(u))`,

where `E_i` is the removed-edge set for course `i`, and logits use temperature
`0.5`. Thus a course with many interactions cannot dominate the adapter update.
If any removed positive lacks all 32 eligible warm-only negatives, the epoch
fails rather than silently dropping that edge and changing the objective.

All fixed optimizer/training knobs are: adapter dimensions 64/64, 15 epochs,
batch size 4096, Adam learning rate `1e-3`, zero weight decay, and no soft
regularizer.

## Validation And Selection

First evaluate epoch 0 using the zero-initialized adapter. It must match the
completed Hot epoch-15 Cold/Hot/Overall R@10 and N@10 within `1e-5`; otherwise
the run fails before training. Epoch 0 is written to the validation CSV and is
the runtime selection baseline.

At every epoch, true validation uses the frozen full-graph Hot bank for all
warm items and the adapter for all 102 train-zero catalog IDs. The evaluator
uses full-catalog ranking, train-only history masks, item-macro Cold/Hot
metrics, and item-count-weighted Overall metrics.

An epoch is eligible when Hot R/N@10 and Overall R/N@10 are each no lower than
the epoch-0 baseline minus `0.003`. Select maximum Cold N@10, then Cold R@10,
then the later epoch. The result records `completed` or
`completed_gate_failed`; a no-eligible outcome still writes the validation rows
and result artifact.

## Outputs

The isolated runner writes `validation_epochs.csv`, epoch adapter checkpoints,
per-item Cold/Hot validation exports, and `adapter_preflight_result.json`. A launcher is
intentionally out of scope until these runner contracts are reviewed.
