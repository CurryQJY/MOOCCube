# Backbone-Anchored Ridge Route Design

## Decision

Use Route A: a Backbone-anchored Ridge residual. Validation-only dose-response
diagnostics found a shared feasible coefficient of `0.075` for all three Junyi
seeds under both the frozen and fine-tuned backbones. Route B, a learned
item-wise gate, is reserved for a pre-registered Route A failure. Route C,
removing Ridge, is rejected because every Junyi seed has positive cold NDCG@10
gain at the shared coefficient.

## Goal

Make the Ridge reconstruction part of the train/validation pathway without
allowing it to replace the complete strict-cold bank. The same anchored bank
must initialize core validation, pseudo-cold PPO training, real-cold PPO
evaluation, and downstream validation. No test split may be loaded before all
route, epoch, coefficient, and policy choices are fixed.

## Model

For selected rows `C`, construct

\[
z_i(\alpha)=\operatorname{normalize}\left((1-\alpha)z_i^{base}
 +\alpha z_i^{ridge}\right),\qquad i\in C,
\]

with `alpha` in `[0, 1]`. Rows outside `C` are copied bit-for-bit from the
Backbone bank. `alpha=1` preserves the released historical behavior;
`alpha=0` is the Backbone identity. The new cross-dataset route uses the
pre-registered shared value `0.075`.

The simulation Ridge bank receives the same transformation on its pseudo-cold
target rows. This prevents PPO from training from a full Ridge replacement and
then evaluating from a different initializer.

## Selection

Core epoch eligibility is defined relative to epoch 0. A candidate must retain
all four metrics within `0.003`:

- hot Recall@10;
- hot NDCG@10;
- overall Recall@10;
- overall NDCG@10.

Among eligible candidates, maximize cold NDCG@10, then overall NDCG@10, then
prefer the earlier epoch. Epoch 0 remains a legal identity result. PPO retains
the same four-metric rule relative to the anchored Ridge bank.

## Automatic Route Gate

Route A first runs Junyi seed 2026 with `--skip-test`. It advances when:

1. the selected core epoch is nonzero;
2. validation cold NDCG@10 exceeds the anchored epoch-0 value;
3. all four retention metrics pass the `0.003` floor;
4. the downstream PPO stage completes without reading test.

If Route A passes, repeat validation-only on seeds 2025 and 2027. The route is
locked when every seed satisfies retention and the mean cold NDCG@10 gain is
positive. Only then may a fresh output directory run the delayed test replay.

If seed 2026 fails because the selected epoch is zero or cold gain is not
positive, do not tune `alpha` on test. Switch to Route B: a learned per-item gate
bounded by `0.075`, initialized at zero, optimized in the full-ranking core
loss with an identity penalty. Route B requires a separate design and tests
before implementation.

## Compatibility and Provenance

- Default `ridge_alpha=1.0` keeps old launchers and released artifacts
  behavior-compatible.
- Manifests record `ridge_alpha` and identify the bank as
  `backbone_anchored_ridge`.
- Existing output directories are never overwritten; the new route uses
  `outputs/xds_junyi_anchored_a075/`.
- The active COCO ablation process and its output directories are out of scope.

## Tests

Unit tests must verify endpoint identities, unchanged non-target rows, row
normalization, invalid coefficient rejection, downstream parameter forwarding,
and strict four-metric epoch selection. A CPU smoke run uses only train and
validation files. Full experimental runs verify manifests and split-read
provenance before any result is accepted.
