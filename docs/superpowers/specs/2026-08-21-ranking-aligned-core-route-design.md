# Ranking-Aligned Core Route Design

## Trigger

Route A established a stable Backbone-anchored Ridge/PPO path on Junyi, but its
seed-2026 core decomposition failed the component requirement:

- Graph-only selected cold NDCG@10: `0.060186`;
- Pseudo-only selected cold NDCG@10: `0.059824`;
- Full selected cold NDCG@10: `0.059828`.

The pseudo increment is negative and the course increment is effectively zero.
The course hinge is active on only 0.19%-1.10% of Junyi relation pairs. Route
B2 therefore changes the auxiliary objectives, not the already validated Ridge
coefficient.

## Decision

Keep `ridge_alpha=0.075`, the strict four-metric retention gate, the existing
graph objective, and all split boundaries. Add opt-in ranking-aligned auxiliary
loss modes while preserving the released cosine/margin defaults.

## Course Objective

For each related pair `(i, j)`, sample a deterministic catalog negative `n`
that differs from both endpoints. Optimize

\[
L_{course}^{rank} =
\frac{\sum w_{ij}\,\operatorname{softplus}((s_{in}-s_{ij})/\tau)}
     {\sum w_{ij}},
\]

where `s` is cosine similarity and `tau=0.2`. Unlike the old hinge, this loss
has a nonzero gradient even when a positive pair already exceeds cosine 0.2.

## Pseudo-Cold Objective

For the stochastic edge-masked item set `M`, score each normalized masked
representation against the detached factual catalog bank and apply catalog
cross-entropy:

\[
L_{pseudo}^{rank} = -\frac{1}{|M|}\sum_{i\in M}
\log\frac{\exp(\cos(z_i^{masked},z_i^{factual})/\tau)}
{\sum_j\exp(\cos(z_i^{masked},z_j^{factual})/\tau)}.
\]

This retains the target-detachment and edge-removal guarantees while training
ranking geometry rather than only positive cosine.

## Compatibility

- `--course-loss-mode margin` remains the default.
- `--pseudo-loss-mode cosine` remains the default.
- `--aux-temperature 0.2` is used only by contrastive modes.
- Existing manifests remain readable; new runs record both modes and the
  temperature through their config block.

## Automatic Gate

Run seed 2026 validation-only for graph-only, contrastive pseudo-only, and
contrastive Full. Route B2 advances only if:

1. pseudo-only minus graph-only cold NDCG@10 is positive;
2. Full minus pseudo-only cold NDCG@10 is positive;
3. both selected rows satisfy strict hot/overall Recall/NDCG retention;
4. neither result uses test.

If either component increment is non-positive, stop Route B2. Do not tune loss
weights or temperature on Junyi test. If both pass, repeat the frozen
configuration on seeds 2025 and 2027 before any delayed test replay.

## Tests

Tests prove that both contrastive objectives retain nonzero gradients in cases
where the old objectives are saturated, factual pseudo targets remain detached,
negative sampling excludes pair endpoints, defaults preserve old behavior, and
invalid loss modes or temperatures are rejected.
