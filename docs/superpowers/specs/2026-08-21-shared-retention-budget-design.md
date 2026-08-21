# Shared Retention Budget Design

## Decision

Use one Backbone-anchored hot/overall retention budget for the complete
`Backbone -> anchored Ridge -> PPO` route. Keep historical commands compatible
by retaining the existing Ridge-relative gate as the default, while the new
route explicitly selects `retention_reference=backbone`.

## Problem

The existing staged route gives anchored Ridge a `0.003` retention allowance
against Backbone and then gives PPO another `0.003` allowance against Ridge.
Every stage can therefore pass while the final model is more than `0.003`
below Backbone. This occurs in the validation-only decoupled runs: mean final
Hot Recall/NDCG deltas are `-0.00350/-0.00307` on MOOCCube, and mean final Hot
NDCG delta is `-0.00307` on Junyi.

The code matches the previous stage-relative design, but that design cannot
support a paper claim that the final method strictly retains hot and overall
quality relative to its Graph-KNP source.

## Alternatives

### Keep stage-relative budgets

This preserves current results but requires describing retention as two local
constraints. It leaves a reviewer-visible mismatch between the method claim
and final baseline deltas.

### Use a global budget with the old coarse residual grid

This is semantically correct, but the smallest nonzero displacement is `0.05`.
Several seeds have less than `0.001` retention budget left after Ridge, so the
selector would unnecessarily fall back to identity.

### Global budget with a refined residual grid (selected)

Use the Backbone metrics as the common retention reference and evaluate a
fixed grid with additional near-zero values:

```text
0, .0025, .005, .0075, .01, .015, .02, .03, .04, .05,
.075, .10, .15, .20, .25
```

The policy, rewards, training epochs, maximum displacement, and simulator do
not change. The finer grid only selects how much of an already trained residual
is applied at validation and deployment.

## Interface

- Add `--retention-reference {ridge,backbone}`, default `ridge`.
- Add `--delta-grid` as one or more floats, defaulting to the historical grid
  `(0,.05,.10,.15,.20,.25)`.
- Validate that the grid is finite, unique, increasing, starts at zero, and
  stays in `[0,1]`.
- Evaluate and record Backbone and anchored Ridge validation metrics.
- Use the selected retention reference for PPO, greedy, and random-arm
  eligibility.
- Record the resolved reference and grid in the run manifest.
- Thread both options through the staged core runner.

## Selection

Among rows that keep Backbone-relative Hot Recall@10, Hot NDCG@10, Overall
Recall@10, and Overall NDCG@10 within `0.003`, maximize Cold NDCG@10, then
pseudo validation cosine, then Overall NDCG@10, then prefer smaller residuals
and earlier epochs. Epoch 0, delta 0 remains the identity fallback.

## Validation

1. Unit-test grid validation, historical defaults, explicit fine-grid parsing,
   reference selection, and downstream forwarding.
2. Run MOOCCube seed 3030 with `(alpha_real,alpha_sim)=(0.075,1.0)`, Backbone
   retention, and the fixed fine grid. Require a nonzero selected delta,
   positive PPO Cold NDCG@10 over Ridge, and all four final metrics within
   `0.003` of Backbone.
3. If the screen passes, freeze this exact route and repeat validation-only on
   three seeds of MOOCCube, Junyi, and COCO from source Graph-KNP checkpoints.
4. Require positive mean PPO Cold NDCG@10, at least two positive seeds per
   dataset, and per-seed global retention.

No test split is read. Because this design responds to observed validation
results, all current splits remain development data. A paper-level
confirmatory claim requires new blind splits and matching source training after
the route is frozen.

## Scope

No reward, policy architecture, Ridge fit, pseudo simulator, course signal,
training epoch, or test-release behavior changes.
