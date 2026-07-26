# CKG-RL V3.6 Globally Stable Action Distillation Design

## Objective

V3.6 addresses the two measured V3.5 failures without score fusion:

1. The actor captured only a small fraction of the legal-action oracle.
2. Local panel improvement raised strict-cold metrics but displaced hot items in
   the full catalog.

It retains the V3.2 teacher, V3.2 vector generator, legal target-free action
space, and inference rollout.  It changes only policy supervision.

## Fixed Controls

The following must be identical to V3.5 for seed 2025:

- Outer split, `H_train/H_val`, and deterministic `H_G/P_train/P_val` item
  allocation.
- Teacher fitting on `H_train`, teacher selection on `H_val`, and H_G-only
  vector generator fitting.
- Candidate retrieval, `END`, step size, displacement cap, course logit bias,
  full-ranking evaluator, and target-free deployment path.
- P-only checkpoint selection.  `C_val`, `C_test`, and `static_test.pkl` are
  not evaluated or loaded by the V3.6 screen.

## Globally Stable Counterfactual Utility

For pseudo item `i`, state `h`, candidate action `a`, and teacher item vector
`e_i`, V3.5 uses local panel gain `g_local(i, h, a)`.  V3.6 constructs a fixed
128-user anchor bank `A` only from `H_G` interaction users, selected by a
seeded SHA-256 ordering.  Let:

```text
D(h, e_i; A) = mean_{u in A} (u^T(h - e_i))^2
delta_D(a) = D(h(a), e_i; A) - D(h, e_i; A)
g_v36(a) = g_local(a) - lambda_stability * delta_D(a)
g_v36(END) = 0
pi_T(a | h, i) = softmax(g_v36(a) / action_temperature)
```

All vectors are existing normalized teacher vectors.  The stability term uses
the privileged teacher item only while building offline labels; inference has
no teacher item, anchor target, positive user, or reward access.

`lambda_stability=10.0` is fixed by a read-only V3.5 P_train calibration.  It
reduces mean local-best anchor drift from `3.22e-4` to `1.45e-4` while retaining
mean local action gain `0.01430` versus `0.01493` without the term.  The
calibration used no C_val or test row.

## Expert-State Mixture

V3.6 labels the actor's action distribution at every state as in V3.5, but the
state transition uses a deterministic 50/50 mixture of the teacher argmax and
the actor argmax.  The mixture decision is SHA-256 keyed by
`seed, epoch, rollout_step, item_id`, so it is reproducible and independent of
process-global RNG.  Teacher transitions expose high-value trajectories;
actor transitions retain deployment-state coverage.  Gradients never pass
through either state transition.

The implementation exposes both switches for ablations:

```text
expert_action_fraction = 0.50
global_stability_weight = 10.0
```

Setting either to zero removes only that component.

## Selection and Falsification

The selected policy maximizes non-negative P_val local panel rank gain, with
epoch 0 identity eligible.  Global stability and action agreement are recorded
diagnostics, not additional selection objectives.

The seed-2025 P-only screen passes only if the selected non-identity epoch:

- exceeds V3.5 P_val gain `0.0047182762`;
- exceeds V3.5 training action agreement `0.23880598`; and
- retains the V3.5 teacher and generator hashes exactly.

Otherwise V3.6 is rejected.  No seed-2025 C_val or test replay is permitted
for model selection.  Any later outer evaluation must use a fresh strict-cold
split/seed not used to develop V3.6.

## Required Artifacts

- `v36_manifest.json`
- `global_anchor_manifest.json`
- `generator_vector_epochs.csv`
- `policy_stable_action_epochs.csv`
- `p_val_selected_metrics.json`
- teacher, generator, and selected policy checkpoints

The manifest must state `test_loaded=false`, `outer_c_val_evaluated=false`,
`policy_optimizer=globally_stable_action_distillation`, and record both
component switches.
