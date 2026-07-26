# CKG-RL V3.3 Rank-Distilled USIM Design

## Status

Approved for an isolated seed-2025 acceptance experiment after an implementation-plan audit. V3.2 remains unchanged and is the direct clean reference. Historical V2, V3, and V3.1 remain read-only comparison routes.

## Problem

V3.2 has a clean train/inference boundary but its content generator is supervised only by item-vector reconstruction and its PPO reward is an item-vector/positive-score proxy. Consequently, a rollout can act as an arbitrary residual vector editor instead of learning a user-behaviour simulation. The V3.3 objective is to make each rollout improve an observable, frozen-teacher user-affinity ranking distribution.

## Visibility Protocol

The existing outer split and warm partitions remain fixed:

```text
H_train = static_train
H_val, C_val = warm/cold rows in static_val
H_test, C_test = warm/cold rows in static_test
H_train items = H_G disjoint P_train disjoint P_val
```

The frozen `CleanTeacher` is fitted only on `H_train` and selected on `H_val`. `H_test` and `C_test` are not loaded before policy selection.

The prior draft had one invalid shortcut: directly rank-calibrating the generator on `P_train` would let policy examples influence the generator before the pseudo-cold policy phase. V3.3 instead uses only an internal item-level split of `H_G` for generator vector and rank supervision. `P_train` is used only to train the policy reward; `P_val` is used only to measure policy-side rank calibration. This preserves a content-only frozen initial state for both policy partitions.

## Deterministic Teacher Panels

After the teacher is frozen, build one fixed user panel per item in `H_G`, `P_train`, and `P_val`, using only `H_train` interactions and the teacher tables:

1. Deterministically select up to `panel_positive_count` observed `H_train` users of the item.
2. Add teacher hard users with the highest frozen teacher affinity, excluding already selected positives.
3. Add deterministic seeded non-positive users, excluding existing panel users.
4. If necessary, fill from the remaining user IDs in deterministic order. The global panel width is `min(panel_size, n_users)`, so no padding duplicate changes a distribution.

The panel builder stores only IDs, source counts, seed, width, and a SHA-256 digest in the run manifest. It cannot read outer validation/test rows or any strict-cold target interaction.

For an item `i` and panel `U_i`, define the frozen teacher distribution and a state distribution as:

```text
q_T(u | i) = softmax(normalize(e_u^T)^T normalize(e_i^T) / tau)
q(u | h)   = softmax(normalize(e_u^T)^T normalize(h) / tau)
```

`q_T` is detached. It is available only for warm pseudo items during training and validation, never for strict-cold inference.

## Stage G: Rank-Calibrated Content Generator

For the internal `H_G` generator-training items, optimize:

```text
L_G = L_vector(G(c_i), e_i^T) + lambda_rank * KL(q_T(. | i) || q(. | normalize(G(c_i))))
```

The existing normalized MSE/cosine vector objective supplies `L_vector`. Generator checkpoint selection occurs on the disjoint internal `H_G` validation items, ordered by rank KL and then vector loss. Neither `P_train`, `P_val`, nor outer validation/test item interactions select the generator.

## Stage P: Counterfactual Rank-Gain Rollout

The V3.3 policy retains V3.2's inference-legal candidate construction: current-state nearest frozen teacher users plus `END`. It does not receive true positive users, target residual candidates, or a cold target ID at inference.

For each training pseudo item, the policy starts at `h_0 = normalize(G(c_i))`. The frozen target distribution is calculated on its training-only panel. For an active transition `h_t -> h_{t+1}`:

```text
r_rank = KL(q_T || q(. | h_t)) - KL(q_T || q(. | h_{t+1}))
r = r_rank + lambda_course * r_course
    - step_penalty - lambda_delta * ||h_{t+1} - h_t||_2
```

`r_course` remains the existing course signal over catalog metadata and target-excluded `H_train` user histories. The former embedding-distance and full-positive-score rewards are not used by V3.3. PPO replay remains detached and retains the existing terminal critic anchor.

The policy checkpoint gate keeps V3.2's hot-validation retention floor. A PPO epoch is eligible only when its `P_val` rank gain is non-negative; epoch 0 (the identity generator) is always retained as the safe baseline. Eligible epochs are ranked by `C_val` NDCG@10 then Recall@10, with lower `P_val` rank KL / higher `P_val` rank gain as deterministic tie-breakers. It never reads outer test metrics during selection.

## Deployment

The deployed model is one route, not a score fusion:

```text
hot item: frozen teacher item bank
cold item: catalog content -> rank-calibrated generator -> target-free V3.3 rollout
```

Strict-cold rollout receives only state, frozen teacher user bank, catalog metadata, and train-only target-excluded user histories. It receives no target teacher item vector, panel, known positive user, test interaction, or reward calculation.

## Required Tests and Artifacts

Unit tests must establish:

1. panels are deterministic, fixed-width, disjoint from outer rows, and depend only on `H_train`;
2. rank KL and incremental rank gain have the stated direction and are finite;
3. generator rank loss uses only `H_G` targets and selects from its internal holdout;
4. V3.3 policy reward ignores the legacy embedding/positive-score terms and uses the panel target only during training;
5. target-free inference rejects target/panel/positive-user inputs; and
6. hot item vectors are byte-equal to the frozen teacher bank.

Each run writes `rank_panel_manifest.json`, `generator_rank_epochs.csv`, `policy_rank_epochs.csv`, `v33_manifest.json`, and `final_metrics.json`. The first experiment is seed 2025 with a smoke preflight. A three-seed campaign is allowed only when the smoke passes and the full run demonstrates non-degenerate panel-KL/rank-gain diagnostics and a retained hot validation score.
