# Ridge-Initialized Course-Reward RL Pilot

## Goal

Run one leakage-safe MOOCCube seed-2025 pilot to determine whether the existing
course-reward simulation policy adds value when its initial cold-item vectors
come from the selected Ridge cold-bank reconstruction.

This is a feasibility experiment, not yet a replacement of the three-dataset
main model.

## Scope and fixed protocol

- Dataset: `processed_data_hin_clean_pop5`.
- Split: `outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025`.
- Backbone checkpoint: `outputs/graph_knp_final/seed2025/best.pt`.
- Training information: `H_train` only. The outer validation split selects the
  policy arm and displacement cap; the outer test split is read once afterward.
- Evaluation: full-catalog item-macro course-cold, hot, and overall
  Recall/NDCG, using the existing evaluator.
- No test-course interaction, test-derived Ridge fit, or target-including
  history may enter fitting or policy rollout.

## Proposed model

1. Load the frozen graph-gated backbone and construct its factual user/item
   banks using the existing strict-cold evaluator.
2. Fit the content-to-collaborative Ridge map only on warm items. Select its
   regularization from an inner warm-only split, as in the current Ridge arm.
   For the pseudo-cold simulation, reserve a deterministic warm donor pool and
   fit the simulation Ridge map on that pool only; neither pseudo-cold policy
   partition may contribute its factual target to that map. The final outer
   cold bank may use all available warm donors, matching the current Ridge arm.
3. For each pseudo-cold warm item used to train the policy, remove all of that
   item's train bipartite edges and renormalize the graph with
   `mask_bipartite_item_edges`. The pseudo-cold initial vector is the Ridge
   prediction, while the factual warm vector is a detached training target.
4. Run the existing legal user-simulation policy from the Ridge vector. The
   policy chooses only retrieved user directions. Its bounded residual is
   projected with the existing per-item displacement cap.
5. Build the pseudo-cold partitions from eligible warm items with train
   popularity in `[1, 25]`, using a seed-2025 deterministic permutation: 80%
   policy-train and 20% policy-validation, with no item overlap. Train the
   policy with PPO on the first partition and select it on the second. The
   reward retains embedding gain,
   positive-user ranking gain, step cost, and optionally the existing
   target-excluded course compatibility reward.
6. At outer evaluation, replace only real cold rows with the Ridge vector or
   the target-free Ridge+policy vector; warm rows remain factual backbone rows.

The first implementation should use the existing `CleanUSIMEngine` and
`CleanRecPPO` interfaces. It should add an adapter for a frozen Ridge
initializer rather than changing the semantics of the historical clean RL
route.

## Arms and decision gate

The pilot must produce these arms under the same seed and split:

1. `ridge_base`: Ridge bank, no policy.
2. `ridge_greedy_course_fit`: current deterministic `course_fit` walk, retained
   as a compatibility control.
3. `ridge_ppo_no_course_reward`: Ridge initialization with PPO and the existing
   embedding/ranking reward only.
4. `ridge_ppo_course_reward`: Ridge initialization with PPO plus the
   target-excluded course reward.

The primary decision is not raw cold improvement. The proposed arm advances
only if it improves cold N@10 over `ridge_base` at a matched hot N@10 level,
while satisfying the validation overall-R@10 floor. A raw cold gain that is
paid for by a larger hot loss is recorded as a failed trade-off.

The pilot is inconclusive if the policy selects epoch zero, collapses to a
zero residual, or fails the hot-retention constraint. Such an outcome is a
result, not permission to tune on the test split.

## Required diagnostics

- Manifest records the checkpoint hash, split hash, Ridge lambda, pseudo-cold
  item partitions, reward weights, policy epoch, displacement cap, and all
  validation selection rules.
- Assert that pseudo-cold target edges are absent from the training user graph
  and from target-excluded histories.
- Report policy residual norm, action/end rates, each reward component, and
  per-item cold/hot deltas.
- Compare against a zero-information uniform cold-bias frontier at matched hot
  cost, reusing the existing Ridge audit convention.

## Expected implementation boundary

Create a dedicated pilot runner and focused tests. Keep the selected main
backbone and existing Ridge exporter unchanged until the pilot passes the
matched-hot gate. Do not add the pilot arm to the paper's `Ours` row before a
multi-seed replication.
