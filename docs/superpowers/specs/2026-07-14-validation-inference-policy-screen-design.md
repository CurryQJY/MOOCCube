# Validation Inference-Policy Screen Design

## Goal

Using the three frozen recovered main-table checkpoints, compare validation cold item-macro performance under five inference policies without retraining or reading test metrics for selection.

## Fixed experimental boundary

- Checkpoints: seeds 2025, 2026, and 2027 from `main_table_51ea12fc_candidate`.
- Target data: the strict-cold validation fold only; MOOCCube has 34 validation cold courses per seed.
- Training history: train-only, matching the strict protocol.
- Full catalog, train-history masking, course-macro aggregation, and `K={5,10,20}` remain unchanged.
- Simulator horizon `T=5`, residual scale `1.0`, candidate construction, course knowledge, and state transition remain unchanged.
- Checkpoints are loaded read-only and no parameter update is allowed.

## Policies

1. `static`: no rollout.
2. `ppo`: trained Actor logits with argmax action selection.
3. `greedy_similarity`: choose the candidate most similar to the current course state.
4. `course_fit`: choose the candidate with the highest course-fit score.
5. `random`: choose a random candidate with fixed evaluation RNG seed 7001.

The random arm is a deterministic screening reference, not a variance estimate. If it is competitive, a later random-seed replication is required.

## Validation targeting

The evaluation wrapper delegates the original strict split construction, then routes the original validation dataframe to the final evaluation slot. It preserves the original training dataframe, validation dataframe, split metadata used to construct the checkpoint, train-only user history, and item train-popularity. The original test dataframe is never passed to the ranking evaluator.

Every output manifest and audit file must declare `evaluation_target=validation` and record the policy and evaluation RNG.

## Selection rule

Select the policy with the highest three-seed mean validation cold item-macro NDCG@10. Use validation cold item-macro Recall@10 only as a tie-breaker. Hot and interaction-weighted values may be retained as diagnostics but do not affect selection.

## Interpretation

- PPO best: supports using the learned Actor as the inference policy for this frozen PPO-trained representation model.
- Greedy or course-fit matches PPO: rollout/state transition may matter more than learned PPO action selection.
- Random matches PPO: the result does not support a PPO-policy claim and random-seed replication is required.
- Static best: keep current static inference and do not promote test-time rollout.

Because all checkpoints were trained with PPO trajectories, this screen answers which inference policy works best for the frozen model. It does not replace a full training-policy ablation.

## Outputs and tests

- Per-seed full-ranking validation cold item-macro CSVs for all policies.
- A combined policy-by-seed table and a three-seed mean/std ranking table.
- Call-path audits showing zero rollout calls for static and non-zero rollout calls for every rollout policy.
- Tests proving validation is routed to evaluation, test rows are excluded, PPO uses argmax, fixed random is repeatable, checkpoint writes remain blocked, and the selection table uses only cold item-macro NDCG@10/Recall@10.
