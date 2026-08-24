# CKG-RL V3.4 Rank-Reward-Only Control Design

## Objective

Isolate the failed V3.3 component. V3.4 retains the working V3.2 vector-only content generator and changes only the USIM policy reward to V3.3's incremental teacher-panel rank gain. It is a causal diagnostic, not a new paper method or a score fusion.

## Frozen Controls

The following must be behaviorally identical to clean V3.2 seed 2025:

```text
outer strict-cold split and H_train/H_val/H_test/C_val/C_test visibility
H_G, P_train, P_val item partitions
CleanTeacher fit, checkpoint selection, and teacher item/user tables
ContentGenerator architecture, vector MSE/cosine objective, H_G-only labels,
internal H_G vector-loss checkpoint selection, and generator hyperparameters
full-ranking evaluator, train-only history, catalog course metadata, and hot bank
```

The V3.4 manifest must record the teacher/generator stage hashes and state `generator_rank_loss=false`. After the run, the generator hash is compared with V3.2 seed 2025; matching hashes establish that the cold-generator input is held fixed.

## Changed Variable

V3.4 constructs deterministic frozen teacher user panels only for `P_train` and `P_val`, after the vector-only generator checkpoint is fixed. The policy keeps V3.2 legal state-retrieval candidates and target-free inference, but replaces the legacy embedding/positive-score reward by:

```text
r = KL(q_T || q(h_t)) - KL(q_T || q(h_{t+1}))
    + lambda_course * r_course
    - step_penalty - lambda_delta * ||h_{t+1} - h_t||_2
```

The panel target, teacher item vector, and pseudo-item interaction are available only during training/validation. Inference sees no panel target, positive user, teacher target item, residual candidate, or reward computation.

## Selection and Falsification

Epoch 0 remains the vector-generator identity baseline. A PPO epoch is eligible only if it retains hot validation and has non-negative `P_val` rank gain. Eligible epochs are selected by `C_val` NDCG@10, Recall@10, then P_val rank diagnostics. Outer test is read only after selection.

Interpretation is predeclared:

- If V3.4 generator hash matches V3.2 and cold metrics recover, V3.3 collapse was caused by generator rank distillation.
- If the selected PPO epoch improves cold over epoch 0 with positive P_val gain, rank-guided USIM rollout remains viable.
- If epoch 0 is selected or PPO has no material cold gain, the rollout has no demonstrated paper-level contribution and USIM should not remain the main mechanism.

## Required Artifacts

`control_manifest.json`, `rank_panel_manifest.json`, `generator_vector_epochs.csv`, `policy_rank_epochs.csv`, `final_metrics.json`, and per-item hot/cold exports are required. The first and only planned run is isolated seed 2025.
