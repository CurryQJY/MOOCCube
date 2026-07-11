# RecPPO Research-Grade Repair Design

## Scope

Repair only `usim_feedback_fast3_content_delta_repaired.py` and its repaired runner/tests. The legacy entrypoint remains an unchanged comparison baseline. ContentDelta implementation is explicitly out of scope and is disabled by default in the repaired entrypoint so RecPPO experiments are attributable.

## Training Architecture

Training has two phases controlled by a repaired model epoch hook:

1. Warm supervised phase: train the existing warm user/item/content backbone with the RL residual and PPO loss disabled.
2. RecPPO phase: freeze every warm-model parameter, keep the resulting behavior embedding space fixed, and optimize only the RecPPO actor/critic and learned stop candidate with a dedicated optimizer.

The RecPPO optimizer owns all policy parameters and performs an optimizer step inside every PPO epoch. The legacy outer optimizer excludes those parameters. This makes old log-probabilities genuinely old after the first PPO step and makes clipping and `ppo_epochs` effective.

## MDP And Supervision

The action set contains retrieved users plus a learned stop action. Per-row active masks support early termination. Continuing pays a step cost; stopping receives zero continuation reward, so the policy can learn trajectory length.

The observed interaction user is injected only during training and used as first-step behavior supervision. Later steps use `ignore_index`, avoiding the false assumption that one positive interaction is a repeated multi-step demonstration.

## Reward

The reward is action-dependent and uses:

- normalized embedding-distance improvement;
- local candidate-ranking error improvement, weighted by target-candidate relevance and logarithmic rank discount;
- optional course terms already available from train-only histories;
- a continuation step cost.

Global batch duplicate/coverage statistics remain diagnostics only and are not included in reward.

## Inference And Evaluation

Training candidates may be sampled. Deterministic inference selects the highest-scoring retrieved candidates without `multinomial`. The evaluation item bank is cached, and positive target scores are read from the same bank used for negatives. Repeated inference for the same checkpoint and input must be identical.

## Defaults And Selection

Repaired defaults use partial tail pseudo-cold simulation instead of all-item masking. The repaired runner defaults to a longer schedule and patience that cover both warm and RL phases. Validation uses a repaired guarded score that combines cold Recall/NDCG while preventing catastrophic hot collapse. Setting `PpoLossWeight=0` disables the phase transition, policy updates, and inference residual, providing a real supervised ablation.

## Reproducibility

The repaired runner fixes `PYTHONHASHSEED` and CUBLAS configuration before process start; the entrypoint explicitly seeds the imported legacy pipeline and enables strict deterministic PyTorch algorithms. The repaired manifest records the repaired entrypoint hash and every RecPPO default. Epoch diagnostics include actor/critic loss, behavior CE, terminal value loss, KL, clip fraction, entropy, reward moments, and stop rate. Checkpoint payloads preserve the dedicated RecPPO optimizer state.

## Verification

Tests must demonstrate that:

- PPO epochs cause parameter changes and non-trivial probability ratios;
- deterministic inference is repeatable;
- positive full-ranking vectors come from the cached bank;
- stop actions end trajectories and mask later transitions;
- behavior CE applies only to the first step;
- warm parameters freeze at the RecPPO transition;
- the repaired manifest identifies the repaired script.
