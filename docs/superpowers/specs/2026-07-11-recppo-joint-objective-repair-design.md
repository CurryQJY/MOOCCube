# RecPPO Joint Objective Repair Design

## Scope

Change only the repaired RecPPO entrypoint, its repaired PowerShell wrapper, and
the repaired test suite. The legacy method remains the comparison baseline.
The strict item-cold split and train-only history protocol are unchanged.

## Joint Training

The warmup phase trains the supervised recommender only. After warmup, each
training batch performs two independent updates:

1. the dedicated RecPPO optimizer updates actor, critic, and stop head from a
   detached rollout;
2. the existing outer optimizer updates the recommender backbone from main,
   auxiliary, and prerequisite losses.

The two optimizers own disjoint parameter sets. RecPPO actions remain
non-differentiable inputs to the supervised forward pass, so policy gradients
cannot leak through the recommender loss.

## Train-Positive Listwise Reward

For every warm training interaction, the observed training user is the positive
user. Hard negatives are mined from all users with non-empty training histories,
excluding every user who interacted with the item in the training split. The
reward is the reduction in listwise cross-entropy between the positive user and
the globally mined hard negatives. No validation or test interaction is read.

Hard-negative IDs are cached only within an epoch. The cache is invalidated at
every epoch boundary because joint training changes the user and item embedding
banks.

## Reward Balance And Supervision

The positive-vs-hard-negative rank gain is the principal reward. Behavior-target
embedding gain is retained as a weak shaping term. Course prerequisite, concept,
difficulty, and redundancy terms are combined, scaled, and clipped before being
added, preventing a single structural penalty from dominating the return.

First-step behavior CE starts at 0.20 and linearly decays to 0.02 over the first
10 PPO epochs. Stop remains a zero-reward terminal action; continuation pays the
step cost. With the repaired reward, a negative continuation is therefore worse
than stopping.

## Selection And Diagnostics

The repaired early-stop minimum improvement defaults to zero so every genuine
validation improvement can replace the checkpoint. Diagnostics record effective
behavior CE weight, embedding gain, rank gain, scaled course contribution, and
the existing PPO stability statistics.

## Verification

Tests must prove that the supervised backbone receives gradients during the PPO
phase, RecPPO parameters remain excluded from the outer optimizer, hard negatives
come from train-history users and exclude item positives, a positive-directed
transition produces positive rank gain, CE decays to its configured floor, the
course contribution is bounded, and epoch boundaries invalidate hard-negative
caches.
