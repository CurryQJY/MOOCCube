# RecPPO Global Listwise Reward Design

## Scope

Change only the repaired RecPPO entrypoint, its tests, diagnostics, and repaired runner controls. Keep the legacy method, course reward, ContentDelta default, pseudo-cold protocol, optimizer, schedule, and evaluation unchanged.

## Training Boundary

The listwise teacher may use only warm-item behavior embeddings learned from the training split and users with non-empty train histories. Validation/test interactions never enter the user pool, teacher scores, cache, reward, or checkpoint selection inputs. Users that also occur in validation/test remain eligible only through their train-history embeddings.

## Reward

At the first PPO use of a warm training item, retrieve the target embedding's Top-K users from the normalized full training-user bank and cache their user IDs for the frozen PPO phase. For the cached Top-K support, form a discounted teacher distribution from target-item scores and compare it with the previous and next state distributions using listwise cross-entropy. The rank reward is the decrease in listwise cross-entropy:

`rank_gain = CE(teacher, previous_state) - CE(teacher, next_state)`.

Positive gain therefore means that the transition makes the refined item reproduce the target behavior ranking more closely. The cache is cleared when entering the PPO phase and is never populated during the moving-backbone warmup phase.

## Efficiency And Diagnostics

Top-K retrieval is performed once per unique warm item during the frozen PPO phase and cached as user IDs. Training-user embeddings are gathered from the existing detached epoch user bank. Diagnostics report the listwise gain, cache hit rate, pool size, and explicitly identify the rank reward source as `global_train_user_topk`.

## Verification

Tests must prove that validation/test-only users are excluded, global users outside the local action candidates can affect reward, a transition toward the teacher ordering yields positive gain, cache reuse is deterministic, and the existing course reward configuration remains unchanged.
