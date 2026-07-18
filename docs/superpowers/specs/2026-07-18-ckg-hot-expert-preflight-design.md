# CKG-RL Hot-Expert Preflight Design

## Goal

Before adding any CBI adapter, test whether an isolated, CKG-integratable graph warm expert can recover strong Hot performance on the strict MOOCCube split. This is a feasibility gate, not a main-table result.

## Question being tested

The current CKG-RL/TDInit route has strong Cold performance but weak Hot ranking. The preflight tests only the missing premise for the proposed dual-route model:

> Can a graph collaborative expert trained from the same training graph provide a strong Hot score bank under the existing strict split and full-catalog item-macro evaluator?

If the answer is no, a CBI cold adapter must not be added or tuned.

## Scope

- Seed: 2025 only.
- Split: existing `strict_item_cold_balanced_thr1_seed_2025` artifacts; never regenerate a split.
- Train data: `static_train.pkl` only.
- Model: a fresh, isolated graph warm expert with graph-ranking and warm pseudo-cold edge-reconstruction objectives. It uses the repository's CGRC-style graph primitives as an implementation reference, but it trains from scratch, is logged as a CKG integration preflight, and never loads a CGRC checkpoint.
- Evaluation: validation during training; final test is disabled for this preflight unless the validation gate passes and the user explicitly requests its release.
- Main-table files and existing CBI outputs remain untouched.

## Explicit exclusions

The preflight must not enable:

- CBI/TDInit content deltas or any item-specific cold parameter;
- CBI user simulation, PPO, course rewards, reranking, or course-sampling bias;
- external checkpoint vectors or score-level fusion;
- test-set checkpoint or hyperparameter selection;
- reconstructed Cold edges injected into the final evaluation graph.

## Model boundary

The warm expert produces one normalized user bank and one normalized Hot-item bank from `R_train`. Its pseudo-cold reconstruction head is trained only on warm training courses that are masked in its reconstruction view. True strict-Cold course IDs never enter an ID embedding route at training or inference. The first run has a content-only Cold fallback solely so that mixed-bank metrics can be exported; that fallback is diagnostic and is not a candidate Cold method.

This first run evaluates the warm graph route itself. It records Cold metrics only as a diagnostic; it does not claim a Cold solution yet.

## Validation protocol

At each saved epoch, export full-catalog item-macro validation metrics for Cold, Hot, and count-weighted Overall. Checkpoint selection is a fixed Hot-capacity score because the preflight deliberately has no CBI Cold branch; Overall is recorded but cannot be used as the final-method gate at this stage.

The preflight passes only if all of the following hold:

1. The graph route is evaluated with the audited strict split, train-history mask, full catalog, and item-macro aggregation.
2. Validation Hot R@10 is at least `0.2219` and Hot N@10 is at least `0.1442`. These are the seed-2025 CGRC validation values (`0.2269`/`0.1492`) minus a pre-registered absolute tolerance of `0.005`.
3. The output manifest, split hashes, and protected-file hashes verify cleanly.

The reference is a validation-only CGRC reproduction for the matching seed, never a test result. If it does not pass, stop this branch and investigate the warm expert before implementing any CBI component. A strong Overall guard begins only in the next frozen-expert-plus-adapter stage, where both Hot and Cold have valid representations.

## Reproducibility and artifacts

The launcher writes a fresh output/checkpoint/log root and a manifest containing:

- split hashes and source hashes;
- all training knobs and the validation reference;
- per-epoch validation metrics;
- protected-file hashes before and after execution;
- an explicit `test_evaluation=false` flag.

## Follow-on decision

Only a passing preflight unlocks the next separate experiment: frozen warm expert plus a shared content adapter evaluated on item-level pseudo-cold courses. That experiment will be separately specified and will begin without simulation, PPO, graph edge write-back, or course rewards.
