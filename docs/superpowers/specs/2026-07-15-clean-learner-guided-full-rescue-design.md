# Clean Learner-Guided Full Rescue Design

## Objective

Replace the internally inconsistent PPO-based Full with a minimal learner-guided cold-item adaptation method that uses no cold-item test interactions, has identical train/inference refinement semantics, and can be validated before the AAAI deadline.

## Protected assets

- Do not overwrite any existing main-table output or checkpoint.
- Keep `backups/main_code_pre_cleanup_20260714_105010/` and `docs/repair_snapshots/2026-07-15-coursefit-pseudocold-pre-second-stage.md` as reconstruction sources.
- Implement the rescue in a new Python entry point and new run scripts.
- Select configurations only with validation cold item-macro N@10. Read test only after the configuration is frozen.

## Method

The clean Full contains four components:

1. A content-aware two-tower recommender.
2. A fixed item-level pseudo-cold training set with ID, history, and auxiliary-alignment leakage blocked.
3. A non-negative learner-fit gate for selecting representative historical learners.
4. A bounded T-step residual update applied only to true/pseudo-cold items.

The method contains no Actor, Critic, PPO loss, entropy objective, terminal-MSE reward, course reward, prerequisite auxiliary ranking loss, SAGE, CGRC, PAAC, or LLM score.

## Learner-fit

For candidate learner `u` and course `i`, define a positive base signal from normalized representation similarity and concept demand:

`base = clamp(0.5 * (cos(h_i, e_u) + 1) + w_c * concept_match, 0, 1)`.

Educational constraints act as gates rather than subtractive penalties:

`fit = base * exp(-beta_p * prereq_gap) * exp(-beta_d * difficulty_gap) * (1 - redundancy)`.

All terms are in `[0,1]`; therefore `fit` is finite and non-negative. The target item and every fixed pseudo-cold item are removed from learner histories before all fit terms are computed.

## Bounded update

At step `t`, select the highest-fit learner not used earlier in the same episode. For rows with `fit <= min_fit`, perform no update and mark the row stopped. Otherwise:

`direction = normalize(e_u - h_t)`

`step = min(step_cap, eta * fit) * direction`

`h_{t+1} = project_total_displacement(h_t + step, h_0, total_cap)`.

Stopped rows remain unchanged in later steps. The update is deterministic. Training and inference call the same refinement function.

## Training objective

The initial rescue uses recommendation loss only:

`L = L_rec`.

The old InfoNCE auxiliary loss is disabled (`AuxWeight=0`) because its weighted contribution currently exceeds the recommendation loss. A single validation-only alternative with `AuxWeight=0.05` is permitted after the clean Full is established.

## Screening matrix

All screening uses MOOCCube seed 2025 and validation selection:

| Arm | T | Learner policy | Aux | Purpose |
|---|---:|---|---:|---|
| A | 0 | none | 0 | clean training baseline |
| B | 1 | learner-fit | 0 | one-step feedback |
| C | 3 | learner-fit | 0 | proposed Full |
| D | 3 | random valid learner | 0 | selection causal control |
| E | 3 | learner-fit | 0.05 | weak auxiliary check |

Run 15-epoch quick screens first. Continue an arm to 35 epochs only if its best validation N@10 is within 0.02 of the best arm. Run 60 epochs only for the best Full candidate and A. The Full must reach at least the established T=0 reference `0.3058` and exceed the clean T=0 arm by at least `0.003` before seeds 2026/2027 are authorized.

## Required diagnostics

Log per epoch:

- update activation ratio by step;
- stopped ratio by step;
- selected fit mean and quantiles;
- step displacement mean/max;
- total displacement mean/max;
- repeated-user rate, expected zero;
- pseudo-cold count and effective ratio;
- recommendation, auxiliary, and total losses.

Reject a run if any fit/displacement is non-finite, repeated-user rate is nonzero, warm rows change, or the source/config lock changes on resume.

## Paper claims allowed after validation

If C beats A and D consistently, the paper may claim that interpretable learner selection and bounded multi-step refinement improve strict cold-course recommendation. It must not claim PPO or long-horizon reinforcement learning. If C fails to beat A, the refinement is removed from Full and retained only as a negative diagnostic.

