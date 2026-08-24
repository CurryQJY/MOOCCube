# Pre-registration: centered course reward as the main-model PPO route

Written 2026-08-24, before any run in this matrix. Companion to
`2026-08-23-prereg-ppo-simulated-interaction-component-ablation.md`, whose
`ppo_core` clause closed the same day at 3/5 on the five blind seeds
(3030-3034). This document does not reopen that clause. It registers a
*different* arm set — the course-reward arms — against two objective defects
that were measured today and never varied in any prior batch.

## Question

With the two defects below repaired, does a course-reward PPO arm beat **all
three** zero-training displacement nulls in its own run, on test `cold_N@10`?
Only a yes makes course PPO admissible as the paper's main model.

## Defect 1 — the course reward is a near-constant penalty, not a reward

Measured from `outputs/xds_mooccube_ppo_component_ablation/blockA2/seed2025/run.log`,
training-time per-step reward means, all five epochs:

| term | `ppo_core` | `ppo_course_reward_only` | `ppo_full` |
|---|--:|--:|--:|
| embed | +9.0e-4 | +9.3e-4 | +8.8e-4 |
| rec | +2.0e-4 | +1.5e-4 | +1.8e-4 |
| **course** | 0.000000 | **-1.75e-2** | **-1.72e-2** |
| step | -8.8e-3 | -8.8e-3 | -8.7e-3 |

The course term is ~15x the two positive terms combined, negative, and varies
by only ~9% across epochs (-0.0166 .. -0.0181). A near-constant reward carries
almost no action-discriminating signal once the critic absorbs it, and before
that it makes stopping optimal — which is the measured `end_rate > 0.88` in
8 of 20 cells of the earlier batch.

Mechanism: `compat = 0.04*concept_bonus - 0.08*prereq_gap - 0.03*difficulty_gap`.
The largest weight sits on a penalty, and a strict-cold course's candidate users
have typically seen none of its prerequisites, so `prereq_gap ~ 1` and
`compat ~ -0.08`, scaled by `course_reward_scale 0.5` to ~-0.04. Same order as
the measured -0.0175.

**The fix already exists and has never been run.** `--course-reward-mode
centered` wires `course_reward_baseline_fn -> course_signal.candidate_reward_baseline`,
subtracting a candidate-set baseline so only the "which candidate is better"
component survives. Audit of the whole repository: 19 shell scripts, every one
passes `--course-reward-mode absolute`; 153 of 153 run manifests that record a
mode record `absolute`; **zero** record `centered`.

Independent support for centering as the operative mechanism, from the
readiness component of the same paper (`outputs/e13f_hotfloor_junyi.json`):
uncentered `V1` gives -0.003795 (1/3 seeds) on Junyi while per-item-centered
`V3` gives +0.017284 (3/3). Removing the constant item-level offset is what
separates passing from failing the permutation null there.

## Defect 2 — the entropy bonus outweighs the reward signal ~27x

`loss = actor + value_weight*critic + terminal_weight*terminal - entropy_weight*entropy`,
with `ppo_entropy_weight = 0.01` (`ckg_rl_usim_v32_clean.py`), never varied in
any prior batch. With 20 candidates and END masked, max entropy is
`ln(20) = 2.996`, so the entropy term reaches 0.030 while the measured signal is
~1.1e-3. Consistency check on the observed `train_loss = -0.0188` at epoch 5:
critic and terminal terms are non-negative and `|actor| <~ 1e-3` at that reward
scale, so `H >= 1.78`, i.e. at least 59% of maximum. **The optimum of this
objective is the uniform policy, which is literally the `ridge_random_policy`
null.** That is the simplest explanation for `ppo_core ~ random_policy` within
+/-0.0005 across eight seeds.

`--ppo-entropy-weight` was added to `ridge_course_reward_rl_pilot.py` today;
its default stays 0.01 so every prior run reproduces unchanged.

## Scope, fixed now

MOOCCube only. `--ridge-alpha 1.0`, `--retention-reference ridge`,
`--reward-geometry cosine`, `--no-end-action`, `--step-penalty 0.0`,
`--candidate-count 20`, `--max-steps 5 --step-size 0.05`, `--hot-tolerance 0.003`,
course weights `concept 0.04 / prereq 0.08 / difficulty 0.03 / redundant 0.0`,
`--course-reward-scale 0.5`, `--course-bias-scale 0.2`, 15-point delta grid
`0 .0025 .005 .0075 .01 .015 .02 .03 .04 .05 .075 .1 .15 .2 .25`,
`--policy-epochs 5 --policy-batch-size 8 --policy-lr 0.0003`.

Every run emits the three displacement nulls plus the two degenerate rungs:
`--with-random-policy-arm --with-centroid-step-arm --with-global-shift-arm
--with-norm-only-arm`.

**Stage 1 uses seeds 2025-2029** (`--ckpt-root outputs/graph_knp_final`).
The blind confirmatory seeds 3030-3034 are **reserved** for stage 2 and are not
touched in stage 1.

## Run matrix, fixed now

| config | arm | mode | entropy |
|---|---|---|--:|
| `F0_absolute_ref` | `ridge_ppo_course_reward_only` | absolute | 0.01 |
| `F1_centered` | `ridge_ppo_course_reward_only` | centered | 0.01 |
| `F2_centered_ent1e3` | `ridge_ppo_course_reward_only` | centered | 0.001 |
| `F3_centered_ent1e4` | `ridge_ppo_course_reward_only` | centered | 0.0001 |
| `F4_absolute_ent1e4` | `ridge_ppo_course_reward_only` | absolute | 0.0001 |

`F0` re-runs the current caliber so stage 1 contains its own reference.
`F4` isolates defect 2 from defect 1: if `F4` matches `F3`, centering is not
what mattered; if `F4` fails and `F3` passes, both are needed.

5 configs x 5 seeds = 25 runs.

## Metrics, fixed now

Primary: test `cold_N@10`, within-run paired differences of the PPO arm against
`ridge_random_policy`, `ridge_centroid_step`, `ridge_greedy_course_fit`.
Reported alongside, never instead of: the paired difference against
`ridge_base`, plus `hot_N@10` and `overall_N@10`, plus matched-hot on the
densified 10-point frontier. Floor for calling a per-seed difference nonzero:
**5.7e-5**, unchanged from the companion prereg.

## Criteria, fixed now

**Stage 1 qualification.** A config qualifies iff its PPO arm beats **all three**
nulls on **>= 4 of 5** seeds beyond the floor. Beating only `random_policy` does
not qualify — `greedy_course_fit` consumes the identical course signal with zero
training and is the harder, more honest reference.

**Stage 1 negative outcome.** If no config qualifies, course PPO is not
admissible as the main model. Recorded outcome in that case: the main table
reverts to the ridge + readiness composition, and the course-PPO material becomes
an analysis section documenting why an objective whose optimum is the uniform
policy cannot be repaired by reward engineering.

**Stage 2 confirmation.** Only the single best-qualifying stage-1 config is run
on the blind seeds 3030-3034, same bar: **>= 4/5 against all three nulls**.
Confirmed only if stage 2 passes. If two configs tie in stage 1, the one with
fewer deviations from the historical default is promoted.

**No third config set. No re-scoping. The floor does not move.** A stage-1 pass
followed by a stage-2 failure is reported as a seed-set artifact, exactly as the
`ppo_core` clause was.

## Provenance, disclosed now

`graph_knp_consistent.py`, `ridge_course_reward_rl_pilot.py` and
`ckg_rl_usim_v32_clean.py` are **untracked by git** in this repository
(`git ls-files` covers 3735 files and excludes them), so the shas pinned in
`outputs/graph_knp_confirmatory_source/source_preflight.json`
(`ckg_rl_usim_v32_clean.py = 66965314...`, `graph_knp_consistent.py = bbffd7d7...`)
**cannot be checked out.** Current shas are recorded at run time in each
`run_manifest.json` and must be quoted in any write-up. No claim of
byte-identical reproduction of the 2026-08-20 blind protocol may be made.

## Outcomes, decided now

* All three nulls beaten >= 4/5 in stage 1 **and** stage 2 -> course PPO is the
  main model; ablation reports the three null contrasts in the same table.
* Stage 1 passes, stage 2 fails -> seed-set artifact; family closes.
* Stage 1 fails -> family closes; material becomes an analysis section.

## Stage 1 result (recorded 2026-08-24): NO CONFIG QUALIFIES

25 runs, `outputs/xds_mooccube_centered_course_ppo/`, all five configs at 5/5
seeds. Manifests verified against the registered matrix: `course_reward_mode` and
`ppo_entropy_weight` match row for row, with `no_end_action=True` and
`step_penalty=0.0` throughout.

### The mechanism gate passed -- centering is not inert

Training-time `course=` term, seed2025, all five epochs:

| config | epoch 1..5 |
|---|---|
| `F0_absolute_ref` | -0.020362 -0.020088 -0.019985 -0.020046 -0.020218 |
| `F1_centered` | -0.000111 +0.000054 +0.000157 +0.000200 -0.000084 |
| `F2_centered_ent1e3` | -0.000111 +0.000051 +0.000071 +0.000142 -0.000218 |
| `F3_centered_ent1e4` | -0.000111 +0.000070 +0.000046 +0.000142 -0.000200 |
| `F4_absolute_ent1e4` | -0.020362 -0.020068 -0.019969 -0.019957 -0.020194 |

Centering turns a -0.0203 near-constant penalty into a **sign-changing** signal
two orders of magnitude smaller. Defect 1 is real and the fix does what it claims.
`--course-reward-mode centered` has now been run for the first time.

### But the registered bar is not met

Paired within-run differences on test `cold_N@10`, PPO arm minus each null:

| config | vs random | vs centroid | vs greedy | qualifies |
|---|--:|--:|--:|:-:|
| `F0_absolute_ref` | -0.00068 (1/5) | -0.00005 (1/5) | -0.00023 (2/5) | no |
| `F1_centered` | -0.00017 (2/5) | +0.00046 (3/5) | +0.00028 (3/5) | no |
| **`F2_centered_ent1e3`** | **+0.00019 (3/5)** | **+0.00082 (5/5)** | **+0.00064 (3/5)** | **no** |
| `F3_centered_ent1e4` | -0.00042 (2/5) | +0.00021 (3/5) | +0.00003 (3/5) | no |
| `F4_absolute_ent1e4` | -0.00060 (1/5) | +0.00003 (2/5) | -0.00015 (2/5) | no |

**Stage 1 fails.** The registered bar is all three nulls at >= 4/5; the best
config clears one of three.

Three things the matrix does establish, reported because they were measured:

1. **Centering is the operative fix, not the entropy weight.** `F4`
   (absolute + entropy 1e-4) behaves like `F0` (1/5, 2/5, 2/5), so lowering
   entropy alone changes nothing. This is exactly the `F3`-vs-`F4` discriminator
   registered in advance.
2. **Entropy has an interior optimum at 1e-3.** `F2` (1e-3) beats both `F1`
   (1e-2) and `F3` (1e-4) against all three nulls, so the answer was not
   "less entropy is better".
3. **`random_policy` is the binding constraint.** `F2` reaches 5/5 against
   `centroid_step` -- the only 5/5 in the matrix -- and 3/5 against both `random`
   and `greedy`. Per-seed vs random: -0.00075, -0.00032, +0.00073, +0.00034,
   +0.00094 (mean +0.00019, sd 0.00071, sd 3.7x the mean).

`F2` absolute figures, 5-seed means: cold 0.2211, hot 0.1447, overall 0.1528,
matched-hot +0.0017, `delta*` 0.0210, against `random_policy` 0.2209 / 0.1447 /
0.1528 / +0.0015 / 0.0180 in the same runs. cold 0.2211 is the highest PPO number
measured anywhere in this program, and the margin over the training-free null is
2e-4 with a per-seed sd of 7e-4.

### Registered outcome, executed

Per "Stage 1 negative outcome": **course PPO is not admissible as the paper's main
model.** The main table reverts to the ridge + readiness composition. The
course-PPO material becomes an analysis section documenting that centering repairs
the reward's constant-offset defect and still lands inside the uniform-policy
neighbourhood.

**Stage 2 is NOT run.** Only a qualifying stage-1 config was licensed to touch the
blind seeds 3030-3034, and none qualifies. Those seeds remain unused by this
matrix. No third config set, no re-scoping, the floor did not move.

## Addendum: Block G -- the course-reward ablation, redone under `centered`

Registered 2026-08-24, **before any Block G run**, after stage 1 closed.

### Why this is not a repeat of Block D

Block D (`2026-08-23` prereg) already ran leave-one-out over the four course
signals and found all four null. **That measurement could not have detected
anything.** It ran under `--course-reward-mode absolute`, where the whole course
term is a -0.0203 near-constant penalty varying <2% across epochs. A constant
offset carries no action-discriminating signal, so its decomposition into
concept / prereq / difficulty / redundancy is a decomposition of a constant. The
null was a property of the regime, not of the sub-signals.

Stage 1 verified that `centered` removes the constant: the term becomes
sign-changing at +/-2e-4. Only now is a sub-signal ablation measuring a live
mechanism. This is the ablation the paper's `w/o educ. rewards` row decomposes
into, and it is the reason the earlier decomposition is not reportable.

### Reference and matrix

Reference is `F2_centered_ent1e3` itself, already at 5/5 seeds -- same script,
same 24 config fields verified identical (`ppo_arms
ridge_ppo_course_reward_only`, `centered`, entropy 0.001, `no_end_action`,
`step_penalty 0.0`, scale 0.5, weights .04/.08/.03/0.0, 15-point grid, seeds
2025-2029). No replication gate is needed because no new reference run is made.

The arm is `ridge_ppo_course_reward_only`, whose `use_course_bias` is False, so
the observable candidate bias is inert and these flags change **only the scalar
training reward and its own centering baseline** (`candidate_reward_baseline`
recomputes the candidate-set mean from the same weights, so zeroing a weight
stays internally consistent).

| config | change from `F2_centered_ent1e3` |
|---|---|
| `G1_wo_concept` | `--course-concept-weight 0.0` |
| `G2_wo_prereq` | `--course-prereq-weight 0.0` |
| `G3_wo_difficulty` | `--course-difficulty-weight 0.0` |
| `G4_with_redundant` | `--course-redundant-weight 0.02` (the batch runs it at 0.0, so this row switches it **on**) |
| `G5_wo_course_reward` | `--ppo-arms ridge_ppo_core` -- the whole course reward removed, everything else held. This is the direct analogue of the paper's `w/o educ. rewards` row, measured under the repaired reward. |

5 configs x 5 seeds = 25 runs. Seeds 2025-2029, `--ckpt-root
outputs/graph_knp_final`. Blind seeds 3030-3034 stay untouched -- stage 2 was
never licensed and this addendum does not license it either.

### Criteria, fixed now

1. **A sub-signal contributes** iff `variant - F2` on test `cold_N@10` is
   **negative** beyond the 5.7e-05 floor on **>= 4 of 5** seeds (removal hurts).
   For `G4` the sign is read in reverse, since it adds a signal rather than
   removing one.
2. **Mechanism gate, reported per config.** The training-time `course=` term must
   move relative to `F2`'s (-0.000111 / +0.000054 / +0.000157 / +0.000200 /
   -0.000084 at seed2025). A config whose term is unchanged is reported as
   **inert**, and its null result carries no information -- exactly the mistake
   Block D made. This gate is descriptive, not a pass/fail on the metric.
3. Every row ships with the three zero-training nulls from its own run. No row is
   dropped for being null. Seed sd is reported on every row.
4. This does **not** reopen stage 1. `F2` failed the main-model bar at 3/5
   against random and greedy; a Block G row cannot promote it. The purpose is to
   report *which knowledge component the repaired reward actually uses*, for the
   analysis section.

## Block G result (recorded 2026-08-24): all five rows null, but this time the null is informative

25 runs, all configs 5/5. Artifacts:
`outputs/xds_mooccube_centered_course_ppo/block_g_table.md`.

### Mechanism gate passed for every config

Training-time `course=` term, seed2025, epochs 1..5 -- none is inert:

| config | mean \|term\| | verdict |
|---|--:|---|
| `F2_centered_ent1e3` (reference) | 0.000119 | reference |
| `G1_wo_concept` | 0.000215 | moved |
| `G2_wo_prereq` | 0.000052 | moved |
| `G3_wo_difficulty` | 0.000143 | moved |
| `G4_with_redundant` | 0.000131 | moved |
| `G5_wo_course_reward` | 0.000000 | term fully removed |

This is the difference from Block D: there, every config decomposed a -0.0203
constant. Here every flag demonstrably changes a live, sign-changing reward.
`G2_wo_prereq` cutting the mean magnitude by 57% (0.000119 -> 0.000052) also
confirms the prereq term is the largest contributor to the reward's magnitude,
exactly as the defect-1 analysis predicted -- it just does not convert to metric.

### The ablation, paired against the reference (negative = contributes)

| component | mean | sd | neg/pos/~ | verdict |
|---|--:|--:|:-:|---|
| w/o concept-overlap | -0.00010 | 0.00078 | 2/3/0 | no contribution |
| w/o prerequisite-order | -0.00018 | 0.00104 | 2/3/0 | no contribution |
| w/o difficulty-fit | +0.00003 | 0.00043 | 3/2/0 | no contribution |
| redundancy ON (reversed) | -0.00033 | 0.00054 | 2/1/2 | no contribution |
| **w/o the whole course reward** | **-0.00042** | 0.00079 | **3/2/0** | no contribution |

**No sub-signal reaches 4/5, and neither does removing the entire course reward.**
Every mean is within one sd of zero; the largest effect (-0.00042, the whole
reward) is half its own sd.

### What this settles

1. **The paper's `w/o educ. rewards` row cannot be filled honestly with a
   positive number, even after the reward is repaired.** Removing the whole
   course reward costs -0.00042 at 3/5 seeds -- indistinguishable from zero.
   Under `absolute` the same row was *negative* (removal helped, 4/5). Repairing
   the constant-offset defect moved it from "harmful" to "null", not to
   "contributes".
2. **The knowledge decomposition question is closed on this route.** Block D's
   all-null was uninformative; Block G's all-null is informative, because the
   mechanism gate proves each flag bit. Both agree, and now one of them counts.
3. **Consistent with stage 1.** The reference beats `centroid_step` 5/5 but
   `random_policy` only 3/5. Table 3 shows the same ordering under every ablation:
   removing course signal moves the arm *toward* random (`G5`: -0.00023, 2/5 vs
   random), i.e. the course reward does perturb the policy -- it just does not
   perturb it toward a better one.

No criterion moved, no config was re-scoped, blind seeds 3030-3034 remain
untouched. The registered stage-1 outcome stands: course PPO is not the main
model, and this material is an analysis section.

## Correction to the Block G write-up (recorded 2026-08-24)

The Block G section above claimed: *"Under `absolute` the same row was negative
(removal helped, 4/5). Repairing the constant-offset defect moved it from
'harmful' to 'null'."* **That comparison was not apples-to-apples and the claim is
withdrawn.**

`G5` removes the course reward from an arm that has **no** course bias. The
`absolute` figure quoted was `ppo_full - ppo_course_bias`, which has bias present.
The matching no-bias pair is `ppo_course_reward_only - ppo_core`. All three, in one
convention -- (with reward) minus (without reward), positive means the reward
contributes:

| regime | pair | mean | sd | positive | sd/\|mean\| |
|---|---|--:|--:|:-:|--:|
| absolute, no bias | `blockA2` `ppo_course_reward_only - ppo_core` | **+0.00040** | 0.00079 | **4/5** | 2.0 |
| absolute, with bias | `blockA2` `ppo_full - ppo_course_bias` | **-0.00090** | 0.00099 | 1/5 | 1.1 |
| centered, no bias | `F2 - G5` | **+0.00042** | 0.00079 | **3/5** | 1.9 |

Two corrected readings follow.

1. **Centering did not flip a negative into a null.** The like-for-like `absolute`
   row was already positive at 4/5. What centering changed was the reward
   mechanism -- a 100x magnitude change, from a -0.0203 constant to a
   sign-changing +/-0.0002 term -- while the metric effect stayed at +0.00040 vs
   +0.00042, a difference of 2e-6. **The metric is insensitive to whether the
   reward carries any discriminating signal at all.** That is a stronger statement
   than the one it replaces, and it is the finding worth reporting.
   The 4/5 vs 3/5 gap sits on a quantity whose sd is ~2x its mean; neither count
   should be called a pass.
2. **The sign is governed by whether the course bias is present, not by the
   centering.** Adding the reward on top of the bias costs -0.00090 at 4/5 -- the
   two channels are redundant or interfering. The paper's Full CKG-RL carries
   course-KG inputs, the KG sampler **and** the educational rewards, so its
   analogue is `ppo_full - ppo_course_bias`, the negative row. The +0.0004 positive
   exists only in a configuration that has also dropped the bias, which is not the
   paper's model.

Provenance note on an earlier number: the interim table read at the start of this
work reported this row as +0.00037 with seed2025 at +0.00030. That table was built
from `blockA`, whose seed2025 predates the engine edit documented in the companion
prereg; `blockA2` gives +0.00049 for that seed and +0.00040 overall. The 1.9e-4
gap is the already-recorded `float(1.0)` drift. **`blockA2` is the correct source
and the +0.00040 figure supersedes +0.00037.**

None of this changes a criterion or an outcome: the no-bias row fails the
null-comparison half of the stage-1 bar regardless of its 4/5 or 3/5 count, and
stage 1 remains closed.
