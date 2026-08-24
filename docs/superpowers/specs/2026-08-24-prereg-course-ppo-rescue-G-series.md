# Pre-registration: the G series, last mechanical levers for course PPO

Written 2026-08-24, before any G run exists. Predecessors:
`2026-08-23-prereg-ppo-simulated-interaction-component-ablation.md` (blocks A-E,
the E3 replication) and `2026-08-24-prereg-centered-course-ppo-main-model.md`
(the F series). Both are settled; their verdicts are inputs here, not open
questions.

## Settled inputs (do not relitigate)

**1. The E3 / `ppo_core` replication FAILED.** Five fresh blind seeds
3030-3034, `ppo_core - random_policy` on test `cold_N@10`:
+0.001866 / -0.000577 / +0.000179 / +0.001565 / -0.000123 = **3/5** against a
registered bar of >= 4/5. mean +0.000582, sd 0.001074, sign-test p = 0.500.
Pooled with 2025-2029: 7/10, mean +0.000465, p = 0.172. Recorded honestly: the
verdict turns on seed3034's -0.000123, which is 2.2x the 5.7e-5 floor. The
predecessor prereg forbids re-scoping, so the FAIL stands and that family is
closed.

**2. `ridge_norm_only` gains exactly +0.00000000 with `delta* = 0.000` on 10/10
seeds.** The displacement gain is not a norm/hubness effect. Settled.

**3. The F series identified and fixed two objective defects, and still did not
clear the bar.** Best config F2 (centered, entropy 1e-3, scale 0.5): vs
centroid 5/5 mean +0.000820 (t = +2.85 > 2.776, genuinely significant at n=5);
vs greedy 3/5 mean +0.000641; vs random **3/5 mean +0.000187, sd 0.000710**.
Required n at 80% power: centroid ~5, greedy ~28, **random ~113**.

**4. Mechanism checks that did fire.** Training-time `course=` term, seed2025
last epoch: absolute -0.020218 -> centered **-0.000084** (240x). Entropy weight
is non-monotone: 0.01 -> arm 0.220706, 1e-3 -> **0.221063**, 1e-4 -> 0.220456,
consistent with the measured fact that fully decisive selection is a
disadvantage here (`greedy_course_fit` +0.00496 < `random_policy` +0.00541).
Lowering entropy without centering buys +0.000080, i.e. nothing: the two fixes
are not independent and centering is the prerequisite.

## Lever dropped before running, and why

An earlier plan of mine proposed replacing the reward with the actual ranking
metric (change in cold N@10 per rollout batch), on the argument that policy
gradient does not need differentiability. **That lever is withdrawn here,
before any run, because it is not implementable on this data:**

* The policy never trains on real cold courses. `strict_cold_count = 102` but
  `pseudo_target_count = 95`, drawn from warm items with `train_pop <= 25`
  masked to look cold (`policy_train_ids` 76, `policy_val_ids` 19). A ranking
  reward would therefore be computed on pseudo-cold items.
* Pseudo-cold is measured on this project to rank designs **backwards**: three
  Junyi feature sets scored +0.019..+0.115 pseudo-cold and -0.007..-0.028 true
  cold under matched-hot, and the pseudo-cold champion was the true-cold worst
  (`pseudocold-cannot-rank-designs`).
* The alternative, rewarding on the 34 real validation cold courses, trains on
  exactly the items whose reuse already produces the arm's val->test drop, and
  splitting them 17/17 gives no power against a ~1e-3 effect with ~1e-3 sd.

This is itself a finding for the analysis section: the objective can only be
measured on items that do not exist in training, and the standard surrogate is
anti-correlated with it on this data.

## Run matrix, fixed now

Reference is the existing F2 (`outputs/xds_mooccube_centered_course_ppo/
F2_centered_ent1e3`), not re-run. All G configs: MOOCCube, seeds 2025-2029,
`--ckpt-root outputs/graph_knp_final`, `--ppo-arms
ridge_ppo_course_reward_only`, `--course-reward-mode centered`,
`--ppo-entropy-weight 0.001`, `--reward-geometry cosine`, `--step-penalty 0.0
--no-end-action`, all four nulls present in every run.

| config | change vs F2 | lever attacks |
|---|---|---|
| `G1_scale2` | `--course-reward-scale 2.0` | signal is smaller than the entropy term even after centering (course -0.0002 vs embed +0.0009) |
| `G2_scale8` | `--course-reward-scale 8.0` | same, further |
| `G3_aligned` | `--max-steps 5 --step-size 0.004`, delta grid capped at 0.02 | train/deploy budget mismatch (rollout \|d\|/delta* was 11.1x); E1 tested alignment only under the broken objective |
| `G4_noepochsel` | `--no-epoch-selection` | selection asymmetry: 90 combinations on 34 val cold courses vs the nulls' 15 |

Seeds **3035-3039 are reserved** and must not be touched by any G run. Blind
seeds 3030-3034 are spent on the E3 replication and are no longer virgin for
this family.

## Criteria, fixed now

Floor 5.7e-5, same as predecessors. Within-run paired differences only.

**Primary, unchanged bar:** a config qualifies only if its PPO arm beats **all
three** zero-training nulls (`random_policy`, `centroid_step`,
`greedy_course_fit`) on **>= 4/5 seeds**. Beating `centroid_step` alone does not
qualify; `greedy_course_fit` consumes the same course signal with zero
training, and `random_policy` isolates whether the learned choice beats uniform
sampling.

**Reported for every config regardless of outcome:** per-seed differences, mean,
sd, sd/mean, sign-test p, leave-one-seed-out mean range, and the n required for
80% power against each null.

**G4 is registered as an integrity correction, not a rescue.** It is expected to
lower the arm. If G4 lowers the arm and it still qualifies, that is the
strongest available result. If G4 lowers the arm below the nulls, the G4 number
supersedes F2's in any write-up, because F2's selection budget is the one the
nulls do not get.

## Outcomes, decided now

* **Any config qualifies** -> exactly one config (highest minimum margin across
  the three nulls) is confirmed on fresh blind seeds 3035-3039 at the same
  >= 4/5 bar. Only after that may course PPO be described as a validated
  component. Even then it is a component: `random_policy - ridge_base` is
  +0.0054 and `ridge_base` is +14.5% cold over tuned CGRC, so the main-table
  headline does not come from the policy and must not be written as if it does.
* **No config qualifies** -> the course-PPO-as-main-model route closes. Main
  table becomes ridge + per-item-centered readiness. Course PPO appears only in
  an analysis section built from the F0-F4 defect-isolation ladder, the
  mechanism measurements above, and the dropped-lever finding.
* **No third config set, no re-scoping, no moving the floor.** If the G series
  fails, the next action is writing, not another matrix.

## Reproducibility gates, fixed now

* **G0 (blocking).** `--no-epoch-selection` and `--ppo-entropy-weight` were
  added to `ridge_course_reward_rl_pilot.py` today. Before any G run is
  interpreted, F2 seed2025 is re-run under the new code with identical flags and
  every arm compared to the stored result. Threshold: max |diff| on
  `cold_N@10` across all seven arms < 1e-12. The `float(1.0)` finding in the
  predecessor prereg is why this is measured rather than assumed.
* Code SHA of `ridge_course_reward_rl_pilot.py` and
  `ckg_rl_usim_v32_clean.py` recorded before the matrix starts and not changed
  during it.
* Test is loaded only after all selection, as in every predecessor run.

## OUTCOME (2026-08-24, 20/20 runs complete)

**No config qualifies. Registered consequence applies: the
course-PPO-as-main-model route closes.**

| config | arm cold | vs base | vs random | vs centroid | vs greedy |
|---|--:|--:|--:|--:|--:|
| F2 (ref, scale 0.5) | 0.221063 | +0.005598 | +0.000187 3/5 | +0.000820 5/5 | +0.000641 3/5 |
| `G1_scale2` | 0.220732 | +0.005266 | -0.000145 2/5 | +0.000488 3/5 | +0.000310 3/5 |
| `G2_scale8` | 0.220259 | +0.004793 | -0.000618 1/5 | +0.000015 2/5 | -0.000163 2/5 |
| `G3_aligned` | 0.218997 | +0.003531 | -0.000108 2/5 | -0.000372 1/5 | -0.000268 1/5 |
| `G4_noepochsel` | 0.220314 | +0.004848 | -0.000563 1/5 | +0.000070 3/5 | -0.000109 2/5 |

Three readings, all stronger than a null result.

**1. Amplifying the course signal monotonically degrades the metric, and the
mechanism probe proves the policy was following it.** Training-time `course=`
term vs scale: 0.5 -> **-0.000218**, 2.0 -> **+0.000002**, 8.0 -> **+0.005690**.
The centered term averages ~0 under uniform selection, so a positive value means
the policy is systematically choosing above-average-compatibility candidates —
it *is* optimizing course fit, increasingly so. Over the same sweep arm
`cold_N@10` falls 0.221063 -> 0.220732 -> 0.220259 and vs-random goes +0.000187
-> -0.000145 -> -0.000618. **Both columns monotone: the better the policy gets
at satisfying the course-knowledge criterion, the worse the ranking gets.** This
is now the causal version of what `greedy_course_fit` (+0.00496) losing to
`random_policy` (+0.00541) showed correlationally, and of the non-monotone
entropy result. Three independent measurements agree.

**2. F2's edge was a selection-budget artifact, as pre-registered.** F2's
`selected_epoch` is [5, 2, 2, 1, 2] — four of five seeds pick a non-final epoch,
which no null can do. `G4_noepochsel` forces [5, 5, 5, 5, 5] and the arm drops
0.221063 -> 0.220314 (-0.000749), vs-random +0.000187 -> **-0.000563**. Per the
commitment above, **the G4 numbers supersede F2's.** In particular F2's one
statistically significant result (vs centroid, 5/5, t = +2.85) does not survive:
G4 vs centroid is +0.000070 at 3/5. The single surviving positive claim of the F
series dissolves when the selection budget is equalized.

**3. Budget alignment costs displacement gain.** `G3_aligned` is the worst arm at
+0.003531 over base vs F2's +0.005598, and loses to all three nulls (2/5, 1/5,
1/5). E1 found this under the broken objective; it reproduces under the fixed
one, so it is a property of the mechanism, not of the old reward.

**Honest bottom line: with the selection budget equalized, the course PPO arm is
below the uniform-random null.** Seeds 3035-3039 are NOT consumed and the
confirmation run is not triggered, because nothing qualified for it.

## G0 measured, recorded before any G run (2026-08-24)

F2 seed2025 re-run under the new code, all flags identical, compared to the
stored result across 302 metric cells / 7 arms / 2 splits:

* **All ten primary metrics differ by exactly 0.000e+00**, including
  `cold_N@10` (0.1961294949 both). The registered gate PASSES.
* Three fields differ: `matched_hot_cold_vs_ridge_bias` 8.95e-08,
  `hot_N@20` 6.72e-09, `overall_N@20` 6.34e-09.

**Those three were then shown to be run-to-run nondeterminism, not the code
change.** A second run of the *same new code with the same flags* differs from
the first by `matched_hot_cold_vs_ridge_bias` 8.95e-08, `cold_N@20` 6.21e-09,
`overall_N@20` 6.57e-10, while `cold_N@10` stays 0.000e+00. The matched-hot
field is absent from the run2-vs-stored diff entirely, i.e. its value alternates
between runs — the signature of nondeterministic reduction order, consistent
with N@20 (more accumulation terms) moving while N@5/N@10 do not.

Two consequences, fixed now:

1. The G0 gate in `run_course_ppo_G_series.sh` is evaluated on the primary
   metrics only. Excluding the three derived fields is forced by measurement,
   not a relaxation.
2. **`matched_hot_cold_vs_ridge_bias` must not be quoted beyond ~1e-7.** The
   5.7e-5 decision floor is ~640x this noise, so sign counts are unaffected, but
   any matched-hot delta reported at 1e-8 precision in earlier notes is spurious
   at that digit.
