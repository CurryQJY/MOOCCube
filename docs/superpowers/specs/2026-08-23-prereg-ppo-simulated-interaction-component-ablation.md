# Pre-registration: component ablation of the PPO simulated-interaction module

Written 2026-08-23, **before** any run in this batch executes. Registered because
this project has six recorded instances of "positive result -> build narrative ->
run the control much later -> mechanism dies" (memory: `run-null-controls-before-
narrative`), and because the immediately preceding batch established that the
headline ablation `ppo - base` is real (5/5, +0.0068) while `ppo - random` is not
(2/5, +0.0015) -- i.e. an ablation table for this module is easy to over-sell.

## Question

Which components of the PPO simulated-interaction module carry measurable
credit on strict item-cold MOOCCube, once each removal is compared against the
zero-training displacement nulls in the same table?

## Scope, fixed now

Dataset MOOCCube (`processed_data_hin_clean_pop5`, `static_item_cold_balanced`),
reward geometry `cosine` (the fixed geometry), alpha 1.0, seeds **2025-2029**,
and **test is exported in every run** (no `--skip-test`). All other flags copied
from `outputs/xds_mooccube_cosine_7arm/seed2025/run_manifest.json`.

Test export is mandatory here, not optional: the previous batch measured that all
four PPO arms lost -0.0010..-0.0041 going val->test while the zero-training arms
moved +0.0002/-0.0003, so a validation-only ablation table is not reportable.

## Run matrix

**Block A -- component grid, one run per seed, all eight arms sharing one Ridge
fit and one pseudo-cold split** (so arm differences cannot come from the fit):

| arm | displacement | learned policy | course bias in policy | course term in reward |
|---|---|---|---|---|
| `ridge_base` | none | - | - | - |
| `ridge_centroid_step` | deterministic centroid | no | no | no |
| `ridge_random_policy` | uniform random candidate | no | no | no |
| `ridge_greedy_course_fit` | course argmax | no | yes | no |
| `ridge_ppo_core` | PPO | yes | no | no |
| `ridge_ppo_course_bias` | PPO | yes | **yes** | no |
| `ridge_ppo_course_reward_only` | PPO | yes | no | **yes** |
| `ridge_ppo_full` | PPO | yes | **yes** | **yes** |

**Block B -- reward-term ablation, `ridge_ppo_core` only, one run per config per
seed.** The training reward is
`embed_gain + recommendation_gain - step_penalty + course_reward`; Block A varies
only the last term, so Block B removes the other three and the geometry:

* B1 `--embedding-reward-weight 0` -- drop the target-progress term
* B2 `--recommendation-reward-weight 0` -- drop the positive-user score term
* B3 `--step-penalty 0.0` -- drop the per-step cost (removes the incentive to END)
* B4 `--reward-geometry euclidean` -- the pre-fix norm-sensitive geometry

B1/B2 require two new weight flags. They must default to 1.0 and reproduce the
current runs bit-for-bit; that is a gate below, not an expectation.

**Block C -- structural knobs, `ridge_ppo_core` only.** Registered because these
two were never ablated and were previously identified as setting the whole
family's ceiling:

* C1 `--candidate-count 5`, C2 `--candidate-count 50` (baseline 20)
* C3 `--max-steps 1 --step-size 0.25`, C4 `--max-steps 10 --step-size 0.025`
  (both hold `max_steps x step_size = 0.25`, the baseline displacement budget)

45 runs total (5 + 20 + 20).

## Metrics, fixed now

PRIMARY: **test** `cold_N@10` minus `ridge_base` in the same run.
CO-PRIMARY: **test** `matched_hot_cold_vs_ridge_bias` -- cold N@10 read at
matched hot N@10 against the uniform-bias frontier. This is the only comparison
that is not confounded by cold/hot exposure trade-off, which on this task is
zero-sum (memory: `cold-exposure-is-zero-sum`).
SECONDARY, reported not gating: validation `cold_N@10`, per-seed spread (sd),
mean/sd, `end_rate`, `active_steps`, `rollout_delta_l2`, selected delta/epoch.

Reported as a per-seed table plus mean/sd/sign-count. Single-seed means are not
reportable on their own -- the previous batch's `+0.0077` on seed2026 turned out
to be an outlier at 5 seeds with sd 2.6x the mean.

## Criteria, fixed now

For a component X removed in arm `A_minus_X` relative to the arm that has it:

1. **Component-contributes** requires the removal to hurt the PRIMARY on
   **>= 4 of 5 seeds**, AND the with-X arm to beat **all three** zero-training
   nulls (`ridge_centroid_step`, `ridge_random_policy`, `ridge_greedy_course_fit`)
   on the PRIMARY on **>= 4 of 5 seeds**.
2. If (1) fails on the null comparison but passes on the removal comparison, the
   row is reported as **"displacement-attributable, not learning-attributable"**
   and must be printed next to the null arms. It may not be described as an
   ablation of a learned component.
3. Any per-seed difference with `|diff| < 5.7e-05` is reported as
   indistinguishable from zero, not as a sign. (3x the 1.9e-05 same-code rerun
   drift measured on val cold N@10 in
   `2026-08-23-prereg-ppo-vs-random-null-5seed.md`. If measured test drift turns
   out larger, the larger floor is used and the change is recorded here.)
4. Sign counts are reported for every row, including rows that fail. No row is
   dropped from the table for being null.

## Reproducibility gates, fixed now

* G1 `ridge_base` test and validation `cold_N@10` must be **bit-identical across
   every run of the same seed** (Ridge fit depends only on seed and data). Any
   drift invalidates the cross-block comparisons and the batch is rerun.
* G2 With `--embedding-reward-weight 1 --recommendation-reward-weight 1` (the new
   defaults), a rerun must reproduce `outputs/xds_mooccube_cosine_7arm/seed2025`
   `ridge_ppo_core` validation `cold_N@10` to <= 1.9e-05.
* G3 Existing run directories are never overwritten; this batch writes to
   `outputs/xds_mooccube_ppo_component_ablation/`.

## Outcomes, decided now
* If no component passes criterion (1): the module is written up as a **diagnosed
  negative** -- the ablation table shows displacement matters and no learned or
  knowledge-derived component inside it does. This is the currently expected
  outcome given `displacement-direction-carries-no-information`, and it is
  reportable as an analysis section, not as a main-table contribution.
* If some component passes (1): the claim is MOOCCube-scoped only until Junyi and
  COCO are rerun with the same matrix, and must state the previous cross-dataset
  contradiction (course signal: MOOCCube 2/3, Junyi 2/3 below floor, COCO 1/3).
* Either way the table ships with all three null arms and both val and test
  columns.

## Deviations recorded during execution
Appended as they happened. Nothing here changes a metric, a criterion, or a
threshold; each entry is a change to what gets written to disk.

* **2026-08-23 23:2x, mid-batch: `arm_diagnostics` added to the runner's JSON
  output.** While Block A was running it turned out the runner computed each
  arm's selected delta and each PPO arm's rollout statistics and then discarded
  them -- only the PPO arms' delta reached the manifest. Since delta is the
  exposure knob previous batches identified as the only variable that moves this
  family, a table without it is not readable. The added block is write-only JSON.
  Consequence: Block A's five runs all predate it (verified: none of the five
  `pilot_results.json` files contain the key, so v1 is internally consistent), so
  Block A is rerun as `blockA2` under the current code. v1 vs v2 must agree
  bit-for-bit on every arm; that is now an extra reproduction check, reported
  with the results.
* **2026-08-23, two new CLI flags** `--embedding-reward-weight` and
  `--recommendation-reward-weight` (default 1.0), required by B1/B2. Gate G2 was
  checked before B/C launched: Block A `seed2025` reproduces
  `outputs/xds_mooccube_cosine_7arm/seed2025` on validation `cold_N@10` with
  diff exactly 0.000e+00 for `ridge_base`, `ridge_greedy_course_fit`,
  `ridge_random_policy`, and `ridge_ppo_core`. Covered by
  `tests/test_reward_term_weights.py` (8 tests).
* **`end_rate` and `active_steps` are not recoverable for Block A.** The runner
  never persisted them anywhere before the change above -- not in the JSON, not
  in the eval bundle, not in the log -- so for Block A the displacement
  diagnostic is the per-epoch `rollout_delta` line in `run.log` at the selected
  epoch. Blocks B/C and `blockA2` carry the full `rollout_stats`.
* **B/C runs prune their 102MB `*_selected_eval.pt` after completion** (the run
  volume was at 99% with 15GB free at batch start). Test metrics are computed
  in-run and already in `pilot_results.json`; the 339KB policy checkpoint is
  kept, so a B/C arm can still be replayed.

## Addendum: Block D -- inside the course reward

Registered 2026-08-24, **before Block D runs**. Blocks A and B are complete at
this point and their numbers are known; Block D asks a question none of them
answered, and its criterion is the one already fixed above, unchanged.

### Why

Block B reached the two top-level reward terms and the step penalty, and Block A's
2x2 reached the course reward as a single on/off unit. But the course reward is a
weighted sum of four signals -- concept overlap, prerequisite order, difficulty
fit, redundancy -- and no run in this batch varies them. A reward ablation that
stops at "course reward: on/off" is not a reward ablation of the thing that is
actually in the paper, which is the knowledge-derived reward.

The runner was already built for this: the observable candidate bias is
deliberately rebuilt at fixed reference weights (concept .04, prereq .08,
difficulty .03, redundant .02) so that changing the reward weights changes *only*
the scalar training reward and not what the policy can see. That is the correct
leave-one-out isolation and it is why this block is cheap.

The base arm is `ridge_ppo_course_reward_only`, not `ridge_ppo_core`: the weights
are inert unless the arm's `use_course_reward` is true. Batch weights in use are
concept 0.04, prereq 0.08, difficulty 0.03, **redundant 0.0** (already off), and
`course_reward_scale` 0.5.

### Runs, 5 configs x seeds 2025-2029

* D0 `ridge_ppo_course_reward_only` alone, weights unchanged -- **gate G4**, below
* D1 `--course-concept-weight 0` -- leave out concept overlap
* D2 `--course-prereq-weight 0` -- leave out prerequisite order
* D3 `--course-difficulty-weight 0` -- leave out difficulty fit
* D4 `--course-redundant-weight 0.02` -- switch the one signal the batch leaves off
  back **on**, at the reference weight. Recorded as an addition, not a removal;
  it is here because "we set it to zero" is not evidence that it does nothing.

### Gate G4, fixed now

D0 must reproduce Block A's `ridge_ppo_course_reward_only` on test `cold_N@10` to
<= 1.9e-05 at every seed. Block A trains four arms in one process with
`_seed_everything(seed)` before each; D0 trains that arm alone. If arm order
matters, the leave-one-out rows cannot be read against Block A and D1-D4 are
rescored against D0 instead. Reported either way.

### Criterion, unchanged

Criterion 1 above, applied with `ridge_ppo_course_reward_only` as the
with-component arm: a signal counts only if removing it hurts test cold N@10 on
>= 4/5 seeds AND the arm carrying it beats all three zero-training nulls on
>= 4/5 seeds. The second half is already measured to fail for every PPO arm in
Block A, so the honest prediction is that D1-D4 can at most produce
"displacement-attributable" rows. Registering that prediction now so a positive
D-row cannot be re-narrated later as a knowledge-reward win.

## Measured reproducibility, recorded 2026-08-24

Written after Blocks A/A2/B/C and D0 landed, before Block D was scored.

**Gate G1 as literally written (bit-identity) FAILS; at the 5.7e-05 floor it
passes.** Across the 12-13 runs per seed that share one code version:
validation `ridge_base` cold N@10 is *exactly* identical at every seed, and test
`ridge_base` differs by at most **5.7e-07** (seed2027; 1.5e-08 at seed2025, 0.0
elsewhere). That is float nondeterminism in the test evaluation path, ~1000x below
the floor. Both readings are reported rather than redefining the gate after the
fact.

**Gate G4 PASSES exactly (0.0e+00 at all five seeds).** D0 trains
`ridge_ppo_course_reward_only` alone; Block A2 trains it third of four arms in one
process. They agree bit-for-bit, so arm order does not matter and Block D is read
against Block A2 as planned. Same-code cross-process PPO training is therefore
exactly reproducible on this machine.

**`blockA` is superseded by `blockA2` and excluded from the identity pool.**
`blockA` seed2025 started 23:05:09 and `ckg_rl_usim_v32_clean.py` was edited at
23:07:51; it is the only run in the batch built from the pre-edit engine (blockA's
other four seeds started 23:11:03+). Of the 40 arm-seed cells, 37 reproduce at
exactly 0.0 between the two Block A passes. The three that do not:

| cell | \|diff\| test cold N@10 | reading |
|---|--:|---|
| `ridge_ppo_course_reward_only@2025` | 1.835e-04 | pre/post-edit engine |
| `ridge_greedy_course_fit@2029` | 9.504e-08 | eval-path float noise |
| `ridge_base@2025` | 1.463e-08 | eval-path float noise |

The first cell is the substantive one, and it is worth recording as a method fact:
**multiplying a reward tensor by `float(1.0)` is not bit-neutral.** The added
weights default to 1.0 and `x*1.0` is exact in IEEE754, but the extra `Mul` node
reorders gradient accumulation in the backward graph, and after 5 PPO epochs one
arm-seed lands 1.8e-04 away on test at the *same* selected epoch (5) and delta
(0.03). So "default preserves behaviour" is true of the forward value and false of
the trained policy. Anything in this family that reports differences at the 1e-4
scale needs the code version pinned, not just the seed.

Consequence for this batch: the main grid is `blockA2`, which is internally one
code version, and every B/C/D run postdates both edits. The 5.7e-05 floor stands
as pre-registered.

## Addendum: Block E -- the two structural defects, and the hubness null

Registered 2026-08-24, **before any Block E run**, after Block A2/B/C/D closed at
70 runs with 0 of 16 removal rows judged "component-contributes". Block E is the
**last** batch on this family. It exists because the closed batch surfaced two
defects that are mis-specifications rather than knob settings, plus one surviving
positive claim that has never had its own null.

### Defect A -- the training objective is evaluated at a point 6-12x away

`ridge_course_reward_rl_pilot.py:1424` sets `rollout_cap = max(delta_grid)`, so in
`delta_grid` mode the engine's `max_delta` is 0.25 no matter what
`--fixed-max-delta` says. The PPO reward is therefore computed on trajectories of
norm <= 0.25, while validation then selects a deployment delta of 0.02-0.03 in
12 of 20 arm-seed cells (measured, `blockA2`: rollout `|d|` 0.16-0.24 against
`delta*` 0.02-0.03). `project_displacement` keeps direction and discards
magnitude, and magnitude is the only dimension shown to carry signal. **The policy
optimises an operating point it is never evaluated at.**

Because the cap is `max(delta_grid)`, aligning the two needs no engine edit: shrink
the grid and shrink `step_size` so that `max_steps x step_size = max(delta_grid)`.
This is deliberate -- see the `float(1.0)` finding above; no new arithmetic is
added to the reward path.

### Defect B -- END plus step_penalty makes the no-op an optimal solution

`step_penalty=0.01` is charged per active step unconditionally while cosine
`embed_gain` is bounded and small, so ENDing at step 0 locks in reward 0 whereas
moving risks negative reward. Measured in `blockA2`: **8 of 20 arm-seed cells have
`end_rate > 0.88` with `active_steps` 0.11-0.64**, and on seed2028
`ridge_ppo_course_reward_only` and `ridge_ppo_full` reach `end_rate = 1.000`,
`delta* = 0.000`, and are **bit-identical to `ridge_base`** on both val and test
(diff 0.0e+00). Two independent rows already point at this: B3 (removing the step
penalty) is the largest single effect in the batch and its sign says "removing it
helps"; C3 (1 step x 0.25, no room to END early) is the best PPO variant measured.

### Defect C -- the one surviving positive claim has no null of its own

The only row in the closed batch that is negative on >= 4/5 seeds in the
"removal hurts" direction is `displacement (whole simulated interaction)`
(-0.00290, 4/1). Every existing null (`random_policy`, `centroid_step`,
`greedy_course_fit`) is **per-item**: each cold row retrieves its own candidates
and moves along its own direction. So none of them can separate "displacing this
row toward these users" from "increasing this row's inner product with everyone".
Two strictly weaker nulls are added:

- `ridge_global_shift` -- **one** direction for every cold row (the normalised
  mean of the whole user bank). No per-item retrieval at all.
- `ridge_norm_only` -- **no direction change whatsoever**: the cold row is scaled
  radially, so `delta = c * init` and the grid delta controls the norm increase.
  This is the purest available hubness null.

### Run matrix, fixed now

Seeds 2025-2029. Reward geometry cosine, ridge alpha 1.0, test exported in-run
after all policy selection, exactly as Block A2. Every config carries
`ridge_base`, the three existing nulls, the two new nulls, and `ridge_ppo_core`.

| config | step_size | max(delta_grid) | budget | END | step_penalty | fixes |
|---|--:|--:|--:|:-:|--:|---|
| `E0_hubness_nulls` | 0.05 | 0.25 | 0.25 | on | 0.01 | none -- anchors the new nulls at the `blockA2` operating point |
| `E1_aligned_noend` | 0.004 | 0.02 | 0.02 | off | 0.0 | A + B |
| `E2_aligned_only` | 0.004 | 0.02 | 0.02 | on | 0.01 | A |
| `E3_noend_only` | 0.05 | 0.25 | 0.25 | off | 0.0 | B |

`blockA2`'s `ridge_ppo_core` is the fourth cell of the 2x2 (neither fix) and is
not re-run.

E1/E2 use a different delta grid from `blockA2`, so cross-config absolute
comparison is confounded by the grid. **Every criterion below is therefore a
within-run paired comparison against arms from the same run and the same seed.**

### Criteria, fixed now

1. **RL route stays open** only if some E config's `ridge_ppo_core` beats that
   same run's `ridge_random_policy` on test cold N@10 by more than the 5.7e-05
   floor on **>= 4 of 5 seeds**. If no config clears this, the RL/displacement
   route is closed permanently and no further batch is run on it.
2. **Hubness verdict.** If `ridge_norm_only - ridge_base` is positive beyond the
   floor on **>= 4 of 5 seeds** in `E0_hubness_nulls`, the displacement gain
   contains a pure-norm component. If in addition `ridge_norm_only`'s mean gain
   over `ridge_base` is **>= 50%** of `ridge_random_policy`'s mean gain over
   `ridge_base`, the displacement family is reported as **substantially a hubness
   artifact**, and the `displacement (whole simulated interaction)` row loses its
   status as evidence for a learned or targeted mechanism.
3. Sign floor 5.7e-05 and per-seed sign counts as in the main criteria. Seed sd
   is reported on every row.
4. **No post-hoc redefinition.** These thresholds are not revisited after the
   numbers are seen. A config that fails (1) is reported as failing, not
   re-scoped.

### Outcomes, decided now

- If (1) fails and (2) fires: the family is written up as a negative result whose
  mechanism is named (hubness mean shift), and the manuscript's simulated-
  interaction contribution claims are removed, not softened.
- If (1) fails and (2) does not fire: still a negative result; the displacement
  gain is real but not learnable, and the write-up says exactly that.
- If (1) passes: the winning config is re-run at 5 fresh seeds before any claim
  is written, because a 4/5 pass at this effect size is not yet a result.

## Block E result, and the replication it triggers (recorded 2026-08-24)

Gate G5 passed at 1.463e-08 worst -- the END-masking edit is bit-neutral on the
default path (`ridge_ppo_core` reproduced blockA2 at exactly 0.0e+00 on all five
seeds; the 1.5e-08 was `ridge_base` eval noise).

**Criterion 2 did not fire, and it refuted the hypothesis that motivated it.**
`ridge_norm_only` gains **exactly +0.00000 on all 5 seeds** with `delta* = 0.000`
-- validation could not find any radial scaling worth taking. So the displacement
gain is **not** a norm/hubness mean shift. The hubness reading carried over from
`kg-cold-only-refuted-by-null-ladder` does not apply here and is withdrawn.

What the ladder does show: `ridge_global_shift`, **one constant vector applied to
every cold course**, earns +0.00380 (5/5) -- 70% of `ridge_random_policy`'s
+0.00541 and **more than the trained `ridge_ppo_core`'s +0.00290 (4/5)**. So the
mechanism is mostly a global rank-1 direction, with a minority contribution from
per-item variation, and training recovers less of it than uniform sampling does.

**Criterion 1 passed for exactly one config: `E3_noend_only` (fix B only).**

| config | fixes | mean vs own random null | sd | +/-/~ | end_rate | \|d\|/delta* |
|---|---|--:|--:|:-:|--:|--:|
| `blockA2` | neither | -0.00251 | 0.00324 | 1/4/0 | 0.399 | 2.4x |
| `E1_aligned_noend` | A + B | -0.00018 | 0.00059 | 2/3/0 | 0.000 | 1.3x |
| `E2_aligned_only` | A only | -0.00148 | 0.00163 | 1/4/0 | 0.374 | 0.8x |
| `E3_noend_only` | **B only** | **+0.00035** | 0.00070 | **4/1/0** | 0.000 | 11.1x |

Both fixes fired mechanically as intended (end_rate 0.399 -> 0.000 for B; the
train/deploy ratio 2.4x -> 1.3x/0.8x for A). The diagnosis was therefore half
right: **defect B was the binding constraint and fixing it flips the sign; defect
A is a real mis-specification but fixing it does not help, and combining A with B
is worse than B alone.** E3 still carries an 11.1x train/deploy gap and wins
anyway, which says the gap was not what was costing the policy.

E3's absolute test cold N@10 is 0.2212, the highest PPO figure measured anywhere
in this program (previous best C3 0.2210, random null 0.2209).

### The registered consequence

The pre-registered outcome for a passing criterion 1 reads: *"the winning config
is re-run at 5 fresh seeds before any claim is written, because a 4/5 pass at this
effect size is not yet a result."* That clause was written before these numbers
existed and it binds here, because **the sd is 2.0x the mean** -- the same shape
that made the earlier `ppo - random` claim collapse from 5/5 to 2/5
(`displacement-direction-carries-no-information`).

`E3_noend_only` is therefore re-run at seeds **2030-2034**, config unchanged, with
the same within-run pairing against `ridge_random_policy`. Registered now, before
those runs execute:

* **Confirmed** only if `ridge_ppo_core - ridge_random_policy` is positive beyond
  the 5.7e-05 floor on **>= 4 of 5 fresh seeds**. Pooled across all 10 seeds it
  must also stay positive.
* **Not confirmed** at <= 3/5 fresh seeds. In that case the 4/5 on 2025-2029 is
  reported as a seed-set artifact, the family is closed as planned, and the
  write-up says a fix appeared to work at 5 seeds and did not replicate.
* No third seed set. No re-scoping of the config. No moving the floor.

## Replication of E3, part 1: the three blind confirmatory seeds (2026-08-24)

**Deviation, recorded before the numbers below were interpreted.** The registered
clause says "5 fresh seeds". Fresh seeds are not free here: the pilot needs a
per-seed backbone checkpoint, and `outputs/graph_knp_final` only holds 2025-2029
(the first launch at 2030-2034 died in 3 seconds per seed with
`FileNotFoundError: outputs/graph_knp_final/seed2030/best.pt`).

What does exist is `outputs/graph_knp_confirmatory_source/`, a blind confirmatory
seed set built 2026-08-20 for seeds **3030/3031/3032** under protocol
`graph-course-core-confirmatory-source-blind-v1`, with `skip_test: true` and
`test_files_hashed: false`. Its `source_config` matches `graph_knp_final` item for
item: epochs 60, batch 2048, n_layers 2, prereq_aux_weight 2.0, embedding_dim 128,
lr 0.001, delta_ref 0.25. These are stronger evidence than seeds built today,
because they predate every Block E number.

E3's config unchanged, `--ckpt-root outputs/graph_knp_confirmatory_source`:

| seed | ppo_core | random_policy | **ppo - random** | base | global_shift | norm_only | end_rate | delta* |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 3030 | 0.2208 | 0.2190 | **+0.00187** | 0.2146 | 0.2181 | 0.2146 | 0.000 | 0.020 |
| 3031 | 0.2630 | 0.2636 | **-0.00058** | 0.2584 | 0.2614 | 0.2584 | 0.000 | 0.020 |
| 3032 | 0.2401 | 0.2399 | **+0.00018** | 0.2363 | 0.2391 | 0.2363 | 0.000 | 0.020 |

**2 of 3 positive beyond the floor, mean +0.00049.** Pooled over all eight seeds
(2025-2029 plus these): 6/8 positive, mean **+0.00040**, so the pooled-positive
half of the criterion holds.

The registered bar is **>= 4 of 5 fresh seeds**. At 3 seeds the observed 2/3 is
below that proportion, but the test is **not yet decided**: 2/3 plus two positives
would be exactly 4/5. Stopping here and calling it either way would be scoring a
5-seed criterion on 3 seeds, which is the move this prereg exists to prevent.

**Criterion 2's refutation, by contrast, replicated exactly.** `ridge_norm_only`
equals `ridge_base` to the printed precision on all three blind seeds as well
(0.2146/0.2146, 0.2584/0.2584, 0.2363/0.2363), with `delta* = 0.000`. That is now
**8 of 8 seeds where pure radial scaling buys exactly nothing**. The displacement
gain is not a hubness/norm effect, and that conclusion is settled.

### Part 2, registered now before it runs

Two more fresh seeds, **3033 and 3034**, to complete the registered five. Splits
come from `gen_balanced_split.py`, which carries a `--validate-against` mode that
regenerates an existing seed in memory and compares order-sensitive per-split
`_row_id` sequences; that gate is run on seed2025 first and part 2 is abandoned if
it does not reproduce. Backbones are trained with the confirmatory `source_config`
above.

Building these after seeing the 3-seed result does not bias the outcome -- the
split RNG and the backbone training are not under my control, and this prereg
already forbids a third seed set, re-scoping, or moving the floor. The commitment
is unchanged and restated: **confirmed only at >= 4/5 fresh seeds; otherwise the
4/5 on 2025-2029 is reported as a seed-set artifact and the family closes.**

## RETRACTION: the "NOT CONFIRMED 3/5" verdict is void (2026-08-24)

I reported E3's replication as **NOT CONFIRMED at 3/5** using seeds 3030-3034.
The two seeds I built today were subsequently renamed on disk to
`seed3033_INVALID_floortol0.003_ckpt` and `seed3034_INVALID_floortol0.003_ckpt`
(parent directory mtime 13:53, after part 2 finished at 13:23:44). Nothing in this
repository emits that string -- `grep -rn "INVALID_floortol" --include=*.py
--include=*.sh` returns nothing -- so the mark came from outside the run. **The
verdict that used them is withdrawn.**

What I can verify about those two runs:

* **The floor-tolerance part of the label does not match what I measure.**
  `graph_knp_consistent.py:157` defaults `--floor-tol` to 0.003 and I did not pass
  it, so 0.003 was the expected value -- but the run log says
  `[select] overall floor=0.2269` against an identity `overall R@10 = 0.2419`, and
  0.2419 - 0.2269 = **0.015**, not 0.003. Both manifests also record the same
  `delta_rule: ... identity - 0.015`. On this axis my backbones match the blind
  protocol.
* **The provenance part of the label is a fair objection.** The blind set's
  `source_preflight.json` pins `ckg_rl_usim_v32_clean.py` at sha
  `66965314e8e932673deebdfae9a4c93ba05a2e6de8bd0268777d16eedcd3770e` (2026-08-20).
  My two backbones were trained today, after the END-masking edit, so their code
  state is not the pinned one. G5 showed that edit is bit-neutral on the default
  path and backbone selection uses the deterministic `course_fit` arm, but "should
  not matter" is exactly the claim the `float(1.0)` finding in this same prereg
  says not to accept without measurement. **Unpinned code is sufficient grounds to
  void them.**
* Every recorded field in `seed3033/run_manifest.json` matches `seed3030`'s
  (n_layers, prereq_aux_weight, epochs, batch_size, emb_dim, lr, delta_ref,
  delta_grid, delta_rule, skip_test, selection). There is also no `run.log` for
  3030-3032 in that directory, so their logs cannot be compared against mine.

### Standing evidence after the retraction

Fresh blind seeds only (3030/3031/3032): `ppo_core - random_policy` =
**+0.00187, -0.00058, +0.00018** -> **2 of 3 positive**, mean +0.00049. Against
the registered `>= 4 of 5` bar this is **still undecided**, exactly where it was
before part 2: 2/3 plus two positives would be 4/5.

To finish the registered test properly, the two extra backbones must be rebuilt
with `ckg_rl_usim_v32_clean.py` checked out at sha `66965314...`, matching the
blind protocol. Until then E3's replication is **incomplete, not failed**, and no
verdict may be quoted either way. The criteria themselves are unchanged.

### Unaffected by the retraction

Criterion 2's refutation stands on the 8 valid seeds (2025-2029 plus 3030-3032):
`ridge_norm_only` gains exactly +0.00000 with `delta* = 0.000` on **8 of 8**, on
all four metrics. The displacement gain is not a hubness/norm effect.
