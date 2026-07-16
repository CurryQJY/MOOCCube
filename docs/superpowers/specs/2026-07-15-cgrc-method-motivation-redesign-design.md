# CGRC Method Motivation Redesign

## Objective

Replace the current dataset-level pedagogical-risk panel with a transfer
diagnosis that shows why a strong generic cold-start item recommender remains
insufficient after adaptation to courses and maps each observed gap to a
CKG-RL design response.

## Evidence Boundary

The motivation figure may use CGRC outputs and model-neutral course signals,
but it must identify CGRC as a generic cold-start transfer baseline rather
than a course recommender, and it must not use CKG-RL outputs. PCGNN supplies
the course-specific counterpart in the main table. CKG-RL-versus-CGRC
comparisons remain in RQ2 as post-hoc validation and boundary analysis.

## Baseline Diagnosis

Panel (a) retains the course-level CGRC NDCG@10 cumulative distribution over
204 strict course-cold units after adapting this generic method to MOOCCube.
It reports the observed tail rates at 0.05 and 0.10 and the CGRC Top-10
cold-course share.

Panel (b) compares ranks 1--10 with ranks 11--20 inside each frozen CGRC list.
Recommendation records are averaged by rank within each `(seed, target
course)`, then across the ten ranks in each bucket. Paired effects therefore
use 204 course-macro units rather than interaction-weighted rows.

The four signals use the model-neutral P1 definitions:

- prerequisite coverage gap, lower is better;
- directional concept continuity, higher is better;
- P95/Top-5 structural difficulty gap, lower is better;
- structural redundancy, lower is better.

The plotted favorable alignment effect is positive when CGRC's Top-10 bucket
is better than its ranks 11--20 bucket. Raw Top-10-minus-bottom-10 differences
remain in CSV output. Paired bootstrap intervals use 10,000 resamples and the
analysis seed 2027.

## Defensible Interpretation

The frozen data show that CGRC's Top-10 ordering improves concept continuity
and difficulty alignment, leaves prerequisite gap unchanged, and increases
structural redundancy. The paper must therefore describe an asymmetric domain
transfer gap: content-graph ranking captures semantic proximity and some
difficulty ordering, but its original generic-item objective does not
prioritize course prerequisites or structural redundancy.

The manuscript must not call CGRC a course recommendation model, present these
gaps as defects relative to CGRC's original task, claim that CGRC is weak on
every pedagogical signal, or claim that all risk proxies cause exposure
failure.

## Problem-To-Design Mapping

- Missing target-course interactions and warm-item competition motivate
  content anchoring, forced cold-ID masking, and learner-state refinement.
- Flat prerequisite alignment motivates prerequisite supervision and explicit
  prerequisite reward terms.
- Top-rank structural redundancy motivates redundancy-aware reward terms.
- Passive course knowledge motivates placing knowledge in sampling, rewards,
  and auxiliary supervision rather than only in the input representation.

## Artifacts

- `paper_aaai27/scripts/analyze_method_motivation.py`
- `paper_aaai27/figures/method_motivation_analysis/rank_course_macro.csv`
- `paper_aaai27/figures/method_motivation_analysis/bucket_course_macro.csv`
- `paper_aaai27/figures/method_motivation_analysis/rank_alignment_paired.csv`
- `paper_aaai27/figures/method_motivation_analysis/manifest.json`
- `paper_aaai27/figures/mooccube_method_motivation.{pdf,svg,png}`

## Acceptance Criteria

- Only CGRC rows enter the baseline alignment analysis.
- Every accepted `(seed, target course)` has ranks 1--20 and contributes one
  paired Top-10-versus-bottom-10 unit.
- Synthetic tests verify list-before-course aggregation and effect orientation.
- The real analysis returns 204 paired units and reproduces the frozen CGRC
  Top-10 means from the P1 audit.
- Figure labels state the favorable-effect direction and do not call P1 a
  motivation experiment.
- Introduction, RQ2, Discussion, and Conclusion preserve the distinction
  between baseline diagnosis and post-hoc method validation.
- The paper compiles without undefined references or overfull boxes.
