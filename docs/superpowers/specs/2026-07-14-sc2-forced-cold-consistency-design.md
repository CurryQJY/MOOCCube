# SC2Rec-Style Forced-Cold Consistency Design

## Goal

Evaluate whether score-distribution consistency between a warm course view and
its counterfactual forced-cold view improves strict course-cold training without
changing the current main-table implementation or evaluation protocol.

The experiment is inspired by SC2Rec's warm/cold generalization principle. It
is an isolated adaptation, not a reproduction of SC2Rec's full cross-domain and
sharpness-aware training procedure.

## Protected Files

The experiment must not edit:

- `usim_feedback_fast3_content_delta.py`
- `run_usim_feedback_fast3_content_delta_static.ps1`
- `aggregate_fast3_static_results.py`
- `paper_aaai27/main_table.tex`

Their SHA-256 hashes are recorded before implementation and must be identical
after the smoke experiment.

| Protected file | SHA-256 before implementation |
|---|---|
| `usim_feedback_fast3_content_delta.py` | `71E91D0DA985A6AC982C3C186FEBADF0A988115C9448F24A3E6342AFEABB943B` |
| `run_usim_feedback_fast3_content_delta_static.ps1` | `B20E6F85B10D57EBC207DC4E717CFFB2224484F8B8ECB3177CC93AFA76AD172E` |
| `aggregate_fast3_static_results.py` | `835DDD07DAC56FC043AB0DCF7DD937FC68293D627373A9235BC55B05B17818FD` |
| `paper_aaai27/main_table.tex` | `6E61EEAE97459CF9BB1CC966351674E374D78CEE88F1CA624ECD55904A1B4F1A` |

## Architecture

Create a separate Python entry point that imports the current CKG-RL main-table
module and installs a subclass/config pair in memory before delegating to the
existing training pipeline. The original module remains unchanged on disk.

For every training batch:

1. Run the existing forward pass unchanged to obtain the CKG-RL loss.
2. Encode each target course twice with ID dropout disabled:
   - teacher view: normal warm ID-content fusion;
   - student view: `force_cold=True`, so no course-ID evidence is available.
3. Detach learner vectors and teacher logits.
4. Score both views against the same batch learners.
5. Reuse the existing known-positive/duplicate-course mask.
6. Minimize temperature-scaled teacher-to-student KL divergence for rows whose
   target course is warm in the observed training data.
7. Add the weighted consistency term to the unchanged base loss.

The loss is one-way distillation. Only the forced-cold course path receives
gradients from this term; the learner bank and teacher view do not move to make
the consistency objective artificially easy.

## Configuration

The isolated entry point reads:

- `USIM_SC2_CONSISTENCY_WEIGHT`, default `0.10`;
- `USIM_SC2_CONSISTENCY_TEMP`, default `0.20`;
- `USIM_SC2_CONSISTENCY_WARM_ONLY`, default `1`.

The runner uses a unique output/checkpoint root and passes the existing strict
course-cold protocol, seed 2025, one epoch, and patience one.

## Diagnostics

Every training batch reports:

- `sc2_consistency_loss`;
- `sc2_consistency_weighted_loss`;
- `sc2_consistency_active_ratio`;
- `sc2_teacher_student_cosine`.

The entry point prints an explicit banner identifying the experiment as an
SC2Rec-style adaptation.

## Testing

Unit tests must establish that:

1. identical teacher/student distributions have approximately zero loss;
2. divergent distributions produce a positive finite loss and student gradient;
3. an all-inactive mask returns differentiable zero;
4. masked candidates do not influence the loss;
5. environment configuration is parsed correctly;
6. the new runner references only the new entry point and isolated paths.

The smoke experiment succeeds when one epoch completes, the consistency loss is
finite and active, a result artifact contains strict cold course-macro metrics,
and all protected hashes remain unchanged.

## Scope Boundaries

- No SAM/sharpness optimizer is added in this first experiment.
- No main-table row or aggregator input is changed.
- No validation/test interaction is used to construct the consistency teacher.
- No claim of reproducing SC2Rec is made.
- No multi-seed or full 60-epoch run is launched at this stage.
