# Current Session Handoff

## Workspace

- Root: `D:\DeskTop\MOOCCube`
- Paper: `D:\DeskTop\MOOCCube\paper_aaai27`
- Main PDF: `D:\DeskTop\MOOCCube\paper_aaai27\main.pdf`
- Python: `D:\anaconda3\envs\req_py312\python.exe`
- Date: 2026-07-16

## Current Goal

Build a persuasive P1 motivation experiment for strict course-cold MOOC recommendation.

The agreed argument is:

1. Compare with a course-specific recommender to diagnose course-structure limitations.
2. Compare with a strong generic cold-start recommender to diagnose strict cold-start ranking limitations.
3. Explain how CKG-RL is designed to target both limitations.
4. Keep the complete effect and risk audit separate from the introductory motivation diagnosis.

Only empirically supported claims may be used.

## Baseline Roles

- PCGNN: course-specific baseline used to diagnose structural mismatch.
- CGRC: generic cold-start baseline, explicitly not described as a course recommender.
- CKG-RL: proposed method.

## Figure Division of Labor

### Figure 1: Motivation diagnosis

Figure 1 now answers why existing methods fall short. It does not show CKG-RL numerical performance gains.

Internal title:

`Why existing methods fall short`

Panel (a):

- Title: `PCGNN: structural mismatch`
- Held-out MOOCCube Top-10 diagnostics.
- Prerequisite gap mean: 0.6427.
- Difficulty gap mean: 0.0963.
- Horizontal bars use diagonal hatching for grayscale printing.
- Error bars are 95% bootstrap intervals over 204 seed-course units.

Panel (b):

- Title: `CGRC: weak cold-course ranking`
- Course-level NDCG@10 distribution over 204 held-out seed-course cases.
- 46% of cases have NDCG@10 no greater than 0.10.
- Descriptive Top-10 cold-course share: 24.6%.
- Histogram bars use dotted hatching for grayscale printing.

Bottom response strip:

- `Structure -> knowledge signals`
- `sampling | rewards | supervision`
- `Cold ranking -> cold embeddings`
- `anchoring | masking | simulation`

The response strip describes design intent only. It does not claim that every risk proxy improves.

Generated files:

- `paper_aaai27/figures/mooccube_method_motivation.pdf`
- `paper_aaai27/figures/mooccube_method_motivation.svg`
- `paper_aaai27/figures/mooccube_method_motivation.png`
- `paper_aaai27/figures/mooccube_method_motivation_existing_diagnostics.csv`

Drawing script:

- `paper_aaai27/scripts/draw_method_motivation.py`

### Figure 3: Complete effect and risk audit

Figure 3 remains the complete double-baseline audit:

- circles: CKG-RL versus PCGNN;
- squares: CKG-RL versus CGRC;
- positive, adverse, and inconclusive results remain visible;
- exposure comparison remains visible.

Figure 3 is responsible for evaluating what CKG-RL improves and what trade-offs remain. Do not duplicate this numerical comparison in Figure 1.

File:

- `paper_aaai27/figures/mooccube_p1_topk_motivation.pdf`

## Motivation Table

The supported motivation evidence table has seven rows:

1. Course-structure limitation:
   - CKG-RL versus PCGNN prerequisite gap.
   - CKG-RL versus PCGNN difficulty gap.
2. Strict cold-start effectiveness:
   - CKG-RL versus CGRC Recall@10.
   - CKG-RL versus CGRC NDCG@10.
3. Course-reward mechanism:
   - cold prerequisite gap;
   - cold difficulty gap;
   - cold-course share.

The table is intended for the supplement, not the main paper.

Files:

- `paper_aaai27/scripts/build_p1_motivation_evidence_table.py`
- `paper_aaai27/tables/mooccube_p1_motivation_evidence.csv`
- `paper_aaai27/tables/mooccube_p1_motivation_evidence.tex`

Important formatting rule:

- LaTeX percentage signs in generated effect strings must be escaped as `\%`.

## Main-Paper Narrative

`paper_aaai27/main.tex` has been aligned as follows:

- Introduction identifies PCGNN and CGRC as complementary references.
- Figure 1 diagnoses existing-method problems.
- The Figure 1 caption is concise and points to Figure 3 for improvement and trade-off verification.
- The RQ2 opening explains that Figure 1 is a diagnosis, not a second performance comparison.
- Figure 3 reports the complete double-baseline audit.

All Figure 1 diagnostics use held-out MOOCCube test data. Validation data are only for model and hyperparameter selection.

## Provenance Boundary

Do not mix the CGRC repaired Top-20/list-audit export with main-table CGRC checkpoint values as if they were the same checkpoint.

Use:

- repaired export diagnostics for the CGRC course-level NDCG distribution and Top-10 exposure diagnosis;
- main-table values only for the main strict cold-start performance table and supported effectiveness table.

The paper already discloses the repaired CGRC rerun provenance and seed-2026 RNG-state limitation.

## Tests

Relevant tests:

- `tests/test_method_motivation.py`
- `tests/test_p1_motivation_evidence_table.py`
- `tests/test_draw_p1_topk_motivation.py`
- `tests/test_p1_motivation_mechanisms.py`
- `tests/test_p1_topk_motivation_analysis.py`
- `tests/test_p1_checkpoint_export_entrypoints.py`

Latest verified command:

    D:\anaconda3\envs\req_py312\python.exe -m pytest tests\test_method_motivation.py tests\test_p1_motivation_evidence_table.py tests\test_draw_p1_topk_motivation.py tests\test_p1_motivation_mechanisms.py tests\test_p1_topk_motivation_analysis.py tests\test_p1_checkpoint_export_entrypoints.py -q --basetemp=.pytest_tmp\figure1_diagnosis_verified

Result:

`62 passed`

## Compilation

Latest compilation command:

    D:\anaconda3\envs\req_py312\python.exe C:\Users\pc\.agents\skills\latex-paper-en\scripts\compile.py main.tex --recipe latexmk

Working directory:

`D:\DeskTop\MOOCCube\paper_aaai27`

Result:

- `main.pdf` generated successfully.
- 12 pages.
- No undefined references.
- No undefined citations.
- No undefined control sequences.
- No Overfull boxes.
- Existing Underfull warnings remain unrelated to Figure 1.

## Design Documents

- `docs/superpowers/specs/2026-07-16-figure1-motivation-redesign.md`
- `docs/superpowers/plans/2026-07-16-figure1-motivation-redesign.md`

## Workspace Constraints

- The git worktree is dirty.
- Do not reset, revert, clean, stage, commit, or push unless explicitly requested.
- Preserve unrelated user changes.
- Use `apply_patch` for manual file edits.
- Use a repository-local pytest `--basetemp`.
- Do not use subagents unless explicitly requested.

## Current Status

The Figure 1 redesign, manuscript alignment, tests, grayscale inspection, and final main-paper compilation are complete.

A temporary grayscale inspection image may still exist at:

`paper_aaai27/figures/mooccube_method_motivation_grayscale_check.png`

An attempted deletion was rejected by the approval layer. It does not affect the paper.
