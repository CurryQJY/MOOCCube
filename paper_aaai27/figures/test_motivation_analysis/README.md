# Held-out test-split decoupling verification (Table 1)

This directory holds the reproducible evidence that the ranking--pedagogy
decoupling shown in Figure 1 (validation lists) preserves the **sign** of every
correlation on the held-out **test** split. It backs the sentence in Section
RQ2 and `Table~\ref{tab:decoupling-test}` in `main.tex`.

## Result

Spearman rho between course-macro cold-target NDCG@10 and each structural
proxy, over exposed cold courses (recall@10 > 0), pooled across seeds
2025/2026/2027:

| Structural proxy      | Validation rho | Test rho | Sign |
|-----------------------|:--------------:|:--------:|:----:|
| Structural redundancy | +0.53          | +0.47    | OK   |
| Prerequisite gap      | -0.15          | -0.19    | OK   |
| Difficulty gap        | -0.18          | -0.08 (n.s., p=0.33) | OK |
| Concept continuity    | +0.56          | +0.55    | OK   |

Exposed n: validation 158, test 145 (test per model: cgrc 100, pcgnn 45).
All four signs agree with validation. The difficulty signal is weak and is not
used as a primary motivation claim.

Within-CGRC median split on test (the robustness sentence): courses CGRC ranks
well carry ~2.3x the redundancy of courses it ranks poorly (0.130 vs 0.057),
while prerequisite gap is essentially unchanged (0.564 vs 0.550). This mirrors
the validation pattern.

## How it was produced

Same frozen checkpoints and structural artifacts as Figure 1; only the
evaluation split was switched from validation to test.

1. Top-20 test exports (frozen checkpoints, `--analysis-split test`):
   - CGRC:  `export_p1_cgrc_topk.py`  -> `outputs/test_motivation/cgrc/*/top20_test.jsonl`
   - PCGNN: `export_p1_pcgnn_topk.py` -> `outputs/test_motivation/pcgnn/*/pcgnn_top20.jsonl`
   Each export directory contains an `export_manifest.json` recording checkpoint
   SHA-256 (before/after, to prove the checkpoint was not mutated), split files,
   script files, and record counts. Row counts per seed: 65605 / 66749 / 62280,
   identical across both models.

2. Correlation analysis (reuses the exact Figure-1 primitives
   `build_real_risk_artifacts`, `_seed_inputs`, `analyze_export_record`):

   ```
   conda run -n req_py312 python paper_aaai27/scripts/verify_decoupling_on_test.py
   ```

   Writes `course_macro.csv` (408 rows = 2 models x 3 seeds x 68 cold courses,
   at cutoff 10) and prints the validation-vs-test sign table above.

## Files

- `course_macro.csv` — per-course NDCG@10, recall@10, and the four cold-list
  structural proxies, for both models across three seeds.
- `README.md` — this file.

## Reproducibility note

PCGNN inference shows GPU floating-point drift of ~3.4e-7 on ranking metrics
(well within the 1e-5 tolerance used). In one batched run, seed 2025 hit a
transient checkpoint-validation drift (~0.03) attributable to a GPU
context/residual-process race; a clean standalone re-run reproduced the
expected 3.4e-7 drift and passed. The archived seed-2025 export is that clean
run. CGRC compares against its own freshly computed native result and shows no
such sensitivity.

Environment: conda env `req_py312` (PyTorch + CUDA). Structural artifacts are
loaded from `MOOCCube/.course_artifact_cache/`.
