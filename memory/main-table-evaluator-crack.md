---
name: main-table-evaluator-crack
description: CRITICAL — the AAAI main table's CKG-RL cold row (legacy 0.2863) and overall row (clean 0.157) come from two different runs+evaluators; cold 0.2863 is not a defensible SOTA
metadata:
  type: project
---

Verified 2026-07-23 by tracing the main-table data sources. This is the single most important correction for the whole cold-start line — do NOT plan around "CKG-RL cold 0.2863 is SOTA."

**The crack:** the two AAAI main tables use DIFFERENT evaluators for the CKG-RL row, while all baselines use one common clean evaluator.
- Cold main table (`paper_aaai27/main_table.tex`, values from `figures/cbi_main_table_comparison/cbi_vs_main_table.csv`): every row's Evidence column says "Paper main table (3-seed mean)". The CKG-RL row 0.2473/0.2863/0.1972/0.2098 is a **legacy historical number carried over**, produced by the legacy `usim_feedback_fast3_content_delta_static` + `fast3_delta/eval.py` path. That evaluator was audited (see [[old-strong-route-audit]]) to generate a cold course TWICE in one ranking (catalog bank vector + separate re-computed positive-sample vector at eval.py:~467) → not a stable unified ranking metric.
- Overall main table (`figures/overall_baseline_comparison/`, script `export_overall_baseline_comparison.py` → `audit_significance_inputs.py` + `export_warm_target_table.py`): all 12 methods incl. CKG-RL are re-computed per-seed from `per_item_full_cold/hot_*.csv` under ONE common clean evaluator, weighted cold_count=68 / hot_count=575. CKG-RL overall = 0.1568/0.0879, rank 9/12, −37%/−45% vs strongest baseline. CKG-RL cold_source there = `outputs/significance_per_item_exports/mooccube/ckg_rl_full/` (Jul 7 re-export), NOT the old .2667 dir.

**The two CKG-RL rows are not even the same run.** Cold table = legacy carried-over 0.2863/0.2098. Overall table = a DIFFERENT run (`significance_per_item_exports/mooccube/ckg_rl_full/`, Jul 7 clean re-export) whose own cold is only 0.146/0.088 and whose own hot is very weak (R@10=0.131/N@10=0.068). So the "cold #1 vs overall rank-9" contradiction comes from (a) two different runs AND (b) two different evaluators — not from a single model. Anyone aligning the two tables will catch this. This is more serious than any single V3.x result.

**IMPORTANT mechanism (corrected 2026-07-23):** the legacy dual-vector trick in `eval.py` (`replace_strict_cold_with_refined` / `build_eval_pos_item_vecs`) is gated on `cold_mask` ONLY — it inflates strict-COLD numbers, never hot. So legacy inflation touches cold, not overall directly. Because overall is course-count weighted with 575 hot vs 68 cold, overall is HOT-dominated. The clean-overall 0.157 is low because THAT run's hot is weak (a model/training property from cold-only early-stopping), NOT because a clean evaluator crushed a legacy-inflated overall.

**Baselines ARE trustworthy / on the common evaluator** (this makes the target well-posed). MOOCCube clean-evaluator lines to beat:
- Cold SOTA target: CGRC R@10=0.2589 / N@10=0.1845 (next: SEMCo .2306/.1416, ALDI .2189/.1521).
- Overall "not too weak": SEMCo R@10=0.2502; CGRC R@10=0.2397 / N@10=0.1586.

**What is and isn't quantified (corrected 2026-07-23):** legacy cold inflation is real but only measured indirectly; do NOT claim a specific legacy→clean overall drop for .2667. The earlier ".2336 → .1454 = legacy inflation" statement was WRONG — the .1454 came from the semantic-repair run, which changed TRAINING (initial_state target), so its low hot/overall is a training effect, not an evaluator effect. .2667 itself has never been re-evaluated under the clean evaluator. Its hot (.2181/.1342, legacy but hot is NOT inflated by the cold-only trick) is genuinely strong, so its clean overall would likely hold near ~.22, NOT collapse. Only its cold needs a clean re-eval to know the de-inflated value. See [[old-2667-status-open]].

See [[old-strong-route-audit]], [[usim-v33-v36-experiment-arc]], [[usim-v2-v3-experiment-arc]].
