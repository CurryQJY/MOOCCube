---
name: old-2667-status-open
description: Status of the old .2667 CKG-RL version vs the "cold SOTA + decent overall" goal — NOT resolved; needs a clean-evaluator re-eval
metadata:
  type: project
---

Checked 2026-07-23 (this file CORRECTS an earlier wrong version that claimed .2667's overall "collapses to .145" under clean eval — that number was from a DIFFERENT run and was a false conclusion).

The old `.2667` version: `outputs/content_delta_pop5/course_ablation_e60_3seed/full/` (model `USIM-Feedback-FAST3-ContentDelta`, 3-seed 2025/2026/2027, cold_count=68 hot_count=575, 60 epochs). Trained WITHOUT pseudo-cold, content-delta off, course reward/prereq-aux on, cold_only early stop.

**Surface numbers (item-macro 3-seed mean):** Cold R@10=0.2667 / N@10=0.1962 (beats CGRC 0.2589/0.1845); Hot R@10=0.2297 / N@10=0.1412; count-weighted Overall ≈ 0.2336 / 0.1470. Hits both lines on paper.

**How the legacy evaluator inflation actually works (verified in `fast3_delta/eval.py`):** the dual-vector trick (`build_eval_pos_item_vecs` refines the positive-target vector separately from the item bank via `replace_strict_cold_with_refined` + `strict_cold_item_mask`) is gated on `cold_mask` ONLY. **Hot items use base vectors in both bank and target — they are NOT inflated.** So legacy inflation touches COLD only, not hot.

**Key implication — overall is hot-dominated (575 hot vs 68 cold), so it will NOT collapse under clean eval.** Even if clean re-eval drops .2667 cold from ~0.253 to ~0.20 (seed-2025), overall R@10 = (68·0.20 + 575·0.2181)/643 ≈ 0.266 — essentially unchanged. .2667's hot (0.218/0.134 seed-2025) is clean and genuinely strong.

**Where the main-table overall 0.157 really comes from:** NOT .2667. The overall table uses a different run `significance_per_item_exports/mooccube/ckg_rl_full/` whose OWN hot is weak (R@10=0.131/N@10=0.068, ~half of .2667's hot). That is a model/training property of that specific cold-over-optimized run, not an evaluator artifact. So the main-table contradiction = cold row filled with legacy-inflated 0.2863 from one run + overall row filled with a hot-collapsed different run (0.157). See [[main-table-evaluator-crack]].

**Corrected conclusion: .2667 is NOT ruled out.** Its hot/overall are likely to hold under clean eval; only its cold needs clean re-scoring to remove legacy inflation. **Open action:** re-evaluate the .2667 checkpoint (if it still exists — check `checkpoints/content_delta_pop5/course_ablation_e60_seed2025/...`, but manifest showed SAVE_CKPT=0 for the run so the checkpoint may NOT have been saved) under the clean evaluator to get its true clean cold, then recompute overall. This is the cheapest path to a candidate that meets both lines — do this before writing off the old route. Contrast with [[old-strong-route-audit]] and [[usim-v33-v36-experiment-arc]].

**Checkpoint is GONE (verified 2026-07-23):** .2667 ran with SAVE_CKPT=0 for all three seeds; the manifest's `USIM_FB_CKPT_DIR` (`checkpoints/content_delta_pop5/course_ablation_e60_seed2025/full_nodelta_course/...`) is empty and NO `*.pt` exists under any `course_ablation_e60*` dir. So a clean re-eval requires RE-TRAINING from the manifest config (all env vars are preserved in `outputs/content_delta_pop5/course_ablation_e60_3seed/full/strict_item_cold_balanced_thr1_seed_2025/static_protocol_manifest.json`), 60 epochs × 3 seeds, batch 2048.

**Corrected mechanism (why overall likely does NOT collapse under clean eval):** legacy dual-vector inflation is cold-only; overall is hot-dominated (575 hot vs 68 cold); .2667's hot (0.218/0.134) is clean and genuinely strong because it did NOT use cold-only aggressive training that crushed hot in later runs. So clean re-eval would at most shave cold a little, leaving overall ~0.22. .2667 is the ONLY known config with "hot alive AND cold decent" — the closest historical lead to the goal, NOT something to exclude. The disqualifier is only that its cold SOTA (0.2667) rides on legacy inflation; the honest question is whether cold can be pushed to ≥0.1845 N@10 by a defensible (non-legacy) mechanism while keeping .2667's live hot.
