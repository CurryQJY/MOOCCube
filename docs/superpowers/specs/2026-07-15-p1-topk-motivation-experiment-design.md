# P1 Top-K Motivation Experiment Design

## Objective

Test whether CKG-RL reduces pedagogical risk in actual full-catalog recommendations compared with CGRC on MOOCCube, using the same three static item-cold splits and checkpoint-selected models as the main effectiveness comparison.

## Models And Checkpoints

- CKG-RL: `checkpoints/recovery_validation/main_table_51ea12fc_candidate/strict_item_cold_balanced_thr1_seed_<seed>/finished.pt`.
- CGRC: `checkpoints/content_delta_pop5/p1_motivation_cgrc_main_table_reproduction/strict_item_cold_balanced_thr1_seed_<seed>/best.pt`.
- Seeds: 2025, 2026, and 2027.
- Checkpoints are read-only. CKG-RL restores `es_best_state`; CGRC restores `best_state`.
- Candidate scoring, seen-item masking, target restoration, and any model-specific score transformation remain inside each model's native full-ranking evaluator.

## Recommendation Export

Each native evaluator streams one JSONL record per cold-test interaction after final score adjustment:

- model and seed;
- sequential sample index;
- user ID and held-out target course ID;
- target train popularity;
- ranked Top-20 course IDs and scores.

The exporter writes to a temporary file and atomically replaces the destination only after successful evaluation. It never modifies a checkpoint. Recomputed R@10 and N@10 must match the model's evaluator output before the export is accepted.

## Primary Analysis Unit

Risk is first computed for every recommended course in each Top-10 list. List-level values are averaged over ranks 1-10, then macro-averaged by `(model, seed, cold target course)`. Model comparisons are paired on `(seed, cold target course)`, matching the main-table course-macro emphasis.

## Risk Definitions

All signals use training history only and are independent of the method's learned reward weights.

1. **Prerequisite coverage gap**: one minus the fraction of a recommended course's prerequisite courses present in the learner's training history. Courses without prerequisites have gap zero.
2. **Concept continuity**: mean directional concept overlap from the recommended course to courses in the learner's training history. Higher is better.
3. **Content-structural difficulty gap**: positive difference between the recommended course's structural complexity and the learner's readiness. Structural complexity is the equal-weight mean of robustly normalized prerequisite count and course-concept count. Readiness is the mean complexity of up to five available structurally most advanced distinct courses in the learner's training history. No popularity feature is used.
4. **Structural redundancy**: maximum of same-family duplication and bidirectional video-unit containment between the recommendation and the learner's history. Higher is worse.

Primary tables report prerequisite gap, concept continuity, difficulty gap, and redundancy across all Top-10 recommendations.

## Auxiliary Analysis

- Cold-course proportion among all Top-10 recommendations, where cold means zero train interactions in that seed.
- The same four risks among recommended cold courses only.
- A Top-10 list with no recommended cold course is missing for the cold-only risk analysis, not assigned zero.
- Top-20 is retained as an export artifact and sensitivity analysis; Top-10 is the paper-facing result.

## Statistics And Gates

- Report three-seed course-macro mean and standard deviation per model.
- Report paired CKG-RL minus CGRC differences over matched `(seed, target course)` rows.
- Use paired bootstrap 95% confidence intervals and a two-sided paired permutation p-value with a fixed analysis seed.
- Directional interpretation: lower is better for prerequisite gap, difficulty gap, and redundancy; higher is better for concept continuity.
- Export coverage must equal the native evaluator's number of cold-test interactions for every model and seed.

## Reproduction Discrepancy Note

The repaired CGRC run uses the same split and hyperparameter configuration as the old main table but a later implementation. The later implementation changes the training numerical path from COO to CSR sparse propagation and adds chunked reconstruction logits. A read-only seed-2025 probe showed that switching only final evaluation from CSR to COO changes R@10 by 0.0000093 and N@10 by 0.0000022, so the observed main-table discrepancy originates in the changed training trajectory rather than final evaluation. Seed 2026 additionally resumed without restoring RNG state. P1 must identify its CGRC source as the same-configuration repaired rerun rather than an exact old-main-table checkpoint.
