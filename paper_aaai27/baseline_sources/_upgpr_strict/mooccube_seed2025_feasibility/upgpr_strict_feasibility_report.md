# UPGPR Strict Feasibility Audit

- Verdict: **FEASIBLE_FOR_FORMALIZATION**
- Seed: `2025`
- Split: `D:\DeskTop\MOOCCube\outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2025`
- Device: `cpu`
- Validation R@10 / N@10: `0.102941` / `0.069914`
- Test R@10 / N@10: `0.080882` / `0.056106`
- Test target path reachability: `0.080882`

This is a capped one-seed feasibility run. It is not eligible for the paper main table.
Cold collaborative positives and CF negatives are excluded; cold course embeddings are reconstructed only from warm-anchored static metadata via mean(tail - relation).
