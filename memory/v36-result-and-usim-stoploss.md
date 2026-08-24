---
name: v36-result-and-usim-stoploss
description: V3.6 global-stable action-distill result (viability gate FAILED) and the pre-registered USIM stop-loss trigger
metadata:
  type: project
---

Ran 2026-07-23 (seed 2025, P-only). V3.6 = V3.5 counterfactual action distillation + expert-trajectory mixing + global anchor stability constraint (anchor=128, stability_weight=10.0, expert_fraction=0.5, vector generator identical hash to V3.5/V3.2). Files: `ckg_rl_usim_v36_global_stable_distill.py`, launcher `run_ckg_rl_usim_v36_global_stable_distill_seed2025.ps1`.

**Result: viability_gate_passed = FALSE. No test run (correct per P-only protocol — gate not passed ⇒ never reads C_val/C_test; `test_loaded=false`, `outer_c_val_evaluated=false`).**

The gate (`_viability_gate`) is a COMPOUND condition — must beat V3.5 on BOTH axes:
- `source_hash_gate_passed` ✓
- `epoch > 0` ✓ (selected 15)
- `p_val_rank_gain > 0.0047182762` (V3.5) → 0.006513 ✓ (improved, ~9.7% of oracle 0.06744, up from V3.5's 7%)
- `train_action_agreement > 0.2388` (V3.5) → 0.2301 ✗ **← this is what failed**

So V3.6 raised the KL rank-gain but LOWERED action agreement → not strictly better than V3.5 → gate fails. The stability constraint DID work (`p_val_anchor_drift` ≈ -2.5e-05, near zero). The action-distillation half still fails: actor learns only ~16-23% of teacher greedy actions; epochs 1-11 had NEGATIVE p_val gain, only turned positive at epoch 12.

**Do NOT run test for V3.6.** Reasons: (1) gate-false is the designed stop signal; (2) generator hash identical to V3.5 whose test replay is already known (Cold 0.2501/0.1375), and V3.6's lower action-agreement means cold cannot beat V3.5; (3) seed-2025 test already contaminated by prior route selection.

**USIM stop-loss triggered.** Pre-registered rule (from earlier session): if P_val gain can't clear the bar, stop USIM as main line and move to static content-behavior graph model. USIM-add-mechanism marginal returns are exhausted: V3.4→3.5→3.6 = 7%→7%→9.7% of oracle. See [[usim-v33-v36-experiment-arc]], [[main-table-evaluator-crack]], [[old-2667-status-open]].
