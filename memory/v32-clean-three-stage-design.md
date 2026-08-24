---
name: v32-clean-three-stage-design
description: The clean T->G->V3 three-stage rebuild plan for cold-start (V3.2-Clean), implemented and run — see arc memories for results
metadata:
  type: project
---

The conclusion of the 2026-07-22 diagnosis session: stop patching V3.1, rebuild as a clean three-stage `V3.2-Clean` that separates behavior teacher, content initializer, and USIM policy. Motivated by teacher contamination (current V3.1 warm checkpoint already ran old PPO/course components) and proxy leakage. Mirrors the original USIM training boundary (see [[usim-original-paper-rl-design]]).

**Outer strict-cold split (fixed):** H=train-hot items, C_val/C_test=strict-cold items. Inside H (fixed before training): H_G (content-generator training), P_train (pseudo-cold RL training), P_val (pseudo-cold RL val/early-stop). H also splits interactions H_train/H_val/H_test (teacher train / early-stop / final report only), matching original paper's 8:1:1 IV split.

**Stage A — behavior teacher T:** train recommendation backbone on H_train only → E_u^T, E_H^T. NO PPO, rollout, course reward, course candidate bias, pseudo-cold, random ID mask, refined eval. Early-stop on H_val. T may see P_train/P_val interactions ONLY to supply offline oracle e_i^T and positive-user sets — it is not the deployed model. Frozen after; hot items always use E_H^T at inference (guarantees hot parity).

**Stage B — content generator G:** train only on H_G = H \ (P_train ∪ P_val), mapping G(c_i) → e_i^T. No item ID input, no RL. Keeping P_train/P_val out of G training makes them genuine unseen-behavior cold proxies (avoids proxy contamination). C_val/C_test never visible.

**Stage C — V3 policy:** freeze T and G; train only actor/critic. P_train initial state h0=G(c_i), ID masked. Teacher e_i^T and positive users used ONLY for training reward, never in inference. TURN OFF random ID-dropout (else creates "no-ID-but-no-rollout" mixed samples — currently ~24.5% of samples are no-ID-no-sim vs ~30% real pseudo-cold). Course reward + course-logit bias may stay but user history must exclude the target pseudo item (prevent leakage). Default action pool must match inference: state-retrieval + END; positive/residual candidates only as separate logged ablation, not the silent main training pool. Add final-state trust projection to bound ||h_T - h_0|| (V3 inference displacement was ~1.8× training).

**Inference:** hot → fixed E_H^T, no rollout; strict cold → G(c_i) → V3 policy → refined vector; no item target, positive users, PPO reward, or target history.

**Validation order:** (1) T hot parity; (2) G-only strict-cold baseline; (3) G+V3 no-course-signal; then add course signals. Single seed 2025 first; only extend to 2026/2027 if cold holds AND hot/overall no longer collapse AND simulation behavior is non-degenerate.

**Status:** IMPLEMENTED and run (seed 2025). This design was fully built (`ckg_rl_usim_v32_clean.py`) and became the clean foundation for all later variants. Outcomes and the full V3.2→V3.6 arc are in [[usim-v33-v36-experiment-arc]]. Key V3.2 result: after fixing the epoch-0 identity bug (old epoch 0 was random-actor rollout, not G(c_i)), clean seed2025_identity selected PPO epoch 15 — PPO beat legit identity on C_val cold N@10 by +0.01075, proving the simulator is not a dead component. But test Cold R/N=.2495/.1388, Overall=.2300/.1504 — below main-table baselines, cannot replace the paper's CKG-RL row.

**IMPORTANT scheduling constraint (user, session end):** do NOT launch V3.6 or any GPU experiment — a separate time-efficiency experiment is running and must not be disturbed. V3.6 is implemented but its formal seed-2025 run is deliberately left pending.
