---
name: old-strong-route-audit
description: Audit findings on the legacy high-cold "0.2863/0.2098" route — reproducible but not main-table-usable
metadata:
  type: project
---

Read-only audit (2026-07-23) of the legacy strong route that produced the paper's headline cold `0.2863 / 0.2098`, to decide whether it can be reused as a strong backbone. Source: `usim_feedback_fast3_content_delta_static` / CBI-faithful products (NOT the clean V3.x route).

**No leakage found:** strict-cold items appear 0 times in train; test-cold ∩ train-item = 0; all test users have train history; final eval uses train history only; course graph uses concept-only relations (`prereq_users=0`), so val/test interactions are not used. Old seed-2025 log confirms no checkpoint resume (`InitCheckpointDir` empty). Three-seed old results reproduce byte-for-byte from recovered source → not a fluke.

**Why it still can't be used:** the legacy full-ranking evaluator generates a cold course's vector ONCE for the whole candidate bank, then generates a SEPARATE random vector for the positive sample and overwrites its target score. So the same cold course has two representations in one ranking pass. Not a leak, but makes `.2863/.2098` not a stable, interpretable, unified ranking metric — cannot be kept as evidence for the corrected method.

**Also mis-aligned train/inference:** training simulator uses hot item's real ID vector as target; legacy cold inference has no such target. No pseudo-cold samples — `TrainForceCold` under strict split produces no cold training rows, leaving only 35% random ID dropout. Course reward active in training but legacy inference passes no user history → no personalized course feedback.

**Verdict / reuse boundary:**
- Reusable: strict split, train-only history, catalog/concept graph, base encoders, cold-experiment scaffolding.
- Must discard: legacy positive-sample re-scoring eval, random-candidate eval, old `legacy_id` inference anchor, and writing old numbers back to the main table.
- Do NOT revert 旧版 2025 as the overall route — its hot is very weak; even repaired seed-2025 (semantic repair) is cold .3062/.2068 but overall only ~.1454/.0817.
- Best role: a "repaired cold expert" candidate / legacy cold baseline, never the overall main model.

Semantic-repair regression tests: 8 passed. See [[usim-v33-v36-experiment-arc]] for the Pareto tradeoff vs the clean route.
