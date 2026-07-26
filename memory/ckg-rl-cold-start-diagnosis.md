---
name: ckg-rl-cold-start-diagnosis
description: Root cause of the CKG-RL / USIM AAAI paper's weak ablations and train-inference mismatch
metadata:
  type: project
---

The AAAI paper (CKG-RL, MOOCCube, strict item-cold-start) had two core problems diagnosed in a Codex session on 2026-07-22.

**Problem 1 — ablations don't support "core components".** In the new `.2863` cold R@10 version, only two same-config 3-seed ablations were actually run: removing educational reward *raised* all metrics, and removing the simulator was ~flat (overall R@10 .1568→.1564, overall N@10 even rose). Other RQ3 rows (`w/o KG sampler`, `w/o prereq aux`, `w/o all course signals`) were **provisional** — old e60 ablation drops shifted onto the new Full score, not direct reruns. In the old e60 version, removing PPO loss was nearly flat and a static content-masked scorer scored *higher* than the full RL pipeline. Conclusion: cannot claim reward/PPO/simulator as core performance drivers.

**Problem 2 — training ≠ inference MDP (the real root cause).** In the `.2863` script (`usim_feedback_fast3_content_delta_recovered_51ea_candidate.py`), strict-cold courses are removed from training and `pseudo-cold=false`, so `EffectiveColdRatio=0`. The simulator's refined output gets switched back to `z_i_base` via `_apply_refinement_only_to_effective_cold`, so the main ranking loss trains on the warm base embedding, not the refined vector. At inference, strict-cold courses get one offline global simulator pass (no target anchor, no serving user) producing a cached cold vector used in dot-product ranking. So it is NOT the per-user sequential RL policy the paper describes.

The `.2863` version is a **cold-specialized** model: cold R@10 .2667→.2863 but hot R@10 .2297→.1415 and overall R@10 .2336→.1568 (aggregated over ~68 cold / ~575 hot courses). It trades hot collapse for cold gains.

See [[usim-original-paper-rl-design]] and [[ckg-rl-v2-v3-experiment-history]].
