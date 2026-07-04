# CGRC-Aligned FAST3 Reconstruction Design

## Goal

Add a traceable FAST3 improvement that is close to CGRC's original masked graph reconstruction idea: learn, from pseudo-cold warm items, which users a cold item should connect to from content/course signals, then use that signal as a cold-side training auxiliary and candidate-sampling prior.

## Why This Path

The current MOOCCubeX gap to CGRC-paper is mostly cold item-macro N@10 rather than cold R@10. CGRC's strength is not a generic weight change; it reconstructs user-item edges for cold items from item content before ranking. FAST3 already has course-aware reward and user simulation, but it lacks an explicit edge-reconstruction objective. This design adds that missing objective without changing the static split, full-ranking evaluator, or old defaults.

## Paper Alignment

The implementation follows CGRC's core mechanics at the component level:

- Pseudo-cold masking: warm training items are sampled as cold surrogates.
- Edge reconstruction: the model predicts users connected to those pseudo-cold items using item content-derived vectors and propagated/current user states.
- Positive-in-denominator softmax: positives stay inside the candidate set, matching the local paper-faithful CGRC adapter's correction.
- Cold inference use: the learned reconstruction score is used only as a cold/tail candidate prior in FAST3, not as a leaked test-edge source.

This is intentionally not a wholesale LightGCN G-hat replacement inside FAST3; that would duplicate CGRC as a second model. The novelty is to make the CGRC-style reconstruction signal guide FAST3's user simulation and content-delta residual learning.

## Components

1. `fast3_delta/config.py`
   Add disabled-by-default CGRC reconstruction options, all backed by environment variables.

2. `usim_feedback_fast3_content_delta.py`
   Add a small reconstruction head, auxiliary loss, candidate-prior integration, logging, optimizer inclusion, and manifest fields.

3. `run_usim_feedback_fast3_content_delta_static.ps1`
   Add traceable launcher parameters and tracked environment variables so every run records whether the component was active.

## Safety And Traceability

Defaults preserve existing behavior. A run is considered traceable only if logs and `static_protocol_manifest.json` record:

- `use_cgrc_recon`
- `cgrc_recon_aux_weight`
- `cgrc_recon_sample_weight`
- `cgrc_recon_pseudo_ratio`
- `cgrc_recon_topk`
- `cgrc_recon_temperature`
- `cgrc_recon_only_cold_or_tail`

Training logs also report reconstruction loss, active ratio, positive count, and sampling-prior activity.

## Validation Plan

First run seed 2025 only on MOOCCubeX with the fair e30 configuration. Compare against:

- `relations_e30_masktrue` seed2025, same relation dir and concept threshold.
- `relations_aug_v2_e30` seed2025.
- `CGRC-paper` seed2025.

Continue only if test cold item-macro N@10 improves over the fair FAST3 baseline and approaches or exceeds CGRC-paper without collapsing hot N@10.
