# CBI Soft-Anchor Three-Seed Serial Design

## Goal

Run the validated CBI soft-anchor, no-hard-projection method on MOOCCube seeds 2025, 2026, and 2027 under one identical configuration and aggregate the resulting full-ranking metrics.

## Execution model

The existing `run_cbi_anchor_sim_seed2025.py` entrypoint and `CBIAnchorFast3FeedbackUSIM` model are reused unchanged. A new PowerShell launcher passes `Seeds = @(2025, 2026, 2027)` to the existing static runner, whose nested seed loop executes one seed at a time on the same GPU.

Each seed receives a separate static split, output directory, and checkpoint directory. Auto-resume is enabled, so an interrupted launcher can skip completed epochs and continue the current seed without mixing checkpoint state across seeds.

## Locked configuration

- Seeds: 2025, 2026, 2027
- Epoch ceiling: 60
- Patience: 10
- Early-stop score: cold item-macro NDCG@10
- Delta maximum norm: 0.5
- Five simulator steps
- Training target: initial CBI representation
- Hard projection: disabled
- Validation/test: deterministic all-item refined shared bank
- Serial execution only
- Independent roots under `cbi_anchor_sim_3seed_serial`

## Outputs

The static runner writes per-seed reports and checkpoints, then produces `fast3_static_runs_detail.csv` and `fast3_static_multiseed_summary.csv` containing the three-seed aggregate. A top-level manifest records the locked configuration, source hashes, protected shared-code hashes, runtime status, and paths.

## Training and inference semantics

For each course, training starts from the normalized frozen-content-plus-bounded-delta representation. The simulator ignores the caller's ID target and uses this initial CBI vector as a detached soft target while retaining user alignment, terminal/gain rewards, diversity terms, course rewards, PPO, recommendation, InfoNCE, and prerequisite losses. No hard geometric projection is applied.

At validation and test time, model parameters are frozen. Cold courses use `force_cold=True`, hot courses use `force_cold=False`, and both are deterministically refined for five simulator steps with the same CBI soft-anchor behavior. One normalized item bank is reused for candidates and positives before Cold/Hot item-macro full ranking.
