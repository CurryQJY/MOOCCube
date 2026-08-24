# CBI Soft-Anchor Simulator Design

## Goal

Run one isolated MOOCCube seed-2025 experiment that removes the training-time ID-embedding target without adding the hard content-cone projection used by the prior trust experiment.

## Method

The shared FAST3/CBI implementation remains unchanged. An isolated subclass intercepts `run_usim_episode` and ignores the caller-provided `target_emb=id_e_true`. It calls the unchanged parent simulator with `target_emb=init_item_emb.detach()`, so target alignment, terminal reward, step-gain reward, duplicate penalty, coverage bonus, and course rewards remain active while the target becomes the initial CBI course representation.

No per-step trust-domain projection is applied. This isolates the effect of replacing the ID target from the effect of hard projection.

## Evaluation

Validation and test use one deterministic all-item bank. Cold courses use `force_cold=True`; hot courses use `force_cold=False`; both run the simulator with inference semantics and positives are indexed from the same cached bank used for candidate ranking.

## Locked experiment

- Dataset: MOOCCube
- Protocol: `strict_item_cold_balanced`
- Seed: 2025
- Epoch ceiling: 60
- Early-stop patience: 10
- Early-stop score: cold item-macro NDCG@10
- Content-delta maximum norm: 0.5
- Simulator steps: 5
- Output/checkpoint/log roots are unique to `cbi_anchor_sim_single_seed2025`
- Shared main-table code is hash-audited but not modified
- Paper source files are not treated as runtime dependencies, avoiding false failures from concurrent manuscript edits

## Success criteria

The experiment must prove that the simulator receives the initial CBI representation as its effective target regardless of the supplied ID target, pass the existing all-item evaluation regression tests, save a resumable checkpoint, and produce Cold/Hot item-macro full-ranking metrics for comparison with original CBI and test-only all-item refinement.
