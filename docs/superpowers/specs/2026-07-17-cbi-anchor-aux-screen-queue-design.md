# CBI Anchor Auxiliary-ID Screen Queue Design

## Goal

Automatically wait for the active CBI anchor three-seed baseline to complete, then screen `AuxWeight` values `0.0`, `0.1`, and `0.3` on seed 2025 without overlapping GPU work.

## Queue behavior

The queue polls `outputs/cbi_anchor_sim_3seed_serial/run_manifest.json`. It starts the auxiliary screen only when the upstream status is `completed`. If the upstream status becomes `failed`, the queue records the error and exits without launching any screen arm.

## Screen behavior

The screen runs three arms serially in the order `0.0`, `0.1`, `0.3`. Each arm uses the validated CBI soft-anchor entrypoint, seed 2025, 30 epochs, patience 6, cold item-macro NDCG@10 early stopping, delta norm 0.5, five simulator steps, and all-item deterministic refined evaluation. Each arm has an isolated output and checkpoint directory under `cbi_anchor_aux_screen_seed2025`.

The `0.3` arm is rerun instead of reusing the completed long run so all three arms share the same 30-epoch screening budget and queue runtime context. Test metrics are produced but must not be used for hyperparameter selection; selection uses validation Cold N@10 with Cold R@10 as the tie-breaker.

## Reproducibility

The screen and queue write manifests, record source hashes, protect shared training code, support checkpoint resume inside an interrupted arm, and never modify the active three-seed outputs.
