# Warmup Stage Checkpoint Design

## Goal

Save a reusable checkpoint at the supervised-to-RecPPO boundary and allow multiple PPO branches to start from the exact same warmup model, supervised optimizer, and RNG state without inheriting PPO optimizer momentum.

## Behavior

- At `recppo_warmup_epochs`, save `warmup_stage.pt` beside `latest.pt`.
- Preserve the existing `latest.pt` same-configuration resume path.
- A branch may set `USIM_FB_WARMUP_STAGE_CKPT` to a stage checkpoint file.
- Restore model state, outer supervised optimizer state, and RNG state.
- Do not restore RecPPO actor/critic optimizer state; initialize it for the branch configuration.
- Permit branch-only changes to total epochs, patience, residual scale, PPO loss weight, and PPO optimizer hyperparameters.
- Reject changes to data, split, seed, architecture, pseudo-cold configuration, course signals, or warmup length.
- Never overwrite `warmup_stage.pt` during PPO training.

## Safety

- Stage loading is used only when normal same-directory resume did not occur.
- Invalid, incomplete, or incompatible stage checkpoints fail with an explicit error.
- Existing runs and checkpoint formats remain readable.

## Tests

- Stage compatibility ignores approved PPO branch controls.
- Stage compatibility rejects pseudo-cold or split changes.
- Stage state excludes the RecPPO optimizer while retaining outer optimizer and RNG state.
- Existing normal resume behavior remains unchanged.
