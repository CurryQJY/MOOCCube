# Simulator-Training Factorial Completion Design

## Goal

Complete the missing `simulator training T=0 + course-fit inference T=5` cell using the three frozen recovered `wo_simulator` checkpoints, without retraining or changing checkpoint provenance.

## Fixed 2×2

| Training horizon | Static inference | Course-fit inference |
|---|---|---|
| T=5 | recovered main-table static | validation-selected course-fit test result |
| T=0 | recovered `wo_simulator` static | new missing cell |

All cells use the same scripts, data, strict-cold split assignments, course metadata, and three seeds. The only training difference is `USIM_STEPS=5` versus `0`.

## Safe inference override

The new cell must launch with `UsimSteps=0` so checkpoint fingerprint validation remains exact. After the checkpoint has loaded, the evaluation wrapper temporarily sets `cfg.usim_steps=5` only around the deterministic course-fit rollout, records checkpoint/effective step values in the audit, and restores `cfg.usim_steps=0` in `finally`.

The rollout remains `course_fit`, uses residual 1.0, never calls Actor/Critic, receives no behavior target, and uses train-only user histories. Source checkpoints are hash-protected before and after evaluation.

## Interpretation

- T=0 and T=5 training perform similarly under course-fit: course-fit is primarily post-hoc inference refinement.
- T=5 is consistently better under course-fit: simulator training improves the representation path used by inference.
- T=0 course-fit fails to beat T=0 static: course-fit depends on simulator-trained components.

This experiment isolates simulator training, not the existence of the inference-time state transition itself.
