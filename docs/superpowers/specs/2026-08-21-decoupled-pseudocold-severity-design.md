# Decoupled Pseudo-Cold Severity Design

## Goal

Keep the evidence-supported real-cold initializer fixed at backbone-anchored
Ridge `alpha=0.075`, while restoring a nontrivial pseudo-cold training task for
PPO with a separately frozen simulation coefficient `alpha_sim=1.0`.

## Diagnosis

The current runner applies one `ridge_alpha` to two different objects:

1. strict-cold rows used for validation and deployment; and
2. held-out warm rows used to simulate cold items during PPO training.

At `alpha=0.075`, the COCO pseudo-cold starting rows have mean cosine about
`0.9985` with their factual targets. Every deterministic inference rollout
therefore selects END and produces zero displacement, even though the same
PPO route was active when the historical full-Ridge simulation had cosine
about `0.706`. Meanwhile, validation-only evaluation shows that the real-cold
anchored Ridge itself improves Cold NDCG@10 on all three COCO seeds. The failure
is therefore caused by coupling deployment initialization strength to simulator
difficulty, not by the anchored Ridge or the four-metric retention gate.

## Alternatives

### Dataset-conditional real Ridge alpha

Allow COCO to use a larger real-cold coefficient. This restores PPO activity
but makes the method definition depend on dataset identity and weakens the
cross-dataset mechanism claim.

### Validation-selected real Ridge alpha

Choose the deployment coefficient from a grid on each validation split. This
is defensible as model selection but adds another selection layer and does not
address why pseudo-cold training becomes an identity task.

### Decoupled simulator severity (selected)

Use one coefficient for the real-cold initializer and one for pseudo-cold
simulation. Fix `alpha=0.075` and `alpha_sim=1.0` for every dataset. This
changes only the difficulty of the self-supervised training problem; inference
still starts from the same anchored Ridge bank.

## Interface And Data Flow

- Add `--simulation-ridge-alpha`, defaulting to unset.
- Resolve an unset value to `--ridge-alpha` so historical commands reproduce
  their previous behavior exactly.
- Use `ridge_alpha` only when blending strict-cold Ridge rows.
- Use `simulation_ridge_alpha` only when blending pseudo-cold held-out warm
  rows.
- Record both resolved values in `run_manifest.json` under explicit real and
  simulator initializer records.
- Thread the option through `graph_course_core_finetune_pilot.py` so the full
  staged route has one reproducible command surface.

No test dataframe is loaded during route development or selection.

## Validation Protocol

1. Unit-test parser defaults, explicit resolution, manifest provenance, and
   downstream argument forwarding before changing production code.
2. Run COCO seed 2025 with real `alpha=0.075`, simulation `alpha_sim=1.0`, the
   existing five PPO epochs, delta grid, reward, and four-metric retention gate.
3. Accept the mechanism hypothesis only if a nonzero epoch and nonzero delta are
   selected, Cold NDCG@10 improves over anchored Ridge, and all four retention
   metrics pass.
4. If the single-seed screen passes, run the same frozen pair on all three
   seeds of MOOCCube, Junyi, and COCO. Require positive mean PPO Cold NDCG@10,
   at least two positive seeds, and the existing per-seed four-metric gate on
   each dataset.

These existing validation splits are exploratory development data after this
change. A paper-level generalization claim requires newly generated blind
splits and matching source checkpoint training after the route is frozen.

## Scope

This change does not alter PPO rewards, course signals, policy architecture,
Ridge fitting, validation selection, delta grids, test-release code, or any
historical output artifact.
