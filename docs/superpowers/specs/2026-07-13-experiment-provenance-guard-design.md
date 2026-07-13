# Experiment Provenance Guard Design

## Goal

Make formal FAST3/RecPPO experiments reproducible without blocking legitimate checkpoint continuation after source-only maintenance changes. Every run must preserve the exact source context used to start it, while checkpoint compatibility is decided by training configuration and data split rather than raw source bytes.

## Scope

This design covers the FAST3 static experiment runner, static protocol manifest, and feedback checkpoint fingerprint. It does not change model behavior, training defaults, dataset construction, metric calculation, or the currently running recovery experiment.

## Provenance Snapshot

At the first start of each concrete run directory, create a `provenance/` directory containing:

- the selected Python entrypoint;
- all `fast3_delta/*.py` modules;
- the PowerShell runner used to launch the experiment;
- `source_manifest.json`, with the relative path, size, and SHA256 of every captured file;
- runtime metadata: Git commit and dirty state, Python, PyTorch, CUDA, platform, and the normalized launch configuration.

The initial snapshot is immutable. A resume never overwrites it. On every resume, write a timestamped current-source manifest under `provenance/resume_checks/` so source drift remains auditable.

Snapshot failure is fatal for a new formal run because a run without preserved provenance cannot be reproduced. A resume may proceed only after its compatibility checks and resume manifest have been written successfully.

## Compatibility Fingerprints

Store independent fingerprints instead of one coarse fingerprint:

### Training configuration fingerprint

Hash a normalized, explicitly versioned dictionary of values that affect model state or optimization, including:

- seed and epoch-independent stochastic controls;
- model architecture and embedding dimensions;
- enabled components and their weights;
- PPO, simulator, pseudo-cold, course-reward, negative-sampling, and optimizer settings;
- batch size and other settings that change the optimization trajectory.

Exclude operational values that do not change training semantics, such as output paths, checkpoint paths, logging frequency, requested final epoch, patience extension, and source file location. Increasing the epoch ceiling or patience therefore remains resumable.

### Data split fingerprint

Hash the canonical split identity, including the split mode, seed, thresholds, split parameters, assignment hashes, and train/validation/test artifact hashes. A split mismatch is never resumable.

### Source fingerprint

Record per-file hashes for diagnostics only. Source mismatches do not independently invalidate a checkpoint.

The fingerprint schema has an explicit version number. A schema-version mismatch is treated like an unknown legacy checkpoint and requires the legacy override.

## Resume Rules

Checkpoint loading follows this order:

1. If the training configuration fingerprint differs, reject the checkpoint with a field-level difference report.
2. If the data split fingerprint differs, reject the checkpoint with the differing split fields or artifact hashes.
3. If configuration and split match but source hashes differ, print a prominent warning listing added, removed, and modified files, then allow the resume.
4. If everything matches, resume normally.
5. If the checkpoint lacks the new fingerprints, reject it by default. Permit it only through an explicit legacy compatibility switch, and record that override in the run manifest.

No general-purpose "ignore all mismatches" switch will be added. Overrides must not bypass known configuration or split mismatches.

## Manifest Integration

Extend `static_protocol_manifest.json` with:

- `provenance.schema_version`;
- snapshot directory and source-manifest hash;
- training configuration hash and normalized payload;
- split hash and normalized payload;
- Git/runtime metadata;
- resume decision, warnings, and any legacy override.

Keep the existing top-level fields for compatibility with current analysis scripts.

## Failure Handling

- New run cannot create its immutable snapshot: stop before training.
- Resume cannot write its audit record: stop before loading the checkpoint.
- Configuration or split mismatch: stop and print actionable field-level differences.
- Source mismatch only: warn, record the difference, and continue.
- Existing snapshot unexpectedly differs from its own manifest: treat the run directory as corrupted and stop.

## Testing

Add focused tests for:

- snapshot creation and immutability;
- complete source manifest generation;
- identical config/split accepting a checkpoint;
- source-only changes producing a warning but allowing resume;
- configuration changes rejecting resume;
- split or assignment changes rejecting resume;
- epoch or patience extension remaining resumable;
- legacy checkpoint rejection and explicit legacy compatibility;
- corrupted snapshot detection;
- existing static manifest consumers continuing to work.

## Rollout

Introduce the guard without changing current model defaults. Existing checkpoints remain loadable only through the explicit legacy compatibility switch. New formal experiments automatically receive the snapshot and versioned fingerprints. The currently running recovery validation is not modified or restarted.
