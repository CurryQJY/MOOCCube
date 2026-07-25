# CGRC Formal Timing Supervision Design

## Goal

Make the formal CGRC timing queue recover safely from an unexpected child-process exit without changing its fixed timing protocol or reusing incomplete measurements.

## Evidence and constraints

- The interrupted queue completed all five Junyi profiles, then stopped during COCO seed `9101` after epochs 51 and 52. It produced no profile JSON and no recorded queue exit code.
- `run_cgrc_controlled_timing.ps1` already validates completed profiles, skips them on a later invocation, archives an interrupted `run.log`, and recreates the telemetry CSV for an incomplete run.
- The formal protocol is fixed: datasets are Junyi, COCO, and MOOCCube; timing seeds are 9101--9105; each run resumes from epoch 50, warms up for 10 epochs, and times 20 epochs.
- Existing auto-start code verifies only initial launch, then exits. It cannot record a later exit or restart an incomplete queue.

## Recommended architecture

Add a standalone PowerShell supervisor, `supervise_cgrc_timing.ps1`, between every launch entry point and `run_cgrc_controlled_timing.ps1`.

The supervisor owns a lock under `<TimingOutputRoot>\_supervisor`, starts the unchanged formal queue as a child process, and writes:

- an append-only supervisor log;
- an atomically replaced JSON status file containing the attempt number, child PID, start/end times, child exit code, and the latest queue-log update time;
- a periodic heartbeat while the child is alive.

The existing watcher and detached auto-start helper will invoke the supervisor rather than the formal queue directly. The formal queue remains the sole owner of checkpoints, profiles, and telemetry.

## Failure and recovery policy

- A zero child exit code marks the supervisor successful. The formal queue itself remains responsible for validating all result profiles before it returns zero.
- A nonzero exit code is recorded with the child stdout/stderr paths. If attempts remain, the supervisor waits 60 seconds and starts a fresh queue invocation.
- `MaxRestarts` defaults to 3, meaning one initial attempt plus at most three recovery attempts. On exhaustion, the supervisor exits nonzero and leaves its log, status file, and child logs intact.
- A queue log that has not changed for 20 minutes is recorded as stale and surfaced in the status/log. The supervisor does not kill a live child solely because of staleness; a legitimate CGRC epoch can be slow.
- The lock prevents two supervisors from launching concurrent formal queues. A stale marker alone must not block recovery after a prior process has exited.

## Data integrity and idempotency

- Completed timing profiles are never rerun because the unchanged formal queue validates and skips them.
- An incomplete profile is restarted from the fixed epoch-50 checkpoint, not resumed mid-measurement. Its prior `run.log` is archived by the formal queue and its telemetry CSV is recreated, so partial samples cannot contaminate a formal result.
- No automatic recovery runs while a matching formal queue process is already alive.
- `-DryRun` exercises the supervisor's preflight and status paths without starting Python or changing profile artifacts.

## Test plan

Use temporary fixture scripts to verify:

1. A failed first child attempt is logged and retried exactly once when the next child succeeds.
2. Reaching the configured restart limit returns a nonzero exit and preserves terminal status.
3. A held lock rejects a second supervisor without launching another child.
4. Heartbeat/status records include a running child PID and a completed exit code.
5. The existing static-control watcher and detached auto-start path target the supervisor and retain `-TimingOnly`.
6. Existing controlled-timing and telemetry tests still pass, including the protocol and completed-profile validation guards.

## Non-goals

- Changing model hyperparameters, dataset order, timing seeds, checkpoints, or measurement windows.
- Killing a live training process based only on elapsed time.
- Creating or enabling a Windows Scheduled Task, changing power settings, or starting the actual GPU experiment.

## Success criteria

A future unexpected queue exit results in a durable record of the failure and a bounded, duplicate-safe recovery attempt. A normal run produces the same formal profiles and summary as the current launcher.
