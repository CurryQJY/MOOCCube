# Official USIM 3-Dataset 3-Seed Serial Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run official USIM on MOOCCube, Junyi, and COCO for seeds 2025/2026/2027 with resumable checkpoints and per-seed logs.

**Architecture:** Extend `usim_official_static_hin.py` with stage checkpoints for BPR backbone, content mapper, and RL. Add a PowerShell serial queue script that maps each dataset/seed to an existing strict item-cold split, sets runner environment variables, writes logs, skips completed runs, and aggregates each dataset after completion.

**Tech Stack:** PowerShell, Python, PyTorch, existing static split artifacts, existing `aggregate_main_table_static_results.py`.

---

### Task 1: Checkpoint Support

**Files:**
- Modify: `usim_official_static_hin.py`
- Modify: `tests/test_usim_official_static_hin.py`

- [ ] Add config fields for `USIM_OFFICIAL_CKPT_DIR`, `USIM_OFFICIAL_SAVE_CKPT`, `USIM_OFFICIAL_AUTO_RESUME`, and `USIM_OFFICIAL_FORCE_FRESH`.
- [ ] Save and resume backbone, mapper, and RL checkpoints.
- [ ] Verify with the existing inline test harness and `py_compile`.

### Task 2: Serial Queue Script

**Files:**
- Create: `run_usim_official_3datasets_3seed_serial.ps1`
- Create: `tests/test_usim_official_3dataset_serial.ps1`

- [ ] Write a failing PowerShell test that expects dry-run to list nine dataset/seed tasks.
- [ ] Implement the serial runner with dry-run, skip-completed, checkpoint dirs, per-run logs, and per-dataset aggregation.
- [ ] Verify the test passes.

### Task 3: Launch

**Files:**
- No source changes.

- [ ] Run syntax checks and dry-run.
- [ ] Start the serial script in a hidden background PowerShell process.
- [ ] Report the PID plus master log path.
