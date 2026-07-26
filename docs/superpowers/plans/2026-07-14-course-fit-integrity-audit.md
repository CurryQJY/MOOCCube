# Course-Fit Integrity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a read-only integrity audit proving that frozen-checkpoint course-fit inference uses train-only histories, never observes strict-cold targets in candidate histories, and is invariant to target exclusion.

**Architecture:** Instrument only the existing evaluation wrapper by wrapping split construction, history installation, and course-fit scoring. Store provenance counters in the existing JSON audit, run a separate `exclude_target=True` replay, and compare it to the already completed test outputs with a small deterministic reporter.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest, PowerShell, existing recovered static runner.

---

### Task 1: History and target-seen instrumentation

**Files:**
- Modify: `tests/test_main_checkpoint_actor_inference_ab.py`
- Modify: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write failing tests**

Add tests that call wished-for `history_fingerprint`, `make_audited_split`, `audited_set_user_seen_index`, and `audited_compute_candidate_course_fit`. Assert equal mappings have equal fingerprints, the split records train/validation/test sets, train history is classified exactly, and a synthetic candidate history containing its target increments the target-seen counters.

- [ ] **Step 2: Run RED**

Run: `.\py.bat -m pytest tests\test_main_checkpoint_actor_inference_ab.py -q --basetemp .pytest_tmp\course_fit_audit_red`

Expected: FAIL because the audit APIs and counters are absent.

- [ ] **Step 3: Implement minimal instrumentation**

Extend `InferenceAudit` with split, history-source, target-seen, refined-item, behavior-target, and effective-exclusion fields. Implement deterministic SHA-256 history fingerprints and wrappers that call the original methods after recording observations. Install them only from `main()` so importing the module does not mutate the recovered implementation.

- [ ] **Step 4: Run GREEN**

Run the focused wrapper tests and expect all tests to pass.

### Task 2: Exclusion override and enriched audit export

**Files:**
- Modify: `tests/test_main_checkpoint_actor_inference_ab.py`
- Modify: `main_checkpoint_actor_inference_ab.py`

- [ ] **Step 1: Write failing tests**

Test optional boolean parsing, temporary exclusion override/restoration, refined item classification, and JSON export of derived rates and history-source counts.

- [ ] **Step 2: Run RED**

Run only the new test names and expect missing-field or missing-function failures.

- [ ] **Step 3: Implement the override and export**

Read `USIM_COURSE_MATCH_EXCLUDE_TARGET` as an optional boolean. Apply it only around `run_usim_episode`, restore the original checkpoint value in `finally`, collect refined IDs, and derive the composition/rates in `write_audit`.

- [ ] **Step 4: Run GREEN and regression tests**

Run the complete wrapper and legacy inference-probe suites.

### Task 3: Deterministic integrity reporter

**Files:**
- Create: `tests/test_course_fit_integrity_report.py`
- Create: `course_fit_integrity_report.py`

- [ ] **Step 1: Write failing report tests**

Create temporary baseline/audit seed directories. Assert the reporter passes identical aggregate/per-item outputs, rejects a target-seen count above zero, and rejects any metric difference.

- [ ] **Step 2: Run RED**

Run: `.\py.bat -m pytest tests\test_course_fit_integrity_report.py -q --basetemp .pytest_tmp\course_fit_report_red`

Expected: FAIL because the reporter module is absent.

- [ ] **Step 3: Implement the reporter**

Load each seed's final full-ranking CSV, per-item cold CSV, and enriched audit JSON. Emit `course_fit_integrity_by_seed.csv` and `course_fit_integrity_summary.json`; return non-zero when any pass criterion fails.

- [ ] **Step 4: Run GREEN**

Run the report tests and expect all tests to pass.

### Task 4: Read-only three-seed runner

**Files:**
- Create: `run_course_fit_integrity_audit.ps1`

- [ ] **Step 1: Add the runner**

Replay seeds 2025/2026/2027 with the exact recovered main-table configuration, `course_fit`, test targeting, fixed RNG 7001, `T=5`, residual 1.0, and `USIM_COURSE_MATCH_EXCLUDE_TARGET=true`. Write only under `outputs/recppo_research_repair/course_fit_integrity_audit/exclude_target_true` and then invoke the reporter against `test_course_fit_frozen/course_fit`.

- [ ] **Step 2: Parse-check and dry-run**

Use the PowerShell parser and an unavailable seed. Expect zero parser errors and a clean skip.

### Task 5: Verification and audit execution

**Files:**
- Verify all files above.
- Generate only the new audit output root.

- [ ] **Step 1: Run focused tests**

Run wrapper, reporter, and legacy probe tests with a workspace-local pytest base directory.

- [ ] **Step 2: Record checkpoint hashes/timestamps**

Hash the three `finished.pt` files before the audit.

- [ ] **Step 3: Run the three-seed audit**

Run: `& .\run_course_fit_integrity_audit.ps1`

- [ ] **Step 4: Recheck checkpoints and read the report**

Require unchanged hashes/timestamps and a reporter exit code of zero. Report any failed criterion directly rather than interpreting performance.
