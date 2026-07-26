# PAM Validation Checkpoint Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `test-driven-development` while executing this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the PAM adapter select one epoch using only validation-set cold item-macro NDCG@10, then evaluate the held-out test set once from that checkpoint.

**Architecture:** `pam_official_static.py` will persist a unique TensorFlow checkpoint after every trained epoch and evaluate only the validation cold targets at that point. A small deterministic selector will choose the highest validation `N@10` checkpoint, preserving the earliest epoch on a tie. The selected checkpoint will be restored before the existing cold/hot test evaluations; its decision record will be emitted in the result JSON.

**Tech Stack:** Python 3.12 unit tests on Windows; TensorFlow 1.x compatibility execution in WSL2 for the final experiment.

---

### Task 1: Define the checkpoint-selection contract

**Files:**
- Modify: `tests/test_pam_official_static.py`

- [x] **Step 1: Write a failing test**

Add a fake TensorFlow/PAM execution test that trains three fake epochs, supplies validation `N@10` values `[0.20, 0.40, 0.40]`, and asserts that the result selects epoch 2, writes three distinct checkpoint prefixes, restores epoch 2 exactly once, and performs exactly two test evaluations after restoration.

- [x] **Step 2: Run test to verify it fails**

Run: `& .\\py.bat tests\\test_pam_official_static.py -v`

Expected: FAIL because the adapter lacks per-epoch validation selection and currently saves only `pam_official_latest.ckpt`.

### Task 2: Implement the minimal training protocol

**Files:**
- Modify: `pam_official_static.py`
- Test: `tests/test_pam_official_static.py`

- [x] **Step 1: Add deterministic checkpoint names and selection helper**

Add helpers that produce `pam_official_epoch_<N>.ckpt` and select the first maximum `full_cold_item_macro["N@10"]` from epoch validation records.

- [x] **Step 2: Evaluate validation cold targets inside the epoch loop**

Load `pam_val_targets.csv`, evaluate its cold target subset after each saved checkpoint, and store the epoch, loss, checkpoint prefix, metrics, and counts in `validation_epoch_history`.

- [x] **Step 3: Restore the selected checkpoint before testing**

Restore the selector result once, run the existing cold and hot test evaluations once each, and return a JSON-serializable `checkpoint_selection` record plus the complete validation history.

- [x] **Step 4: Run test to verify it passes**

Run: `& .\\py.bat tests\\test_pam_official_static.py -v`

Expected: PASS, including the selection contract test.

### Task 3: Verify and run the experiment

**Files:**
- Modify: `pam_official_static.py`
- Test: `tests/test_pam_official_static.py`

- [x] **Step 1: Run the full PAM adapter unit-test file**

Run: `& .\\py.bat tests\\test_pam_official_static.py -v`

- [x] **Step 2: Launch one fresh MOOCCube seed-2025 WSL GPU run**

Run the existing serial runner with `-Datasets mooccube -Seeds 2025 -Epochs 3 -DisableAutoResumeFromSiblingEpoch`, a new output root, and `-Force`. Confirm the resulting JSON contains exactly three validation records and one selected checkpoint.

- [x] **Step 3: Inspect the selected checkpoint and test metrics**

Read the runner log and result JSON, verify the GPU was visible, validate the selector's chosen epoch against its validation values, and keep the test result out of the main table until all seeds have been run under this protocol.
