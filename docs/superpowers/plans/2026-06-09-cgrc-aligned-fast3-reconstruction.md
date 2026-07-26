# CGRC-Aligned FAST3 Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default CGRC-style reconstruction auxiliary and candidate prior to FAST3 with full run traceability.

**Architecture:** Add the smallest possible reconstruction head inside the existing FAST3 model. It scores user/item pairs from the projected user bank and content-derived item vector, trains with pseudo-cold warm items using a positive-in-denominator softmax, and optionally biases candidate sampling for cold/tail rows.

**Tech Stack:** Python, PyTorch, PowerShell launcher, existing FAST3 static protocol and manifest utilities.

---

### Task 1: Configuration And Launcher Trace

**Files:**
- Modify: `D:/DeskTop/MOOCCube/fast3_delta/config.py`
- Modify: `D:/DeskTop/MOOCCube/run_usim_feedback_fast3_content_delta_static.ps1`

- [x] Add environment-backed config fields with defaults that preserve existing runs:

```python
self.use_cgrc_recon = os.environ.get("USIM_USE_CGRC_RECON", "0") == "1"
self.cgrc_recon_aux_weight = float(os.environ.get("USIM_CGRC_RECON_AUX_W", "0.0"))
self.cgrc_recon_sample_weight = float(os.environ.get("USIM_CGRC_RECON_SAMPLE_W", "0.0"))
self.cgrc_recon_pseudo_ratio = float(os.environ.get("USIM_CGRC_RECON_PSEUDO_RATIO", "0.30"))
self.cgrc_recon_topk = int(os.environ.get("USIM_CGRC_RECON_TOPK", "64"))
self.cgrc_recon_temperature = float(os.environ.get("USIM_CGRC_RECON_TEMP", "0.50"))
self.cgrc_recon_only_cold_or_tail = os.environ.get("USIM_CGRC_RECON_ONLY_COLD_OR_TAIL", "1") == "1"
self.cgrc_recon_tail_pop_ratio = float(os.environ.get("USIM_CGRC_RECON_TAIL_POP_RATIO", "0.10"))
self.cgrc_recon_detach_user = os.environ.get("USIM_CGRC_RECON_DETACH_USER", "0") == "1"
```

- [x] Add matching PowerShell parameters, tracked env vars, env assignment, and startup logging.

### Task 2: Reconstruction Head And Loss

**Files:**
- Modify: `D:/DeskTop/MOOCCube/usim_feedback_fast3_content_delta.py`

- [x] Add a small `cgrc_recon_mlp` to `PAM_RL_Pure_USIM`.
- [x] Add `_cgrc_recon_logits()` that broadcasts user and item vectors like CGRC's edge MLP.
- [x] Add `_compute_cgrc_recon_aux_loss()` that samples pseudo-cold warm rows, scores a top-k user pool plus the positive user, and computes cross entropy with the positive inside the denominator.

### Task 3: Candidate Prior Integration

**Files:**
- Modify: `D:/DeskTop/MOOCCube/usim_feedback_fast3_content_delta.py`

- [x] In `get_candidates()`, compute reconstruction logits for the retrieval pool when enabled.
- [x] Mix retrieval sampling probabilities with reconstruction probabilities for cold/tail rows only.
- [x] Report `cgrc_recon_sample_active` and `cgrc_recon_sample_score`.

### Task 4: Diagnostics, Manifest, Optimizer

**Files:**
- Modify: `D:/DeskTop/MOOCCube/usim_feedback_fast3_content_delta.py`

- [x] Add reconstruction parameters to optimizer automatically through existing `model.parameters()`.
- [x] Add manifest fields under `config`.
- [x] Add epoch diagnostics: `CGRCReconLoss`, `CGRCReconActive`, `CGRCReconPos`, `CGRCReconSampleActive`, `CGRCReconSampleScore`.

### Task 5: Smoke Tests

**Files:**
- Use: `D:/DeskTop/MOOCCube/py.bat`

- [x] Run syntax check:

```powershell
.\py.bat -m py_compile fast3_delta\config.py usim_feedback_fast3_content_delta.py
```

- [x] Run a short CPU/config smoke if time allows:

```powershell
$env:USIM_USE_CGRC_RECON='1'
$env:USIM_CGRC_RECON_AUX_W='0.02'
$env:USIM_CGRC_RECON_SAMPLE_W='0.10'
.\py.bat -m py_compile usim_feedback_fast3_content_delta.py
```
