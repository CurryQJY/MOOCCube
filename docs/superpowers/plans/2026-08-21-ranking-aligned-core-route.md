# Ranking-Aligned Core Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace saturated auxiliary supervision with opt-in ranking-aligned course and pseudo-cold losses, then apply a strict component gate on Junyi validation.

**Architecture:** Add two pure contrastive loss functions beside the legacy losses. Select them through explicit core-runner arguments, leaving historical defaults unchanged. Reuse the anchored Ridge and strict validation implementation from Route A.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing GraphContentScorer runner.

---

### Task 1: Add contrastive loss contracts

**Files:**
- Modify: `tests/test_graph_course_core_finetune_pilot.py`
- Modify: `graph_course_core_finetune_pilot.py`

- [ ] Write tests showing course contrastive loss remains positive with a
  nonzero item-bank gradient when the legacy margin loss is zero.
- [ ] Write tests showing pseudo contrastive loss updates only the masked bank,
  leaves the factual bank detached, and returns scalar zero for an empty mask.
- [ ] Run the focused tests and confirm RED due to missing functions.
- [ ] Implement deterministic endpoint-excluding negatives, weighted softplus
  course ranking, and catalog cross-entropy pseudo ranking.
- [ ] Run the focused tests and confirm GREEN.

### Task 2: Wire opt-in modes without changing defaults

**Files:**
- Modify: `tests/test_graph_course_core_finetune_pilot.py`
- Modify: `graph_course_core_finetune_pilot.py`

- [ ] Test parser defaults `margin/cosine/0.2` and explicit
  `contrastive/contrastive/0.2` values.
- [ ] Test `fine_tune_step` rejects unknown modes and non-positive temperature.
- [ ] Confirm RED.
- [ ] Add `course_loss_mode`, `pseudo_loss_mode`, and `aux_temperature` to the
  fine-tune dispatch and command-line parser.
- [ ] Run the complete core and Ridge test files.

### Task 3: Execute the seed-2026 component gate

**Files:**
- Create outputs under: `outputs/xds_junyi_rankaux_a075/`

- [ ] Run graph-only with legacy defaults and `--skip-downstream`.
- [ ] Run pseudo-only with `--pseudo-loss-mode contrastive` and
  `--skip-downstream`.
- [ ] Run Full with both loss modes set to `contrastive` and
  `--skip-downstream`.
- [ ] Compare selected validation cold NDCG@10 and verify all retention floors,
  `ridge_alpha=0.075`, and `test_loaded=false`.
- [ ] Advance to three seeds only when both incremental component gates are
  positive; otherwise stop without test access.

### Task 4: Verify and report

**Files:**
- Inspect: `graph_course_core_finetune_pilot.py`
- Inspect: `tests/test_graph_course_core_finetune_pilot.py`
- Inspect: `outputs/xds_junyi_rankaux_a075/`

- [ ] Run both complete related pytest files with a workspace-local basetemp.
- [ ] Run `py_compile` on both runners.
- [ ] Confirm no experiment output loaded or wrote a test artifact.
- [ ] Record the automatic route verdict and exact component deltas.
