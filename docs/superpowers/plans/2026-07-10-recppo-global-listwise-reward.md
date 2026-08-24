# RecPPO Global Listwise Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repaired RecPPO's local-candidate rank reward with a cached listwise reward over target Top-K users drawn strictly from users with training histories.

**Architecture:** The repaired model builds a train-user pool from `user_seen_items`, lazily caches target Top-K user IDs after the backbone freezes, and computes cross-entropy improvement on that fixed support. The legacy script and all non-rank reward terms remain unchanged.

**Tech Stack:** Python 3.12, PyTorch, pytest, PowerShell.

---

### Task 1: Lock Strict Train-User Semantics

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`

- [ ] Add a failing test whose bank contains a high-scoring user absent from `user_seen_items`; assert that `_recppo_train_user_pool` excludes it while retaining users with non-empty train histories.
- [ ] Run the focused test and confirm it fails because `_recppo_train_user_pool` does not exist.
- [ ] Implement `_recppo_train_user_pool(user_bank_norm, user_seen_items)` to return detached normalized vectors and original user IDs for non-empty train histories, with a hard failure when the pool is empty.
- [ ] Run the focused test to green.

### Task 2: Implement Cached Global Top-K Retrieval

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`

- [ ] Add a failing test with a globally relevant user outside local action candidates and assert that cached Top-K IDs contain that global user.
- [ ] Add a failing test that calls retrieval twice and asserts identical IDs plus an increased cache-hit counter.
- [ ] Run both tests and confirm the missing global cache behavior.
- [ ] Implement a lazy per-item cache keyed by integer item ID, using normalized target embeddings and the strict train-user pool; clear it in `_activate_recppo_phase`.
- [ ] Run both tests to green.

### Task 3: Replace Local Error With Listwise Gain

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`

- [ ] Add a failing test where `next_h` improves the teacher Top-K ordering relative to `prev_h`; assert positive gain.
- [ ] Add a failing test that changes only a local candidate outside global Top-K and assert unchanged listwise gain.
- [ ] Run both tests and confirm the old `_candidate_rank_gain` behavior fails them.
- [ ] Implement discounted teacher weights and `CE_teacher(prev)-CE_teacher(next)` over cached global Top-K users, using `cfg.temp` for score calibration and detached targets.
- [ ] Pass `item_idx`, the epoch user bank, and train histories into the new rank-gain path in `run_usim_episode`; do not change course reward code.
- [ ] Run both tests to green.

### Task 4: Provenance And Regression Verification

**Files:**
- Modify: `tests/test_usim_strict_cold_repair.py`
- Modify: `usim_feedback_fast3_content_delta_repaired.py`

- [ ] Add manifest assertions for `rank_reward_source=global_train_user_topk`, Top-K, and temperature.
- [ ] Add epoch diagnostics for cache hit rate and strict train-user pool size.
- [ ] Run `py.bat -m pytest tests/test_usim_strict_cold_repair.py -q` and expect all focused tests to pass.
- [ ] Run the existing core/static regression tests and syntax checks.
- [ ] Run a one-epoch smoke test, then the unchanged 30-epoch seed-2025 experiment with ContentDelta off.
- [ ] Compare cold item-macro metrics against `R@10=0.2636315487` and `N@10=0.1783931626`; select checkpoints only from validation metrics.
