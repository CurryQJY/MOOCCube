"""Numerical equivalence test: legacy per-batch construction vs. user_seen_index fast path.

Runs in a few seconds on CPU, no real data required. Verifies that
`_build_seen_mat`, `_build_known_positive_batch_mask`, and the new
`set_user_seen_index` produce bit-identical outputs whether or not the
fast index is enabled.

Run:
    python _test_seen_index_fastpath.py
"""
import os
import sys

os.environ["USIM_FORCE_CPU"] = "1"
os.environ.setdefault("USIM_DISABLE_LLM_SCORE", "1")

import torch

import usim_feedback_fast3_content_delta as M


def make_dummy_model(n_users=50, n_items=20):
    cfg = M.Fast3Config(n_users=n_users, n_items=n_items, content_dim=8)
    cfg.batch_size = 4
    cfg.n_epochs = 1
    content_emb = torch.zeros((n_items, 8), dtype=torch.float32)
    model = M.Fast3FeedbackUSIM(cfg, content_emb)
    model.device = torch.device("cpu")
    return cfg, model


def _build_user_seen_dict(n_users, n_items, density=0.2, seed=0):
    rng = torch.Generator().manual_seed(seed)
    seen = {}
    for u in range(n_users):
        n = int(torch.randint(0, max(1, int(n_items * density)) + 1, (1,), generator=rng).item())
        if n == 0:
            continue
        items = torch.randperm(n_items, generator=rng)[:n].tolist()
        seen[u] = set(items)
    # Add a few out-of-range to verify they get filtered
    seen[0] = (seen.get(0, set()) | {n_items + 5, -1})
    return seen


def test_build_seen_mat_equivalence():
    cfg, model = make_dummy_model()
    seen = _build_user_seen_dict(cfg.n_users, cfg.n_items, density=0.3, seed=42)

    # Pick a deterministic batch of users (with duplicates and out-of-range)
    user_ids_list = [0, 1, 5, 17, 42, 0, 49, 5]
    user_ids_tensor = torch.tensor(user_ids_list, dtype=torch.long)

    # Legacy path (no index)
    model.user_seen_index = None
    legacy_mat, legacy_cnt = model._build_seen_mat(user_ids_tensor, seen)
    legacy_mask_list, _ = model._build_seen_mat(user_ids_list, seen)

    # Fast path (with index)
    model.set_user_seen_index(seen)
    fast_mat, fast_cnt = model._build_seen_mat(user_ids_tensor, seen)
    fast_mat_list, _ = model._build_seen_mat(user_ids_list, seen)

    assert torch.equal(legacy_mat, fast_mat), "_build_seen_mat tensor input mismatch"
    assert torch.equal(legacy_cnt, fast_cnt), "_build_seen_mat count mismatch"
    assert torch.equal(legacy_mask_list, fast_mat_list), "_build_seen_mat list input mismatch"
    print("[PASS] _build_seen_mat: legacy == fast (tensor + list input)")


def test_known_positive_mask_equivalence():
    cfg, model = make_dummy_model()
    seen = _build_user_seen_dict(cfg.n_users, cfg.n_items, density=0.3, seed=7)

    user_ids = torch.tensor([0, 3, 7, 11, 22, 0, 49], dtype=torch.long)
    item_idx = torch.tensor([0, 5, 1, 3, 12, 19, 4], dtype=torch.long)

    model.user_seen_index = None
    legacy = model._build_known_positive_batch_mask(user_ids, item_idx, seen)

    model.set_user_seen_index(seen)
    fast = model._build_known_positive_batch_mask(user_ids, item_idx, seen)

    # Both should be (B, B) bool tensors representing seen[user_i, item_j]
    assert legacy.shape == fast.shape, f"shape mismatch: legacy={legacy.shape}, fast={fast.shape}"
    assert torch.equal(legacy, fast), "_build_known_positive_batch_mask diverges"
    print("[PASS] _build_known_positive_batch_mask: legacy == fast")


def test_set_user_seen_index_correctness():
    cfg, model = make_dummy_model()
    seen = _build_user_seen_dict(cfg.n_users, cfg.n_items, density=0.4, seed=123)

    model.set_user_seen_index(seen)
    idx = model.user_seen_index
    assert idx.shape == (cfg.n_users, cfg.n_items)
    assert idx.dtype == torch.bool

    # Spot check: every (uid, item) in `seen` (within range) must map to True
    for uid, items in seen.items():
        for it in items:
            in_range = (0 <= uid < cfg.n_users) and (0 <= it < cfg.n_items)
            if in_range:
                assert bool(idx[uid, it].item()), f"missing seen[{uid},{it}]"

    # And every True cell must come from `seen`
    nz = idx.nonzero(as_tuple=False)
    for uid, it in nz.tolist():
        assert it in seen.get(uid, set()), f"spurious seen[{uid},{it}]"
    print(f"[PASS] set_user_seen_index: {nz.size(0)} cells, dtype/bounds/contents all correct")


def test_clear_index():
    cfg, model = make_dummy_model()
    seen = _build_user_seen_dict(cfg.n_users, cfg.n_items, density=0.3, seed=1)
    model.set_user_seen_index(seen)
    assert model.user_seen_index is not None
    model.set_user_seen_index(None)
    assert model.user_seen_index is None
    print("[PASS] set_user_seen_index(None) clears the index")


def test_apply_course_rerank_equivalence():
    """When use_course_rerank=False (current default), apply_course_rerank is a no-op
    on both paths -- this just confirms no regression on the no-op shortcut.
    """
    cfg, model = make_dummy_model()
    cfg.use_course_rerank = False
    seen = _build_user_seen_dict(cfg.n_users, cfg.n_items, density=0.3, seed=99)
    model.set_user_seen_index(seen)

    user_ids = [0, 3, 7]
    scores = torch.randn(3, cfg.n_items)
    out = model.apply_course_rerank(scores, user_ids, {}, cand_idx=None, target_pop=None)
    assert torch.equal(scores, out), "rerank no-op path changed scores"
    print("[PASS] apply_course_rerank no-op shortcut preserved")


if __name__ == "__main__":
    print("=" * 60)
    print("Fast-path numerical-equivalence tests")
    print("=" * 60)
    test_set_user_seen_index_correctness()
    test_clear_index()
    test_build_seen_mat_equivalence()
    test_known_positive_mask_equivalence()
    test_apply_course_rerank_equivalence()
    print("=" * 60)
    print("All tests passed.")
