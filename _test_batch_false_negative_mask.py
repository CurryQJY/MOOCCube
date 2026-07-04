"""Tests for batch-level false-negative masking in FAST3 ContentDelta.

The main InfoNCE loss uses a B x B batch matrix. If the same target course
appears more than once in a batch, off-diagonal same-item cells are positives
for other rows but look like negatives unless explicitly masked.

Run:
    python _test_batch_false_negative_mask.py
"""
import os

os.environ["USIM_FORCE_CPU"] = "1"
os.environ.setdefault("USIM_DISABLE_LLM_SCORE", "1")

import torch

import usim_feedback_fast3_content_delta as M


def make_dummy_model(n_users=8, n_items=10, content_dim=8):
    cfg = M.Fast3Config(n_users=n_users, n_items=n_items, content_dim=content_dim)
    cfg.batch_size = 4
    cfg.n_epochs = 1
    content_emb = torch.zeros((n_items, content_dim), dtype=torch.float32)
    model = M.Fast3FeedbackUSIM(cfg, content_emb)
    model.device = torch.device("cpu")
    return cfg, model


def test_same_item_off_diagonal_cells_are_masked():
    cfg, model = make_dummy_model()
    cfg.mask_same_item_neg = True
    cfg.mask_known_pos_neg = False

    user_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    item_idx = torch.tensor([3, 5, 3, 7], dtype=torch.long)
    pos_mask = torch.eye(item_idx.numel(), dtype=torch.bool)

    mask = model._build_batch_false_negative_mask(user_ids, item_idx, None, pos_mask)

    assert mask is not None, "same-item masking should produce a mask"
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    assert bool(mask[0, 2].item()), "row 0 should mask duplicate item 3 at column 2"
    assert bool(mask[2, 0].item()), "row 2 should mask duplicate item 3 at column 0"
    assert not bool(mask.diag().any().item()), "diagonal positives must not be marked as false negatives"
    assert int(mask.sum().item()) == 2, f"expected exactly two same-item off-diagonal masks, got {mask}"
    print("[PASS] same target item off-diagonal cells are masked")


def test_known_positive_and_same_item_masks_are_combined():
    cfg, model = make_dummy_model()
    cfg.mask_same_item_neg = True
    cfg.mask_known_pos_neg = True

    user_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    item_idx = torch.tensor([4, 6, 4], dtype=torch.long)
    pos_mask = torch.eye(item_idx.numel(), dtype=torch.bool)
    user_seen_items = {
        0: {6},  # row 0: item 6 at column 1 is a known positive, not a negative.
        1: set(),
        2: set(),
    }

    mask = model._build_batch_false_negative_mask(user_ids, item_idx, user_seen_items, pos_mask)

    assert bool(mask[0, 1].item()), "known positive item 6 for user 0 should be masked"
    assert bool(mask[0, 2].item()), "same item 4 should be masked for row 0"
    assert bool(mask[2, 0].item()), "same item 4 should be masked for row 2"
    assert not bool(mask.diag().any().item()), "diagonal positives must stay unmasked"
    assert int(mask.sum().item()) == 3, f"expected three combined masks, got {mask}"
    print("[PASS] known-positive and same-item masks are combined")


def test_mask_can_be_disabled_for_legacy_path():
    cfg, model = make_dummy_model()
    cfg.mask_same_item_neg = False
    cfg.mask_known_pos_neg = False

    user_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    item_idx = torch.tensor([4, 4, 5], dtype=torch.long)
    pos_mask = torch.eye(item_idx.numel(), dtype=torch.bool)

    mask = model._build_batch_false_negative_mask(user_ids, item_idx, None, pos_mask)

    assert mask is None, "all false-negative masks disabled should preserve legacy no-mask behavior"
    print("[PASS] batch false-negative mask can be disabled")


def test_candidate_logits_are_masked_after_gather():
    _, model = make_dummy_model()
    cand_logits = torch.tensor(
        [
            [2.0, 1.0, 3.0],
            [2.5, 4.0, 0.5],
        ]
    )
    cand_idx = torch.tensor(
        [
            [0, 1, 2],
            [1, 0, 2],
        ],
        dtype=torch.long,
    )
    false_neg_mask = torch.tensor(
        [
            [False, False, True],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    masked = model._mask_false_negative_candidate_logits(cand_logits, cand_idx, false_neg_mask)

    assert masked[0, 0].item() == cand_logits[0, 0].item(), "target logit should stay untouched"
    assert masked[1, 0].item() == cand_logits[1, 0].item(), "target logit should stay untouched"
    assert masked[0, 2].item() < -1e8, "false-negative candidate should be suppressed after gather"
    assert masked[1, 1].item() < -1e8, "false-negative candidate should be suppressed after gather"
    assert masked[0, 1].item() == cand_logits[0, 1].item(), "valid candidate should stay unchanged"
    print("[PASS] gathered candidate logits suppress false negatives")


if __name__ == "__main__":
    print("=" * 64)
    print("batch false-negative mask tests")
    print("=" * 64)
    test_same_item_off_diagonal_cells_are_masked()
    test_known_positive_and_same_item_masks_are_combined()
    test_mask_can_be_disabled_for_legacy_path()
    test_candidate_logits_are_masked_after_gather()
    print("=" * 64)
    print("All batch false-negative mask tests passed.")
