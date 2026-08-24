import pandas as pd
import torch

from sagerec_static_baseline import (
    SageRecConfig,
    SageRecModel,
    build_padded_user_history,
)


def test_build_padded_user_history_sorts_by_time_and_masks_padding():
    train_df = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 0, 2],
            "i_idx": [4, 5, 6, 7, 8],
            "timestamp": [30, 10, 20, 40, 50],
        }
    )

    hist, mask = build_padded_user_history(train_df, n_users=4, max_hist_len=2, pad_item_id=9)

    assert hist.tolist() == [
        [4, 7],
        [6, 9],
        [8, 9],
        [9, 9],
    ]
    assert mask.tolist() == [
        [True, True],
        [True, False],
        [True, False],
        [False, False],
    ]


def test_sagerec_scores_all_items_with_item_dependent_gate():
    cfg = SageRecConfig(
        n_users=3,
        n_items=5,
        content_dim=4,
        emb_dim=6,
        sample_top_n=2,
        max_hist_len=3,
        bucket_count=4,
    )
    model = SageRecModel(cfg, content_emb=torch.randn(5, 4))
    hist = torch.tensor([[0, 1, 4], [2, 4, 4]], dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    users = torch.tensor([0, 1], dtype=torch.long)
    item_idx = torch.arange(5, dtype=torch.long)
    popularity = torch.tensor([0, 1, 3, 7, 10], dtype=torch.float32)

    scores = model.score_items(users, item_idx, hist, mask, popularity)
    gate = model.gate_weights_for_items(item_idx, popularity)

    assert scores.shape == (2, 5)
    assert gate.shape == (5, 2)
    assert torch.allclose(gate.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.isfinite(scores).all()


def test_sagerec_forward_scores_positive_and_negative_items():
    cfg = SageRecConfig(
        n_users=3,
        n_items=6,
        content_dim=3,
        emb_dim=5,
        sample_top_n=2,
        max_hist_len=3,
        bucket_count=3,
    )
    model = SageRecModel(cfg, content_emb=torch.randn(6, 3))
    users = torch.tensor([0, 1], dtype=torch.long)
    pos = torch.tensor([2, 3], dtype=torch.long)
    neg = torch.tensor([4, 5], dtype=torch.long)
    hist = torch.tensor([[0, 1, 5], [2, 5, 5]], dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    popularity = torch.tensor([0, 2, 4, 6, 8, 10], dtype=torch.float32)

    pos_scores, neg_scores = model(users, pos, neg, hist, mask, popularity)

    assert pos_scores.shape == (2,)
    assert neg_scores.shape == (2,)
    assert torch.isfinite(pos_scores).all()
    assert torch.isfinite(neg_scores).all()
