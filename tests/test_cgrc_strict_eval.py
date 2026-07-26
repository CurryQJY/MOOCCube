from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgrc_paper_static_hin import _apply_train_popularity_for_eval, _cold_items_in
import cgrc_paper_static_hin as cgrc


def test_cgrc_eval_popularity_uses_train_counts_for_cold_items():
    train_df = pd.DataFrame(
        {
            "u_idx": [0, 1, 2],
            "i_idx": [10, 10, 30],
            "popularity": [99, 99, 99],
        }
    )
    val_df = pd.DataFrame({"u_idx": [3, 4], "i_idx": [10, 20], "popularity": [99, 99]})
    test_df = pd.DataFrame({"u_idx": [5, 6], "i_idx": [20, 30], "popularity": [99, 99]})

    _, val_aligned, test_aligned = _apply_train_popularity_for_eval(train_df, val_df, test_df)

    assert val_aligned["popularity"].tolist() == [2, 0]
    assert test_aligned["popularity"].tolist() == [0, 1]
    assert _cold_items_in(val_aligned, cold_threshold=1).tolist() == [20]
    assert _cold_items_in(test_aligned, cold_threshold=1).tolist() == [20]


def test_cgrc_train_progress_line_reports_epoch_batch_and_eta():
    line = cgrc._format_train_progress(
        epoch=1,
        n_epochs=15,
        batch_idx=128,
        n_batches=1763,
        avg_loss=4.25,
        elapsed_s=63.0,
    )

    assert line.startswith("  [CGRC-TRAIN-PROGRESS] Epoch 1/15")
    assert "128/1763" in line
    assert "(7%)" in line
    assert "avg_loss=4.2500" in line
    assert "elapsed=1m03s" in line
    assert "eta=" in line


def test_cgrc_chunked_edge_logits_match_full_broadcast():
    torch.manual_seed(7)
    model = cgrc.CGRCNet(
        n_users=4,
        n_items=5,
        content_dim=3,
        emb_dim=2,
        mlp_hidden=4,
        item_content=torch.randn(5, 3),
    )
    h_u_bar = torch.randn(4, 2)
    x_all = torch.randn(5, 2)
    cold_ids = torch.tensor([0, 2, 4], dtype=torch.long)

    full = model.edge_logits_broadcast(h_u_bar, x_all, cold_ids)
    chunked = cgrc._edge_logits_broadcast_chunked(
        model,
        h_u_bar,
        x_all,
        cold_ids,
        user_chunk=2,
    )

    assert torch.allclose(chunked, full)
