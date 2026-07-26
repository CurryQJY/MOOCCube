from pathlib import Path
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usim_official_static_hin import (
    build_official_rl_records,
    cold_item_ids_from_splits,
    Config,
    load_split_cold_threshold,
    train_official_rl,
)

import usim_official_static_hin as usim_mod


def test_load_split_cold_threshold_prefers_static_summary(tmp_path):
    (tmp_path / "static_split_summary.json").write_text(
        json.dumps({"cold_threshold": 1}),
        encoding="utf-8",
    )

    assert load_split_cold_threshold(str(tmp_path), fallback=5) == 1


def test_cold_item_ids_from_splits_uses_threshold_and_unique_items():
    train_df = pd.DataFrame({"u_idx": [0, 1], "i_idx": [10, 11], "popularity": [3, 2]})
    val_df = pd.DataFrame({"u_idx": [2, 3], "i_idx": [12, 13], "popularity": [0, 2]})
    test_df = pd.DataFrame({"u_idx": [4, 5, 6], "i_idx": [12, 14, 14], "popularity": [0, 0, 0]})

    cold = cold_item_ids_from_splits(train_df, val_df, test_df, cold_threshold=1)

    assert cold.tolist() == [12, 14]


def test_build_official_rl_records_groups_users_and_excludes_cold_items():
    train_df = pd.DataFrame(
        {
            "u_idx": [0, 1, 2, 3],
            "i_idx": [10, 10, 11, 12],
            "popularity": [2, 2, 2, 0],
        }
    )
    content = torch.arange(13 * 3, dtype=torch.float32).view(13, 3)

    records = build_official_rl_records(train_df, content, excluded_item_ids=torch.tensor([12]))

    assert [r["item"] for r in records] == [10, 11]
    assert records[0]["user"] == [0, 1]
    assert records[1]["user"] == [2]
    assert torch.equal(records[0]["item_content"], content[10])


def test_main_table_aggregator_knows_official_usim_result():
    import aggregate_main_table_static_results as agg

    assert "usim_official_static_result.json" in agg.RESULT_FILES
    assert "USIM" in agg.MODEL_ORDER


def test_default_rl_batch_size_is_safe_for_large_user_action_space():
    old_value = os.environ.pop("USIM_OFFICIAL_RL_BATCH_SIZE", None)
    try:
        cfg = Config(n_users=199199, n_items=698, content_dim=768, cold_threshold=1)
        assert cfg.rl_batch_size <= 8
    finally:
        if old_value is not None:
            os.environ["USIM_OFFICIAL_RL_BATCH_SIZE"] = old_value


class _FakeOfficialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.actor_optimizer = torch.optim.Adam([self.weight], lr=0.01)
        self.critic_optimizer = torch.optim.Adam([self.weight], lr=0.01)
        self.buffer = []

    def update_buffer(self, batch, epoch):
        self.buffer.append((batch, epoch))

    def optimize(self, device):
        del device
        loss = (self.weight + 1.0).pow(2).sum()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss.detach(), loss.detach()

    def buffer_clear(self):
        self.buffer.clear()


def test_train_official_rl_writes_latest_and_best_checkpoints(tmp_path, monkeypatch=None):
    original_eval = usim_mod.evaluate_official_model
    scores = iter([0.1, 0.2])

    def fake_eval(*args, **kwargs):
        del args, kwargs
        return {"N@10": next(scores)}, 1

    usim_mod.evaluate_official_model = fake_eval
    try:
        cfg = Config(n_users=3, n_items=4, content_dim=2, cold_threshold=1)
        cfg.rl_epochs = 2
        cfg.rl_batch_size = 1
        cfg.ckpt_dir = str(tmp_path)
        cfg.save_ckpt = True
        cfg.auto_resume = True
        cfg.force_fresh = False
        cfg.save_opt_state = True

        model = _FakeOfficialModel()
        records = [{"item": 1, "user": [0, 1], "item_content": torch.ones(2)}]

        best_epoch, best_val = train_official_rl(
            model,
            records,
            cfg,
            torch.device("cpu"),
            val_loader=[],
            content_emb=torch.ones(4, 2),
            warm_item_ids=torch.tensor([0, 2, 3]),
            cold_item_ids=torch.tensor([1]),
            val_seen={},
        )

        latest = torch.load(tmp_path / "rl_latest.pt", map_location="cpu", weights_only=False)
        best = torch.load(tmp_path / "rl_best.pt", map_location="cpu", weights_only=False)

        assert best_epoch == 2
        assert best_val == 0.2
        assert latest["epoch"] == 2
        assert latest["best_epoch"] == 2
        assert latest["best_val"] == 0.2
        assert latest["actor_optimizer_state"] is not None
        assert latest["critic_optimizer_state"] is not None
        assert best["best_epoch"] == 2
        assert "model_state" in best
    finally:
        usim_mod.evaluate_official_model = original_eval
