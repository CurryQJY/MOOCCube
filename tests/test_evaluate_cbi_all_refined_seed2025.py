from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_cbi_all_refined_seed2025 import (
    build_all_refined_item_bank,
    cached_bank_positive_vectors,
)


class _FakeModel:
    def __init__(self):
        self.cfg = SimpleNamespace(n_items=4, emb_dim=2, cold_threshold=1)
        self.item_popularity = torch.tensor([5.0, 0.0, 3.0, 0.0])
        self.calls = []

    def infer_refined_item_vectors(
        self,
        item_idx,
        llm_s=None,
        item_batch=1024,
        force_cold=True,
        user_bank_raw=None,
    ):
        del llm_s, item_batch, user_bank_raw
        ids = item_idx.detach().cpu().tolist()
        self.calls.append((ids, force_cold))
        if force_cold:
            rows = [torch.tensor([0.0, float(idx + 1)]) for idx in ids]
        else:
            rows = [torch.tensor([float(idx + 1), 0.0]) for idx in ids]
        return torch.stack(rows)


def test_all_refined_bank_simulates_both_cold_and_hot_items():
    model = _FakeModel()

    bank, stats = build_all_refined_item_bank(
        model,
        torch.device("cpu"),
        llm_scores=None,
        item_batch=2,
    )

    assert model.calls == [([1, 3], True), ([0, 2], False)]
    assert torch.equal(bank[0], torch.tensor([1.0, 0.0]))
    assert torch.equal(bank[1], torch.tensor([0.0, 1.0]))
    assert torch.equal(bank[2], torch.tensor([1.0, 0.0]))
    assert torch.equal(bank[3], torch.tensor([0.0, 1.0]))
    assert stats == {"cold_items": 2, "hot_items": 2, "total_items": 4}


def test_cached_positive_vectors_reuse_same_refined_bank_for_hot_and_cold():
    bank = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    selected = cached_bank_positive_vectors(bank, torch.tensor([2, 1]))

    assert torch.equal(selected, bank[[2, 1]])
