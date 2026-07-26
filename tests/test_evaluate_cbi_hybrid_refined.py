from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_cbi_hybrid_refined import build_cold_refined_hot_base_bank


class _FakeModel:
    def __init__(self):
        self.cfg = SimpleNamespace(
            n_items=4,
            emb_dim=2,
            cold_threshold=1,
            candidate_strategy="retrieve_sample",
        )
        self.item_popularity = torch.tensor([5.0, 0.0, 3.0, 0.0])
        self.refine_calls = []

    def _build_user_bank_raw(self):
        return torch.ones(2, 2)

    def get_item_vector(self, item_idx, llm_s, force_cold=False, disable_id_dropout=True):
        del llm_s, force_cold, disable_id_dropout
        rows = [torch.tensor([float(idx + 1), 1.0]) for idx in item_idx.tolist()]
        base = torch.stack(rows)
        return base, base, base

    def infer_refined_item_vectors(
        self,
        item_idx,
        llm_s=None,
        item_batch=1024,
        force_cold=True,
        user_bank_raw=None,
    ):
        del llm_s, item_batch, user_bank_raw
        self.refine_calls.append((item_idx.tolist(), force_cold))
        return torch.stack([torch.tensor([0.0, float(idx + 1)]) for idx in item_idx.tolist()])


def test_hybrid_bank_refines_cold_and_keeps_hot_base_vectors():
    model = _FakeModel()

    bank, stats = build_cold_refined_hot_base_bank(
        model,
        torch.device("cpu"),
        llm_scores=None,
        item_batch=2,
    )

    assert model.refine_calls == [([1, 3], True)]
    assert torch.allclose(bank[0], torch.nn.functional.normalize(torch.tensor([1.0, 1.0]), dim=0))
    assert torch.equal(bank[1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(bank[2], torch.nn.functional.normalize(torch.tensor([3.0, 1.0]), dim=0))
    assert torch.equal(bank[3], torch.tensor([0.0, 1.0]))
    assert stats == {
        "cold_items": 2,
        "hot_items": 2,
        "total_items": 4,
        "cold_refined": True,
        "hot_refined": False,
    }
