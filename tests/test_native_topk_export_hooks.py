import json
from types import SimpleNamespace

import torch
import torch.nn as nn

from fast3_delta.eval import evaluate_usim
from hin_eval_common import evaluate_embedding_ranker


def _read_one(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def _item_bank():
    return torch.tensor(
        [
            [0.8, 0.6],
            [1.0, 0.0],
            [0.9, 0.4358899],
            [0.7, 0.7141428],
        ],
        dtype=torch.float32,
    )


def test_embedding_ranker_exports_after_seen_mask(tmp_path):
    output = tmp_path / "hin.jsonl"
    loader = [
        (
            {"u": torch.tensor([7]), "i": torch.tensor([0])},
            torch.tensor([0]),
        )
    ]

    metrics, count = evaluate_embedding_ranker(
        loader,
        device=torch.device("cpu"),
        n_items=4,
        cold_threshold=1,
        get_user_vectors_fn=lambda batch: torch.tensor([[1.0, 0.0]]),
        all_item_vectors=_item_bank(),
        k_list=(1, 3),
        eval_type="cold",
        full_ranking=True,
        user_seen_items={7: {1}},
        average_mode="item_macro",
        export_topk_path=str(output),
        export_topk_k=3,
        export_topk_metadata={"model": "cgrc", "seed": 2025},
    )

    assert count == 1
    assert metrics["R@3"] == 1.0
    row = _read_one(output)
    assert row["recommended_item_ids"] == [2, 0, 3]
    assert 1 not in row["recommended_item_ids"]
    assert row["model"] == "cgrc"


class _DummyUsim(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(
            n_items=4,
            cold_threshold=1,
            emb_dim=2,
            use_course_rerank=False,
            legacy_train_protocol=True,
        )
        self.user_emb = nn.Embedding(8, 2)
        with torch.no_grad():
            self.user_emb.weight.zero_()
            self.user_emb.weight[7] = torch.tensor([1.0, 0.0])
        self.user_proj = nn.Identity()
        self.user_seen_index = None
        self.bank = _item_bank()

    def get_item_vector(self, item_idx, llm_s, force_cold=False):
        vec = self.bank[item_idx]
        return vec, vec, vec

    def apply_course_rerank(self, scores, user_ids, seen_tensor_cache, cand_idx, target_pop):
        scores = scores.clone()
        scores[:, 3] += 2.0
        return scores


def test_usim_exports_after_seen_mask_and_model_score_adjustment(tmp_path):
    output = tmp_path / "usim.jsonl"
    model = _DummyUsim()
    loader = [
        (
            {"u": torch.tensor([7]), "i": torch.tensor([0])},
            torch.tensor([0]),
            torch.tensor([0.0]),
        )
    ]

    metrics, count = evaluate_usim(
        model,
        loader,
        torch.device("cpu"),
        llm_scores={},
        k_list=(1, 3),
        eval_type="cold",
        full_ranking=True,
        user_seen_items={7: {1}},
        all_item_vecs=_item_bank(),
        average_mode="item_macro",
        export_topk_path=str(output),
        export_topk_k=3,
        export_topk_metadata={"model": "ckg_rl", "seed": 2025},
    )

    assert count == 1
    assert metrics["R@3"] == 1.0
    row = _read_one(output)
    assert row["recommended_item_ids"] == [3, 2, 0]
    assert 1 not in row["recommended_item_ids"]
    assert row["model"] == "ckg_rl"
