from pathlib import Path
import sys
from types import MethodType
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbi_trust_sim import (
    CBITrustFast3FeedbackUSIM,
    project_to_content_cone,
    trust_build_eval_item_vecs,
    trust_build_eval_pos_item_vecs,
)
from fast3_delta.config import Fast3Config


def test_projection_keeps_in_domain_vector():
    anchor = torch.tensor([[1.0, 0.0]])
    state = torch.tensor([[0.9, 0.4358899]])

    projected, stats = project_to_content_cone(state, anchor, cosine_floor=0.8)

    assert torch.allclose(projected, F.normalize(state, dim=1), atol=1e-6)
    assert stats["projected_count"] == 0


def test_projection_hits_cosine_boundary():
    anchor = torch.tensor([[1.0, 0.0]])
    state = torch.tensor([[0.0, 1.0]])
    cosine_floor = 0.8660254037844386

    projected, stats = project_to_content_cone(state, anchor, cosine_floor=cosine_floor)

    assert torch.allclose(projected.norm(dim=1), torch.ones(1), atol=1e-6)
    assert torch.all((projected * anchor).sum(dim=1) >= cosine_floor - 1e-6)
    assert stats["projected_count"] == 1


def test_projection_handles_antiparallel_input_without_nan():
    anchor = torch.tensor([[1.0, 0.0]])

    projected, stats = project_to_content_cone(-anchor, anchor, cosine_floor=0.8660254037844386)

    assert torch.isfinite(projected).all()
    assert torch.equal(projected, anchor)
    assert stats["projected_count"] == 1


def _build_tiny_trust_model(monkeypatch):
    monkeypatch.setenv("USIM_USE_CONTENT_DELTA", "1")
    monkeypatch.setenv("USIM_CONTENT_DELTA_PAPER_STYLE", "1")
    monkeypatch.setenv("USIM_CONTENT_DELTA_REPLACE_ITEM", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = Fast3Config(n_users=4, n_items=4, content_dim=5)
    cfg.usim_steps = 2
    cfg.cbi_trust_cosine_floor = 0.8660254037844386
    model = CBITrustFast3FeedbackUSIM(cfg, torch.randn((4, 5), generator=torch.Generator().manual_seed(7)))
    model.device = torch.device("cpu")
    model.eval()

    def fake_get_candidates(self, current_h, **kwargs):
        del kwargs
        candidate = torch.roll(F.normalize(current_h.detach(), dim=1), shifts=1, dims=1)
        return candidate.unsqueeze(1), torch.zeros((current_h.size(0), 1), dtype=torch.long), {
            "dup_rate": 0.0,
            "topm_coverage": 1.0,
        }

    def fake_sampling(self, current_h, candidates, cand_user_idx, **kwargs):
        del current_h, kwargs
        return candidates, cand_user_idx, None

    def fake_action(self, current_h, time_step, candidates, **kwargs):
        del time_step, kwargs
        batch = current_h.size(0)
        return (
            torch.zeros(batch, dtype=torch.long),
            torch.zeros(batch),
            torch.zeros(batch),
            torch.zeros(batch),
        )

    def fake_course_terms(self, selected_user_ids, **kwargs):
        del selected_user_ids, kwargs
        zeros = torch.zeros((1, 1))
        return {
            "concept_bonus": zeros,
            "prereq_gap": zeros,
            "difficulty_gap": zeros,
            "redundant": zeros,
        }

    model.get_candidates = MethodType(fake_get_candidates, model)
    model._apply_course_sampling_bias = MethodType(fake_sampling, model)
    model._select_rollout_action = MethodType(fake_action, model)
    model._compute_course_reward_terms = MethodType(fake_course_terms, model)
    return model


def test_constrained_simulator_ignores_supplied_id_target_and_respects_floor(monkeypatch):
    model = _build_tiny_trust_model(monkeypatch)
    item_idx = torch.tensor([1])
    initial = F.normalize(model._content_base_embedding(item_idx), dim=1)
    target_a = torch.ones_like(initial)
    target_b = -torch.ones_like(initial)

    output_a, _, stats_a = model.run_usim_episode(
        initial,
        target_emb=target_a,
        item_idx=item_idx,
        target_pop=torch.zeros(1),
        deterministic=True,
    )
    output_b, _, stats_b = model.run_usim_episode(
        initial,
        target_emb=target_b,
        item_idx=item_idx,
        target_pop=torch.zeros(1),
        deterministic=True,
    )

    assert torch.allclose(output_a, output_b)
    assert stats_a["trust_min_cosine"] >= model.cfg.cbi_trust_cosine_floor - 1e-6
    assert stats_b["trust_min_cosine"] >= model.cfg.cbi_trust_cosine_floor - 1e-6
    assert stats_a["trust_steps"] == model.cfg.usim_steps


class _FakeAllRefinedModel:
    def __init__(self):
        self.cfg = SimpleNamespace(
            n_items=4,
            emb_dim=2,
            cold_threshold=1,
            candidate_strategy="none",
        )
        self.item_popularity = torch.tensor([4.0, 0.0, 2.0, 0.0])
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
            return torch.tensor([[0.0, float(idx + 1)] for idx in ids])
        return torch.tensor([[float(idx + 1), 0.0] for idx in ids])


def test_trust_eval_refines_cold_and_hot_and_reuses_cached_bank():
    model = _FakeAllRefinedModel()

    banks = trust_build_eval_item_vecs(model, torch.device("cpu"), llm_scores=None, item_batch=2)
    positives = trust_build_eval_pos_item_vecs(
        model,
        torch.tensor([0, 3]),
        llm_s=torch.full((2,), -1.0),
        pop_sel=torch.tensor([4.0, 0.0]),
        eval_type="all",
    )

    assert model.calls == [([1, 3], True), ([0, 2], False)]
    assert torch.equal(banks["cold"], banks["hot"])
    assert torch.equal(banks["hot"], banks["all"])
    assert torch.equal(positives, banks["all"][[0, 3]])
