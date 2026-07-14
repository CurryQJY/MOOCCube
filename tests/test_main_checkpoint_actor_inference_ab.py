import torch
import torch.nn.functional as F

import main_checkpoint_actor_inference_ab as ab
import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy


def test_deterministic_action_uses_actor_argmax():
    torch.manual_seed(7)
    agent = legacy.FixedSimpleAC(item_dim=4, time_dim=5)
    state = torch.randn(2, 4)
    time_step = torch.zeros(2, 1)
    candidates = torch.randn(2, 3, 4)

    action, log_prob, value, entropy = ab.deterministic_get_action_value(
        agent,
        state,
        time_step,
        candidates,
    )

    with torch.no_grad():
        t_emb = F.one_hot(time_step.squeeze(1).long(), num_classes=5).float()
        feat = agent.common(torch.cat([state, t_emb], dim=1))
        query = agent.actor_head(feat).unsqueeze(1)
        keys = agent.user_proj(candidates)
        expected = torch.matmul(query, keys.transpose(1, 2)).squeeze(1).argmax(dim=-1)

    assert torch.equal(action, expected)
    assert log_prob.shape == (2,)
    assert value.shape == (2, 1)
    assert entropy.shape == (2,)


class TinyInferenceModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.training = False
        self.cfg = type("Cfg", (), {"emb_dim": 2, "candidate_strategy": "retrieve_sample"})()
        self.item_id_emb = type("Emb", (), {"weight": torch.zeros(3, 2)})()
        self.item_popularity = torch.tensor([3.0, 0.0, 0.0])
        self.user_seen_index = torch.zeros((2, 3), dtype=torch.bool)
        self.calls = []

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def _build_user_bank_raw(self):
        bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return bank, F.normalize(bank, dim=1)

    def get_item_vector(self, idx, llm_s, force_cold, disable_id_dropout=True):
        base = torch.stack([idx.float() + 1.0, torch.ones_like(idx).float()], dim=1)
        return base, base, base

    def run_usim_episode(self, init, target_emb, **kwargs):
        self.calls.append({"target_emb": target_emb, **kwargs})
        return init + 0.25, {}, {}

    def _blend_rl_episode_output(self, base, final):
        return final


def test_refinement_never_supplies_behavior_target():
    model = TinyInferenceModel()

    out = ab.infer_actor_refined_item_vectors(model, torch.tensor([1, 2]))

    assert out.shape == (2, 2)
    assert len(model.calls) == 1
    assert model.calls[0]["target_emb"] is None
    assert torch.equal(model.calls[0]["item_idx"], torch.tensor([1, 2]))
    assert model.calls[0]["user_seen_items"] == {}


def test_install_static_does_not_attach_refinement(monkeypatch):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)

    ab.install_mode("static", eval_seed=7001)

    assert not hasattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors")


def test_install_actor_attaches_refinement(monkeypatch):
    monkeypatch.delattr(legacy.Fast3FeedbackUSIM, "infer_refined_item_vectors", raising=False)

    ab.install_mode("actor", eval_seed=7001)

    assert legacy.Fast3FeedbackUSIM.infer_refined_item_vectors is ab.infer_actor_refined_item_vectors


def test_positive_target_vector_comes_from_the_same_refined_bank():
    previous = getattr(ab, "ACTIVE_ITEM_BANK", None)
    ab.ACTIVE_ITEM_BANK = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    try:
        out = ab.bank_aligned_pos_item_vecs(
            model=None,
            item_idx=torch.tensor([2, 0]),
            llm_s=None,
            pop_sel=None,
            eval_type="cold",
        )
    finally:
        ab.ACTIVE_ITEM_BANK = previous

    assert torch.equal(out, torch.tensor([[3.0, 4.0], [1.0, 0.0]]))
