from types import SimpleNamespace

import pytest
import torch

from fast3_delta.v1_protocol import (
    advance_running_retention_selector,
    apply_course_fit_sampling_bias,
    batch_invariant_alignment_score,
    exclude_row_targets_from_history,
    select_retained_cold_checkpoint,
    select_running_retained_cold_checkpoint,
    update_running_retention_n_peaks,
)
from fast3_delta.config import Fast3Config
from fast3_delta.eval import build_eval_item_vecs
from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM


def test_target_exclusion_removes_each_row_target_before_all_course_terms():
    history = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    target_items = torch.tensor([1, 3], dtype=torch.long)

    clean, counts = exclude_row_targets_from_history(history, target_items)

    assert torch.equal(
        clean,
        torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
    )
    assert torch.equal(counts, torch.tensor([[2.0], [2.0]]))
    assert torch.equal(history, torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]]))


def test_simulator_alignment_gradient_has_fixed_reference_scale():
    single_h = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    single_user = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    single_target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    single_alpha = torch.tensor([[0.25]], dtype=torch.float32)

    single_score = batch_invariant_alignment_score(
        single_h, single_user, single_target, single_alpha, reference_batch_size=2048
    )
    single_grad = torch.autograd.grad(single_score, single_h)[0]

    many_h = single_h.detach().repeat(7, 1).requires_grad_(True)
    many_score = batch_invariant_alignment_score(
        many_h,
        single_user.repeat(7, 1),
        single_target.repeat(7, 1),
        single_alpha.repeat(7, 1),
        reference_batch_size=2048,
    )
    many_grad = torch.autograd.grad(many_score, many_h)[0]

    assert torch.allclose(single_grad[0], many_grad[0])


def test_course_fit_changes_sampling_probabilities_before_multinomial():
    base_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    course_fit = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    unchanged = apply_course_fit_sampling_bias(base_probs, course_fit, beta=0.0)
    biased = apply_course_fit_sampling_bias(base_probs, course_fit, beta=0.8)

    assert torch.allclose(unchanged, base_probs)
    assert biased[0, 3] > base_probs[0, 3]
    assert torch.isclose(biased.sum(), torch.tensor(1.0))


def test_v1_selector_requires_hot_and_overall_retention_before_cold_ranking():
    base = {
        "hot": {"N@10": 0.20, "R@10": 0.30},
        "overall": {"N@10": 0.18, "R@10": 0.27},
    }
    rejected = {
        "cold": {"N@10": 0.90, "R@10": 0.90},
        "hot": {"N@10": 0.10, "R@10": 0.12},
        "overall": {"N@10": 0.12, "R@10": 0.14},
    }
    retained = {
        "cold": {"N@10": 0.40, "R@10": 0.35},
        "hot": {"N@10": 0.199, "R@10": 0.298},
        "overall": {"N@10": 0.178, "R@10": 0.268},
    }

    selected = select_retained_cold_checkpoint(
        previous=None,
        candidate=rejected,
        base=base,
        k=10,
        hot_tolerance=0.003,
        overall_tolerance=0.003,
    )
    assert selected is None

    selected = select_retained_cold_checkpoint(
        previous=None,
        candidate=retained,
        base=base,
        k=10,
        hot_tolerance=0.003,
        overall_tolerance=0.003,
    )
    assert selected is retained


def test_v1_selector_breaks_retained_cold_ndcg_ties_with_recall():
    base = {
        "hot": {"N@10": 0.20, "R@10": 0.30},
        "overall": {"N@10": 0.18, "R@10": 0.27},
    }
    previous = {
        "cold": {"N@10": 0.40, "R@10": 0.31},
        "hot": {"N@10": 0.20, "R@10": 0.30},
        "overall": {"N@10": 0.18, "R@10": 0.27},
    }
    candidate = {
        "cold": {"N@10": 0.40, "R@10": 0.35},
        "hot": {"N@10": 0.20, "R@10": 0.30},
        "overall": {"N@10": 0.18, "R@10": 0.27},
    }

    assert (
        select_retained_cold_checkpoint(
            previous=previous,
            candidate=candidate,
            base=base,
            k=10,
            hot_tolerance=0.003,
            overall_tolerance=0.003,
        )
        is candidate
    )


def test_v1_running_retention_rejects_late_cold_gain_after_hot_overall_drop():
    """Use the observed V1 epoch-7/9 validation pattern without test leakage."""
    epoch7 = {
        "cold": {"N@10": 0.188251, "R@10": 0.239999},
        "hot": {"N@10": 0.087122, "R@10": 0.143149},
        "overall": {"N@10": 0.092843, "R@10": 0.148628},
    }
    epoch9 = {
        "cold": {"N@10": 0.194021, "R@10": 0.248059},
        "hot": {"N@10": 0.079301, "R@10": 0.130661},
        "overall": {"N@10": 0.085791, "R@10": 0.137303},
    }
    # Epoch 6 owned the earlier Hot peak; epoch 7 then established the
    # earlier Overall peak.  The candidate must clear both NDCG floors.
    peaks_before_epoch7 = {"hot_n": 0.087419, "overall_n": 0.092241}
    peaks_after_epoch7 = update_running_retention_n_peaks(
        peaks_before_epoch7, epoch7, k=10
    )
    assert peaks_after_epoch7 == {"hot_n": 0.087419, "overall_n": 0.092843}

    selected = select_running_retained_cold_checkpoint(
        previous=epoch7,
        candidate=epoch9,
        base={"hot": epoch9["hot"], "overall": epoch9["overall"]},
        running_n_peaks=peaks_after_epoch7,
        k=10,
        hot_tolerance=0.003,
        overall_tolerance=0.003,
    )

    assert selected is epoch7


def test_v1_running_retention_advances_peaks_after_current_epoch_decision():
    candidate = {
        "cold": {"N@10": 0.188251, "R@10": 0.239999},
        "hot": {"N@10": 0.087122, "R@10": 0.143149},
        "overall": {"N@10": 0.092843, "R@10": 0.148628},
    }
    prior_peaks = {"hot_n": 0.087419, "overall_n": 0.092241}

    selected, next_peaks, base_retained, running_retained = (
        advance_running_retention_selector(
            previous=None,
            candidate=candidate,
            base={"hot": candidate["hot"], "overall": candidate["overall"]},
            running_n_peaks=prior_peaks,
            k=10,
            hot_tolerance=0.003,
            overall_tolerance=0.003,
        )
    )

    assert base_retained is True
    assert running_retained is True
    assert selected is candidate
    assert next_peaks == {"hot_n": 0.087419, "overall_n": 0.092843}


def _v1_model(monkeypatch):
    monkeypatch.setenv("USIM_CKG_RL_V1", "1")
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_AUX_WEIGHT", "0")
    monkeypatch.setenv("USIM_PPO_LOSS_WEIGHT", "0")
    monkeypatch.setenv("USIM_USE_PREREQ_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_SAGE_AUX_LOSS", "0")
    monkeypatch.setenv("USIM_USE_CGRC_RECON", "0")
    cfg = Fast3Config(n_users=2, n_items=4, content_dim=5)
    cfg.dropout_prob = 0.0
    cfg.use_mixed_hard_neg = False
    cfg.cold_threshold = 1
    cfg.feedback_course_sample_beta = 0.0
    model = Fast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))
    model.device = torch.device("cpu")
    return model


def test_v1_forward_simulates_only_fixed_pseudocold_rows_and_detaches_user_anchor(monkeypatch):
    model = _v1_model(monkeypatch)
    model.train()
    model.set_pseudo_cold_item_mask(torch.tensor([False, True, False, False]))
    captured = {}

    def fake_episode(init_item_emb, target_emb=None, **kwargs):
        captured["items"] = kwargs["item_idx"].detach().cpu().tolist()
        return init_item_emb, {"rewards": []}, {"steps": 0}

    model.run_usim_episode = fake_episode
    model.compute_ppo_loss = lambda trajectory: torch.zeros((), dtype=torch.float32)

    loss, _ = model(
        {"u": torch.tensor([0, 1]), "i": torch.tensor([0, 1])},
        torch.tensor([8.0, 8.0]),
        torch.full((2,), -1.0),
        user_seen_items={0: {0}, 1: {1}},
    )
    loss.backward()

    assert captured["items"] == [1]
    assert model.user_emb.weight.grad[1].abs().sum().item() == pytest.approx(0.0)
    assert model.user_emb.weight.grad[0].abs().sum().item() > 0.0


def test_v1_eval_bank_passes_train_only_history_to_refined_inference():
    class HistoryModel:
        def __init__(self):
            self.cfg = SimpleNamespace(
                n_items=4,
                emb_dim=2,
                cold_threshold=1,
                v1_enabled=True,
                legacy_train_protocol=False,
                use_usim_refined_eval=True,
                content_delta_cold_only=False,
                llm_bank_mode="none",
            )
            self.item_popularity = torch.tensor([0.0, 4.0, 5.0, 6.0])
            self.training = True
            self.inference_history = None

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def get_item_vector(self, item_idx, llm_s, force_cold=False):
            del llm_s, force_cold
            vec = torch.stack(
                [item_idx.float() + 1.0, torch.ones_like(item_idx, dtype=torch.float32)], dim=1
            )
            return vec, vec, vec

        def infer_refined_item_vectors(
            self, item_idx, llm_s=None, item_batch=1024, force_cold=True, user_seen_items=None
        ):
            del llm_s, item_batch, force_cold
            self.inference_history = user_seen_items
            return torch.tensor([[9.0, 1.0]], dtype=torch.float32).repeat(item_idx.numel(), 1)

    history = {0: {1, 2}, 1: {3}}
    model = HistoryModel()
    banks = build_eval_item_vecs(
        model,
        torch.device("cpu"),
        llm_scores=None,
        item_batch=2,
        user_seen_items=history,
    )

    assert model.inference_history == history
    assert banks["all"].shape == (4, 2)


def test_legacy_eval_bank_does_not_pass_explicit_history_to_refined_inference():
    class LegacyHistoryModel:
        def __init__(self):
            self.cfg = SimpleNamespace(
                n_items=4,
                emb_dim=2,
                cold_threshold=1,
                v1_enabled=False,
                legacy_train_protocol=False,
                use_usim_refined_eval=True,
                content_delta_cold_only=False,
                llm_bank_mode="none",
            )
            self.item_popularity = torch.tensor([0.0, 4.0, 5.0, 6.0])
            self.training = True
            self.inference_history = "not-called"

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def get_item_vector(self, item_idx, llm_s, force_cold=False):
            del llm_s, force_cold
            vec = torch.stack(
                [item_idx.float() + 1.0, torch.ones_like(item_idx, dtype=torch.float32)], dim=1
            )
            return vec, vec, vec

        def infer_refined_item_vectors(
            self, item_idx, llm_s=None, item_batch=1024, force_cold=True, user_seen_items=None
        ):
            del llm_s, item_batch, force_cold
            self.inference_history = user_seen_items
            return torch.tensor([[9.0, 1.0]], dtype=torch.float32).repeat(item_idx.numel(), 1)

    history = {0: {1, 2}, 1: {3}}
    model = LegacyHistoryModel()

    build_eval_item_vecs(
        model,
        torch.device("cpu"),
        llm_scores=None,
        item_batch=2,
        user_seen_items=history,
    )

    assert model.inference_history is None


def test_v1_refined_inference_is_invariant_to_item_bank_chunking(monkeypatch):
    monkeypatch.setenv("USIM_CKG_RL_V1", "1")
    monkeypatch.setenv("USIM_STEPS", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_PPO_LOSS_WEIGHT", "0")
    monkeypatch.setenv("USIM_ROLLOUT_POLICY", "course_fit")
    monkeypatch.setenv("USIM_DETERMINISTIC_EVAL_CANDIDATES", "1")
    monkeypatch.setenv("USIM_DETERMINISTIC_EVAL_SEED", "1907")
    monkeypatch.setenv("USIM_V1_REFERENCE_BATCH_SIZE", "16")
    monkeypatch.setenv("USIM_N_CANDIDATES", "2")
    monkeypatch.setenv("USIM_RETRIEVE_TOP_M", "4")
    model = Fast3FeedbackUSIM(
        Fast3Config(n_users=4, n_items=3, content_dim=5),
        torch.randn((3, 5), dtype=torch.float32),
    )
    model.device = torch.device("cpu")
    model.set_feedback_item_stats(torch.tensor([0, 0, 0]))

    item_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    whole = model.infer_refined_item_vectors(item_ids, item_batch=3, user_seen_items={})
    chunked = model.infer_refined_item_vectors(item_ids, item_batch=1, user_seen_items={})

    assert torch.allclose(whole, chunked, atol=1e-7, rtol=1e-7)


def test_v1_get_candidates_applies_course_fit_before_sampling(monkeypatch):
    model = Fast3FeedbackUSIM.__new__(Fast3FeedbackUSIM)
    model.cfg = SimpleNamespace(
        v1_enabled=True,
        n_candidates=2,
        candidate_strategy="retrieve_sample",
        n_users=4,
        retrieve_top_m=4,
        candidate_temp=1.0,
        candidate_epsilon=0.0,
        feedback_course_sample_beta=0.8,
        use_sage_lite=False,
        use_cgrc_recon=False,
    )
    model.device = torch.device("cpu")
    model._retrieve_topm_exact = lambda query, bank, top_m, user_bank_norm=None: (
        torch.zeros((query.size(0), top_m)),
        torch.arange(top_m).view(1, -1).expand(query.size(0), -1),
    )
    model._compute_candidate_course_fit = lambda *args, **kwargs: torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )
    captured = {}

    def capture_multinomial(probs, num_samples, replacement=False, generator=None):
        del replacement, generator
        captured["probs"] = probs.detach().clone()
        return torch.topk(probs, k=num_samples, dim=1).indices

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    model.get_candidates(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        user_bank_raw=torch.eye(4, 2),
        item_idx=torch.tensor([0]),
        target_pop=torch.tensor([0.0]),
        user_seen_items={0: {1}},
    )

    assert captured["probs"][0, 3] > captured["probs"][0, 0]
