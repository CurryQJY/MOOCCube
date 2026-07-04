from pathlib import Path
import os
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast3_delta.config import FeedbackConfig
from usim_feedback_fast3_content_delta import (
    FastFeedbackUSIM,
    _sage_only_cold_or_tail_candidate_probs,
    _sage_tail_gate_from_pop,
    _sage_course_sampling_combined_score,
    _sage_two_expert_candidate_probs,
)


def test_sage_tail_gate_is_highest_for_zero_popularity():
    pop = torch.tensor([0.0, 9.0, 99.0])

    gate = _sage_tail_gate_from_pop(pop, max_pop=99.0, gate_min=0.1, gate_max=0.6)

    assert torch.allclose(gate[0], torch.tensor(0.6), atol=1e-6)
    assert gate[0] > gate[1] > gate[2]
    assert torch.allclose(gate[2], torch.tensor(0.1), atol=1e-6)


def test_sage_tail_gate_clamps_invalid_bounds():
    pop = torch.tensor([0.0, 1.0])

    gate = _sage_tail_gate_from_pop(pop, max_pop=1.0, gate_min=0.8, gate_max=0.2)

    assert torch.all(gate >= 0.2)
    assert torch.all(gate <= 0.8)


def test_sage_bucket_mlp_config_reads_traceable_switches():
    old_env = {key: os.environ.get(key) for key in [
        "USIM_SAGE_GATE_MODE",
        "USIM_SAGE_GATE_BUCKETS",
        "USIM_SAGE_GATE_HIDDEN",
        "USIM_SAGE_GATE_BUCKET_STRATEGY",
    ]}
    try:
        os.environ["USIM_SAGE_GATE_MODE"] = "bucket_mlp"
        os.environ["USIM_SAGE_GATE_BUCKETS"] = "7"
        os.environ["USIM_SAGE_GATE_HIDDEN"] = "11"
        os.environ["USIM_SAGE_GATE_BUCKET_STRATEGY"] = "log"

        cfg = FeedbackConfig(3, 4, content_dim=5)

        assert cfg.sage_gate_mode == "bucket_mlp"
        assert cfg.sage_gate_bucket_count == 7
        assert cfg.sage_gate_hidden_dim == 11
        assert cfg.sage_gate_bucket_strategy == "log"
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_sage_bucket_mlp_uses_paper_equal_width_buckets_by_default():
    cfg = FeedbackConfig(4, 5, content_dim=5)
    cfg.use_sage_lite = True
    cfg.sage_gate_mode = "bucket_mlp"
    cfg.sage_gate_bucket_count = 5
    cfg.sage_gate_hidden_dim = 6
    content_emb = torch.randn(5, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.set_feedback_item_stats(torch.tensor([0.0, 24.0, 49.0, 74.0, 99.0]))

    bucket_ids = model._sage_popularity_bucket_ids(torch.tensor([0.0, 24.0, 25.0, 49.0, 50.0, 74.0, 75.0, 99.0]))

    assert getattr(cfg, "sage_gate_bucket_strategy", "paper") == "paper"
    assert torch.equal(bucket_ids.cpu(), torch.tensor([0, 1, 1, 2, 2, 3, 3, 4]))


def test_sage_bucket_mlp_gate_uses_train_popularity_buckets():
    cfg = FeedbackConfig(4, 4, content_dim=5)
    cfg.use_sage_lite = True
    cfg.sage_gate_mode = "bucket_mlp"
    cfg.sage_gate_bucket_count = 5
    cfg.sage_gate_hidden_dim = 6
    cfg.sage_gate_min = 0.1
    cfg.sage_gate_max = 0.6
    cfg.sage_only_cold_or_tail = False
    content_emb = torch.randn(4, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.set_feedback_item_stats(torch.tensor([0.0, 5.0, 20.0, 100.0]))

    bucket_ids = model._sage_popularity_bucket_ids(torch.tensor([0.0, 5.0, 100.0]))
    gate = model._sage_tail_gate(torch.tensor([0.0, 5.0, 100.0]), batch_size=3, n_cols=2)
    gate.sum().backward()

    grads = [
        param.grad
        for param in list(model.sage_gate_bucket_emb.parameters()) + list(model.sage_gate_mlp.parameters())
        if param.requires_grad
    ]

    assert bucket_ids[0].item() == 0
    assert bucket_ids[-1].item() == cfg.sage_gate_bucket_count - 1
    assert gate.shape == (3, 2)
    assert torch.all(gate >= cfg.sage_gate_min)
    assert torch.all(gate <= cfg.sage_gate_max)
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_sage_bucket_mlp_gate_respects_cold_or_tail_mask():
    cfg = FeedbackConfig(4, 4, content_dim=5)
    cfg.use_sage_lite = True
    cfg.sage_gate_mode = "bucket_mlp"
    cfg.sage_gate_bucket_count = 4
    cfg.sage_gate_hidden_dim = 5
    cfg.cold_threshold = 1
    cfg.sage_only_cold_or_tail = True
    cfg.sage_tail_pop_ratio = 0.01
    content_emb = torch.randn(4, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.set_feedback_item_stats(torch.tensor([0.0, 10.0, 50.0, 100.0]))

    gate = model._sage_tail_gate(torch.tensor([0.0, 50.0, 100.0]), batch_size=3, n_cols=1).view(-1)

    assert gate[0] > 0.0
    assert torch.allclose(gate[1:], torch.zeros(2), atol=1e-6)


def test_sage_two_expert_score_fusion_changes_fullrank_scores():
    cfg = FeedbackConfig(3, 4, content_dim=5)
    cfg.use_sage_lite = True
    cfg.sage_gate_mode = "bucket_mlp"
    cfg.sage_gate_bucket_count = 4
    cfg.sage_gate_hidden_dim = 5
    cfg.sage_gate_bucket_strategy = "paper"
    cfg.sage_two_expert_score_fusion = True
    cfg.sage_only_cold_or_tail = False
    cfg.sage_course_temp = 0.20
    content_emb = torch.randn(4, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.set_feedback_item_stats(torch.tensor([0.0, 10.0, 50.0, 100.0]))
    model.sage_gate_mlp[-1].bias.data.fill_(0.0)

    retrieval_scores = torch.tensor([[0.2, 0.4], [0.7, 0.1]], dtype=torch.float32)
    course_fit = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
    target_pop = torch.tensor([0.0, 100.0], dtype=torch.float32)

    fused = model.apply_sage_two_expert_score_fusion(
        retrieval_scores,
        course_fit,
        target_pop=target_pop,
    )

    assert fused.shape == retrieval_scores.shape
    assert not torch.allclose(fused, retrieval_scores)
    assert fused[0, 0] > fused[0, 1]
    assert fused[1, 1] > fused[1, 0]


def test_sage_only_cold_or_tail_masks_hot_gate():
    cfg = FeedbackConfig(3, 4, content_dim=5)
    cfg.cold_threshold = 1
    cfg.use_sage_lite = True
    cfg.sage_gate_min = 0.1
    cfg.sage_gate_max = 0.6
    cfg.sage_only_cold_or_tail = True
    cfg.sage_tail_pop_ratio = 0.10
    content_emb = torch.randn(3, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.set_feedback_item_stats(torch.tensor([0.0, 5.0, 20.0, 100.0]))

    target_pop = torch.tensor([0.0, 5.0, 100.0])
    gate = model._sage_tail_gate(target_pop, batch_size=3, n_cols=1).view(-1)

    assert gate[0] > 0.0
    assert gate[1] > 0.0
    assert torch.allclose(gate[2], torch.tensor(0.0), atol=1e-6)


def test_sage_only_candidate_probs_keep_full_hot_pool():
    top_scores = torch.tensor([
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
    ])
    course_fit_topk = torch.tensor([
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    sage_gate_topk = torch.tensor([
        [1.0, 1.0],
        [0.0, 0.0],
    ])

    probs = _sage_only_cold_or_tail_candidate_probs(
        top_scores,
        course_fit_topk,
        sage_gate_topk,
        candidate_temp=1.0,
        candidate_epsilon=0.0,
        course_temp=1.0,
    )

    expected_hot = torch.softmax(top_scores[1], dim=0)
    expected_active = torch.softmax(course_fit_topk[0], dim=0)
    assert torch.allclose(probs[1], expected_hot, atol=1e-6)
    assert torch.allclose(probs[0, :2], expected_active, atol=1e-6)
    assert torch.allclose(probs[0, 2:], torch.zeros(3), atol=1e-6)


def test_sage_two_expert_candidate_probs_use_uniform_and_course_experts():
    top_scores = torch.tensor([
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
    ])
    course_fit_topk = torch.tensor([
        [0.0, 2.0],
        [0.0, 2.0],
    ])
    sage_gate_topk = torch.tensor([
        [0.5, 0.5],
        [0.0, 0.0],
    ])

    probs = _sage_two_expert_candidate_probs(
        top_scores,
        course_fit_topk,
        sage_gate_topk,
        candidate_temp=1.0,
        candidate_epsilon=0.0,
        course_temp=1.0,
    )

    uniform_expert = torch.full((2,), 0.5)
    course_expert = torch.softmax(course_fit_topk[0], dim=0)
    expected_active = 0.5 * uniform_expert + 0.5 * course_expert
    expected_hot = torch.softmax(top_scores[1], dim=0)
    assert torch.allclose(probs[0, :2], expected_active, atol=1e-6)
    assert torch.allclose(probs[0, 2:], torch.zeros(3), atol=1e-6)
    assert torch.allclose(probs[1], expected_hot, atol=1e-6)


def test_sage_only_course_sampling_score_keeps_beta_for_hot_rows():
    retrieval_score = torch.tensor([
        [0.2, 0.8],
        [0.2, 0.8],
    ])
    fit_norm = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    sage_gate = torch.tensor([
        [0.5, 0.5],
        [0.0, 0.0],
    ])

    combined = _sage_course_sampling_combined_score(
        retrieval_score,
        fit_norm,
        sage_gate=sage_gate,
        beta=0.2,
        use_sage_lite=True,
        sage_only_cold_or_tail=True,
    )

    expected_active = 0.5 * retrieval_score[0] + 0.5 * fit_norm[0]
    expected_hot = retrieval_score[1] + 0.2 * fit_norm[1]
    assert torch.allclose(combined[0], expected_active, atol=1e-6)
    assert torch.allclose(combined[1], expected_hot, atol=1e-6)


def test_sage_aux_loss_is_separate_from_candidate_intervention():
    cfg = FeedbackConfig(4, 4, content_dim=5)
    cfg.use_sage_lite = False
    cfg.use_sage_aux_loss = True
    cfg.sage_aux_weight = 0.02
    cfg.sage_aux_pool_topk = 4
    cfg.sage_aux_only_strict_cold = True
    cfg.sage_aux_detach_user = True
    cfg.n_candidates = 2
    cfg.retrieve_top_m = 4
    cfg.batch_size = 2
    content_emb = torch.randn(4, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.device = torch.device("cpu")
    model.train()
    model.set_feedback_item_stats(torch.tensor([0.0, 2.0, 5.0, 20.0]))
    model.item_concept_overlap = torch.eye(4)
    user_seen_items = {0: {1}, 1: {2}, 2: {3}, 3: {1, 2}}
    model.set_user_seen_index(user_seen_items)

    item_idx = torch.tensor([0, 1], dtype=torch.long)
    item_emb = torch.randn(2, cfg.emb_dim)
    user_bank_raw, user_bank_norm = model._build_user_bank_raw()

    aux_loss, aux_info = model._compute_sage_aux_loss(
        item_emb,
        item_idx=item_idx,
        target_pop=torch.tensor([0.0, 2.0]),
        user_seen_items=user_seen_items,
        user_bank_raw=user_bank_raw,
        user_bank_norm=user_bank_norm,
    )
    _, _, cand_stats = model.get_candidates(
        item_emb,
        user_bank_raw=user_bank_raw,
        user_bank_norm=user_bank_norm,
        item_idx=item_idx,
        target_pop=torch.tensor([0.0, 2.0]),
        user_seen_items=user_seen_items,
    )

    assert aux_loss.item() >= 0.0
    assert aux_info["sage_aux_active_ratio"] > 0.0
    assert cand_stats["sage_active"] == 0.0
    assert cand_stats["sage_gate"] == 0.0


def test_sage_lite_candidate_sampling_path_returns_trace_stats():
    old_env = {key: os.environ.get(key) for key in [
        "USIM_FORCE_CPU",
        "USIM_USE_SAGE_LITE",
        "USIM_N_CANDIDATES",
        "USIM_RETRIEVE_TOP_M",
    ]}
    try:
        os.environ["USIM_FORCE_CPU"] = "1"
        os.environ["USIM_USE_SAGE_LITE"] = "1"
        os.environ["USIM_N_CANDIDATES"] = "2"
        os.environ["USIM_RETRIEVE_TOP_M"] = "4"
        cfg = FeedbackConfig(4, 3, content_dim=5)
        cfg.use_sage_lite = True
        cfg.n_candidates = 2
        cfg.retrieve_top_m = 4
        cfg.sage_pool_topk = 4
        content_emb = torch.randn(3, 5)
        model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
        model.set_feedback_item_stats(torch.tensor([0.0, 5.0, 20.0]))
        model.item_concept_overlap = torch.eye(3)
        user_seen_items = {0: {0}, 1: {1}, 2: {2}, 3: {1, 2}}
        model.set_user_seen_index(user_seen_items)

        item_emb = torch.randn(2, cfg.emb_dim)
        user_bank_raw, user_bank_norm = model._build_user_bank_raw()
        item_idx = torch.tensor([0, 1], dtype=torch.long)
        target_pop = model.item_popularity[item_idx]

        cand_emb, cand_idx, stats = model.get_candidates(
            item_emb,
            user_bank_raw=user_bank_raw,
            user_bank_norm=user_bank_norm,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
        )

        assert cand_emb.shape == (2, 2, cfg.emb_dim)
        assert cand_idx.shape == (2, 2)
        assert stats["sage_active"] == 1.0
        assert stats["sage_gate"] > 0.0
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_sage_only_inactive_batch_skips_course_fit():
    cfg = FeedbackConfig(4, 3, content_dim=5)
    cfg.use_sage_lite = True
    cfg.sage_only_cold_or_tail = True
    cfg.cold_threshold = 1
    cfg.sage_tail_pop_ratio = 0.002
    cfg.n_candidates = 2
    cfg.retrieve_top_m = 4
    cfg.sage_pool_topk = 4
    content_emb = torch.randn(3, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.device = torch.device("cpu")
    model.set_feedback_item_stats(torch.tensor([0.0, 50.0, 100.0]))
    user_seen_items = {0: {0}, 1: {1}, 2: {2}, 3: {1, 2}}
    model.set_user_seen_index(user_seen_items)

    def fail_course_fit(*args, **kwargs):
        raise AssertionError("inactive hot-only SAGE batch should not compute course fit")

    model._compute_candidate_course_fit = fail_course_fit

    item_emb = torch.randn(2, cfg.emb_dim)
    user_bank_raw, user_bank_norm = model._build_user_bank_raw()
    item_idx = torch.tensor([1, 2], dtype=torch.long)
    target_pop = torch.tensor([50.0, 100.0])

    cand_emb, cand_idx, stats = model.get_candidates(
        item_emb,
        user_bank_raw=user_bank_raw,
        user_bank_norm=user_bank_norm,
        item_idx=item_idx,
        target_pop=target_pop,
        user_seen_items=user_seen_items,
    )

    assert cand_emb.shape == (2, 2, cfg.emb_dim)
    assert cand_idx.shape == (2, 2)
    assert stats["sage_active"] == 1.0
    assert stats["sage_tail_active"] == 0.0
    assert stats["sage_gate"] == 0.0
    assert stats["sage_pool_fit"] == 0.0


def test_sage_only_mixed_batch_computes_course_fit_for_active_rows_only():
    cfg = FeedbackConfig(4, 3, content_dim=5)
    cfg.use_sage_lite = True
    cfg.sage_only_cold_or_tail = True
    cfg.cold_threshold = 1
    cfg.sage_tail_pop_ratio = 0.002
    cfg.n_candidates = 2
    cfg.retrieve_top_m = 4
    cfg.sage_pool_topk = 4
    content_emb = torch.randn(3, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.device = torch.device("cpu")
    model.set_feedback_item_stats(torch.tensor([0.0, 50.0, 100.0]))
    user_seen_items = {0: {0}, 1: {1}, 2: {2}, 3: {1, 2}}
    model.set_user_seen_index(user_seen_items)

    fit_call_shapes = []

    def record_course_fit(candidate_user_idx, item_idx=None, target_pop=None, user_seen_items=None):
        fit_call_shapes.append(tuple(candidate_user_idx.shape))
        assert item_idx.tolist() == [0]
        assert target_pop.tolist() == [0.0]
        return torch.ones(candidate_user_idx.shape, dtype=torch.float32)

    model._compute_candidate_course_fit = record_course_fit

    item_emb = torch.randn(2, cfg.emb_dim)
    user_bank_raw, user_bank_norm = model._build_user_bank_raw()
    item_idx = torch.tensor([0, 2], dtype=torch.long)
    target_pop = torch.tensor([0.0, 100.0])

    _, _, stats = model.get_candidates(
        item_emb,
        user_bank_raw=user_bank_raw,
        user_bank_norm=user_bank_norm,
        item_idx=item_idx,
        target_pop=target_pop,
        user_seen_items=user_seen_items,
    )

    assert fit_call_shapes == [(1, 4)]
    assert stats["sage_active"] == 1.0
    assert 0.0 < stats["sage_tail_active"] < 1.0
    assert stats["sage_gate"] > 0.0
    assert stats["sage_pool_fit"] > 0.0


def test_pseudo_cold_episode_pop_activates_sage_gate_for_hot_tail_proxy():
    cfg = FeedbackConfig(4, 4, content_dim=5)
    cfg.use_sage_lite = True
    cfg.sage_only_cold_or_tail = True
    cfg.cold_threshold = 1
    cfg.sage_tail_pop_ratio = 0.002
    cfg.use_pseudo_cold_train = True
    cfg.pseudo_cold_mode = "all_eligible"
    cfg.pseudo_cold_ratio = 1.0
    cfg.pseudo_cold_min_pop = 5
    content_emb = torch.randn(4, 5)
    model = FastFeedbackUSIM(cfg, content_emb).to(torch.device("cpu"))
    model.device = torch.device("cpu")
    model.train()
    model.set_feedback_item_stats(torch.tensor([0.0, 5.0, 20.0, 100.0]))

    pop = torch.tensor([20.0, 100.0])
    true_cold = model._cold_mask_from_pop(pop)
    effective_cold = model._effective_train_cold_mask(pop)
    episode_pop = model._target_pop_with_effective_cold(pop, effective_cold)
    gate = model._sage_tail_gate(episode_pop, batch_size=2, n_cols=1).view(-1)

    assert not bool(true_cold.any().item())
    assert torch.equal(effective_cold, torch.tensor([True, True]))
    assert torch.allclose(episode_pop, torch.zeros_like(pop))
    assert torch.all(gate > 0.0)


if __name__ == "__main__":
    test_sage_tail_gate_is_highest_for_zero_popularity()
    test_sage_tail_gate_clamps_invalid_bounds()
    test_sage_bucket_mlp_config_reads_traceable_switches()
    test_sage_bucket_mlp_uses_paper_equal_width_buckets_by_default()
    test_sage_bucket_mlp_gate_uses_train_popularity_buckets()
    test_sage_bucket_mlp_gate_respects_cold_or_tail_mask()
    test_sage_two_expert_score_fusion_changes_fullrank_scores()
    test_sage_only_cold_or_tail_masks_hot_gate()
    test_sage_only_candidate_probs_keep_full_hot_pool()
    test_sage_two_expert_candidate_probs_use_uniform_and_course_experts()
    test_sage_only_course_sampling_score_keeps_beta_for_hot_rows()
    test_sage_aux_loss_is_separate_from_candidate_intervention()
    test_sage_lite_candidate_sampling_path_returns_trace_stats()
    test_sage_only_inactive_batch_skips_course_fit()
    test_sage_only_mixed_batch_computes_course_fit_for_active_rows_only()
    test_pseudo_cold_episode_pop_activates_sage_gate_for_hot_tail_proxy()
    print("test_sage_lite_helpers.py passed")
