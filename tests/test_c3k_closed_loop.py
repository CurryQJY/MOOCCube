from pathlib import Path

import pandas as pd
import torch

from fast3_delta.config import Fast3Config
from fast3_delta.c3k_eval import build_c3k_item_bank, evaluate_c3k
from fast3_delta.c3k_model import C3KFeedbackUSIM
from c3k_static import (
    C3KRunConfig,
    build_c3k_manifest_payload,
    select_validation_rows,
    summarize_stable_epoch_timing,
)


def _model():
    cfg = Fast3Config(n_users=3, n_items=4, content_dim=5)
    cfg.cold_threshold = 1
    cfg.dropout_prob = 0.0
    cfg.use_mixed_hard_neg = False
    model = C3KFeedbackUSIM(
        cfg,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    model.device = torch.device("cpu")
    model.set_feedback_item_stats(torch.tensor([0.0, 8.0, 5.0, 3.0]))
    model.item_concept_overlap = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.8, 0.0],
            [0.0, 0.8, 1.0, 0.2],
            [0.0, 0.0, 0.2, 1.0],
        ]
    )
    model.item_prereq_item_mat = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    model.item_prereq_item_cnt = torch.tensor([0.0, 1.0, 1.0, 1.0])
    return model


def test_c3k_paired_item_views_bypass_random_id_dropout_and_mask_only_selected_item():
    torch.manual_seed(7)
    model = _model()
    model.train()
    model.cfg.dropout_prob = 1.0

    item_ids = torch.tensor([1, 2], dtype=torch.long)
    pseudo_mask = torch.tensor([False, True])
    llm_scores = torch.full((2,), -1.0)

    full, masked = model.paired_item_views(item_ids, llm_scores, pseudo_mask)

    assert torch.allclose(full[0], masked[0])
    assert not torch.allclose(full[1], masked[1])


def test_c3k_structural_features_remove_the_training_target_from_history():
    model = _model()

    with_target = model.structural_features(
        torch.tensor([0]), torch.tensor([1]), {0: {1, 2}}
    )
    without_target = model.structural_features(
        torch.tensor([0]), torch.tensor([1]), {0: {2}}
    )

    assert torch.allclose(with_target, without_target)


def test_c3k_gate_has_required_signs_and_bound():
    model = _model()
    coefficients = model.knowledge_coefficients(
        torch.randn(3, model.cfg.emb_dim),
        torch.randn(3, model.cfg.emb_dim),
        torch.rand(3, 4),
    )

    assert torch.all(coefficients[:, 0] >= 0)
    assert torch.all(coefficients[:, 1:] <= 0)
    assert torch.all(coefficients.abs() <= model.c3k_gate_max + 1e-7)


def test_c3k_gate_starts_as_a_small_calibration_not_a_large_static_bias():
    model = _model()
    coefficients = model.knowledge_coefficients(
        torch.randn(8, model.cfg.emb_dim),
        torch.randn(8, model.cfg.emb_dim),
        torch.rand(8, 4),
    )

    assert float(coefficients.detach().abs().max().item()) < 0.01


def test_c3k_pair_score_matches_catalog_score_for_same_user_item_evidence():
    model = _model()
    model.eval()
    strict_cold = torch.tensor([True, False, True, False])
    bank = model.build_item_bank(strict_cold)
    history = {0: {1}}

    pair = model.score_pairs(
        torch.tensor([0]), torch.tensor([2]), bank[2:3], history
    )
    catalog = model.score_catalog(torch.tensor([0]), bank, history, item_block=2)

    assert torch.allclose(pair.view(-1), catalog[:, 2].view(-1), atol=1e-6)


def test_c3k_vectorized_structural_grid_matches_individual_pair_features():
    model = _model()
    users = torch.tensor([0, 1])
    candidates = torch.tensor([[1, 2], [2, 3]])
    history = {0: {1, 2}, 1: {2, 3}}

    grid = model.structural_feature_grid(users, candidates, history)
    pairs = model.structural_features(
        users[:, None].expand_as(candidates).reshape(-1),
        candidates.reshape(-1),
        history,
    ).view(2, 2, 4)

    assert torch.allclose(grid, pairs, atol=1e-6)


def test_c3k_forward_uses_paired_pseudocold_rank_loss_without_simulator(monkeypatch):
    model = _model()
    model.train()
    model.set_pseudo_cold_item_mask(torch.tensor([False, True, False, False]))

    def unexpected_legacy_simulator(*args, **kwargs):
        raise AssertionError("C3K must never invoke the legacy simulator")

    monkeypatch.setattr(model, "run_usim_episode", unexpected_legacy_simulator)
    batch = {
        "u": torch.tensor([0, 1, 2], dtype=torch.long),
        "i": torch.tensor([1, 2, 3], dtype=torch.long),
    }
    loss, diagnostics = model(
        batch,
        torch.tensor([8.0, 5.0, 3.0]),
        torch.full((3,), -1.0),
        user_seen_items={0: {1}, 1: {2}, 2: {3}},
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["pseudo_cold_count"] == 1
    assert diagnostics["consistency_loss"] >= 0.0
    assert diagnostics["gate_regularization"] >= 0.0


def test_c3k_item_bank_masks_strict_cold_but_retains_warm_id_evidence():
    model = _model()
    model.eval()

    bank = build_c3k_item_bank(model, torch.device("cpu"), item_batch=2)

    assert bank.strict_cold_mask.tolist() == [True, False, False, False]
    assert not torch.allclose(bank.item_vectors[1], bank.all_cold_item_vectors[1])
    assert bank.item_bank_seconds >= 0.0


def test_c3k_evaluator_routes_full_ranking_through_shared_catalog_score(monkeypatch):
    model = _model()
    model.eval()
    calls = []

    def fake_catalog(user_ids, item_bank, user_history, *, item_block, calibration=True):
        del item_bank, user_history, item_block, calibration
        calls.append(user_ids.detach().cpu().tolist())
        return torch.zeros((user_ids.numel(), model.cfg.n_items))

    monkeypatch.setattr(model, "score_catalog", fake_catalog)
    loader = [
        (
            {"u": torch.tensor([0]), "i": torch.tensor([2])},
            torch.tensor([0.0]),
            torch.tensor([-1.0]),
        )
    ]
    metrics, count, timing = evaluate_c3k(
        model,
        loader,
        torch.device("cpu"),
        eval_type="cold",
        k_list=(1,),
        user_seen_items={0: set()},
        query_block=1,
        item_block=2,
    )

    assert calls == [[0]]
    assert count == 1
    assert set(metrics) == {"R@1", "N@1"}
    assert timing["candidate_count"] == model.cfg.n_items


def test_c3k_stable_epoch_timing_excludes_first_epoch():
    summary = summarize_stable_epoch_timing(
        [
            {"epoch": 1, "seconds": 10.0, "batches": 5, "samples": 20},
            {"epoch": 2, "seconds": 2.0, "batches": 5, "samples": 20},
            {"epoch": 3, "seconds": 4.0, "batches": 5, "samples": 20},
        ],
        seed=2025,
        source_hash="abc",
        selected_epoch=3,
    )

    assert summary["stable_epoch_count"] == 2
    assert summary["mean_seconds"] == 3.0
    assert summary["samples_per_second"] == 20.0 / 3.0
    assert summary["excluded_warmup_epoch"] == 1


def test_c3k_manifest_and_launcher_are_isolated_from_legacy_rl():
    payload = build_c3k_manifest_payload(
        seed=2025,
        source_hash="abc",
        pseudo_cold={"selection_source": "train_popularity_only"},
        run_config={"hot_tolerance": 0.003},
    )
    launcher = Path("run_c3k_3seed.ps1").read_text(encoding="utf-8")

    assert payload["method"] == "C3K"
    assert payload["id_dropout"] == "disabled"
    assert payload["pseudo_cold"]["selection_source"] == "train_popularity_only"
    assert "USIM_CKG_RL_V1" not in launcher
    assert "USIM_PPO_LOSS_WEIGHT" not in launcher
    assert "USIM_USE_REFINED_EVAL" not in launcher


def test_c3k_run_config_supports_validation_only_preflight(monkeypatch):
    monkeypatch.setenv("C3K_VALIDATION_ONLY", "1")
    monkeypatch.setenv("C3K_GATE_MAX", "0.17")
    monkeypatch.setenv("C3K_CONSISTENCY_WEIGHT", "0.08")
    monkeypatch.setenv("C3K_GATE_WEIGHT", "0.002")
    monkeypatch.setenv("C3K_TRAIN_NEGATIVES", "12")
    monkeypatch.setenv("C3K_WARM_SEEN", "6")
    monkeypatch.setenv("C3K_REDUNDANCY_THRESHOLD", "0.65")

    config = C3KRunConfig.from_environment()

    assert config.validation_only is True
    assert config.gate_max == 0.17
    assert config.consistency_weight == 0.08
    assert config.gate_weight == 0.002
    assert config.train_negatives == 12
    assert config.warm_seen == 6.0
    assert config.redundancy_threshold == 0.65


def test_c3k_validation_subset_is_seeded_and_keeps_requested_row_count():
    frame = pd.DataFrame({"u_idx": range(10), "i_idx": range(10)})

    first = select_validation_rows(frame, max_rows=3, seed=2025)
    second = select_validation_rows(frame, max_rows=3, seed=2025)

    assert len(first) == 3
    assert first.index.tolist() == second.index.tolist()
