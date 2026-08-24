from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from usim_feedback_fast3_content_delta_recovered_51ea_candidate import (
    Fast3FeedbackUSIM,
    _apply_refinement_only_to_effective_cold,
    _batch_invariant_alignment_grad,
    _build_fixed_tail_pseudo_item_mask,
    _coursefit_active_update_mask,
    _deterministic_candidate_positions,
    _exclude_previously_selected_users,
    _finite_tensor_mean,
    _remove_masked_items_from_seen_history,
    _remove_target_from_seen_history,
    _training_episode_target,
)
from fast3_delta.config import FeedbackConfig
from fast3_delta.eval import refined_eval_enabled


def test_refinement_is_applied_only_to_effective_cold_rows():
    base = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    refined = base + 10.0
    effective_cold = torch.tensor([False, True, False])

    actual = _apply_refinement_only_to_effective_cold(base, refined, effective_cold)

    expected = torch.tensor([[1.0, 1.0], [12.0, 12.0], [3.0, 3.0]])
    assert torch.equal(actual, expected)


def test_alignment_gradient_is_independent_of_actual_batch_length():
    current_one = torch.zeros((1, 2), dtype=torch.float32)
    user_one = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    target_one = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    alpha_one = torch.tensor([[0.25]], dtype=torch.float32)

    grad_one = _batch_invariant_alignment_grad(
        current_one,
        user_one,
        target_one,
        alpha_one,
        reference_batch_size=2048,
    )

    grad_many = _batch_invariant_alignment_grad(
        current_one.repeat(7, 1),
        user_one.repeat(7, 1),
        target_one.repeat(7, 1),
        alpha_one.repeat(7, 1),
        reference_batch_size=2048,
    )

    assert torch.allclose(grad_one[0], grad_many[0])


def test_training_script_locks_pseudocold_coursefit_configuration():
    script = (ROOT / "run_coursefit_pseudocold_minimal_seed2025.ps1").read_text(
        encoding="utf-8"
    )

    required_fragments = [
        '[switch]$ForceFresh',
        '-UsePseudoColdTrain $true',
        '-PseudoColdMode "item_tail"',
        '-PseudoColdRatio 0.3',
        '-PseudoColdMinPop 5',
        '-TrainForceCold $true',
        '-UsimSteps 5',
        '-PpoLossWeight 0.0',
        '-RolloutPolicy "course_fit"',
        '-UseCourseReward $false',
        '$env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET = "1"',
    ]
    for fragment in required_fragments:
        assert fragment in script

    assert "coursefit_pseudocold_minimal_seed2025" in script
    assert "coursefit_rollout_train5_seed2025" not in script


def test_training_helpers_are_defined_before_script_main_entrypoint():
    source = (
        ROOT / "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py"
    ).read_text(encoding="utf-8")

    main_index = source.index('if __name__ == "__main__":')
    assert source.index("def _apply_refinement_only_to_effective_cold") < main_index
    assert source.index("def _batch_invariant_alignment_grad") < main_index


def test_training_model_natively_enables_refined_validation(monkeypatch):
    monkeypatch.setenv("USIM_USE_REFINED_EVAL", "1")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = FeedbackConfig(n_users=3, n_items=4, content_dim=5)
    cfg.use_usim_refined_eval = True
    model = Fast3FeedbackUSIM(cfg, torch.zeros((4, 5), dtype=torch.float32))

    assert hasattr(model, "infer_refined_item_vectors")
    assert refined_eval_enabled(model) is True


def test_coursefit_without_ppo_uses_same_unanchored_target_as_inference():
    base = torch.randn((3, 4), dtype=torch.float32, requires_grad=True)

    coursefit_target = _training_episode_target(base, "course_fit", 0.0)
    ppo_target = _training_episode_target(base, "ppo", 1.0)

    assert coursefit_target is None
    assert torch.equal(ppo_target, base.detach())
    assert ppo_target.requires_grad is False


def test_fixed_item_tail_mask_targets_popularity_mass_not_batch_rows():
    popularity = torch.tensor([1.0, 2.0, 3.0, 10.0, 0.0])

    mask = _build_fixed_tail_pseudo_item_mask(
        popularity,
        ratio=0.30,
        min_pop=1,
    )

    assert mask.tolist() == [True, True, True, False, False]
    item_rows = torch.tensor([0, 3, 0, 2, 3, 2])
    assert mask[item_rows].tolist() == [True, False, True, True, False, True]


def test_target_is_removed_from_shared_history_for_all_course_terms():
    seen = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float32
    )
    counts = seen.sum(dim=1, keepdim=True)
    targets = torch.tensor([1, 2], dtype=torch.long)

    cleaned, cleaned_counts = _remove_target_from_seen_history(
        seen,
        counts,
        targets,
    )

    assert torch.equal(cleaned, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert torch.equal(cleaned_counts, torch.ones((2, 1)))
    assert torch.equal(seen, torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))


def test_all_pseudocold_items_are_removed_from_course_history():
    seen = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    pseudo_items = torch.tensor([False, True, False, True])

    cleaned, counts = _remove_masked_items_from_seen_history(seen, pseudo_items)

    assert torch.equal(cleaned, torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(counts, torch.tensor([[2.0], [0.0]]))


def test_previously_selected_users_are_excluded_per_episode_row():
    candidates = torch.tensor([[4, 5, 6], [4, 5, 6]])
    previous = torch.tensor([[5, 9], [4, 6]])
    scores = torch.tensor([[0.2, 0.9, 0.3], [0.8, 0.7, 0.6]])

    masked = _exclude_previously_selected_users(scores, candidates, previous)

    assert torch.equal(masked.argmax(dim=1), torch.tensor([2, 1]))
    assert torch.isneginf(masked[0, 1])
    assert torch.isneginf(masked[1, 0]) and torch.isneginf(masked[1, 2])


def test_nonpositive_coursefit_causes_no_representation_update():
    fit = torch.tensor([[-0.2, -0.1], [-0.1, 0.3], [0.0, -0.4]])

    active = _coursefit_active_update_mask(fit, torch.tensor([1, 1, 0]))

    assert active.tolist() == [False, True, False]


def test_coursefit_candidate_pool_selection_is_deterministic_and_unique():
    probs = torch.tensor([[0.1, 0.4, 0.2, 0.3], [0.7, 0.1, 0.1, 0.1]])

    first = _deterministic_candidate_positions(probs, 3)
    second = _deterministic_candidate_positions(probs, 3)

    assert torch.equal(first, second)
    assert first.tolist() == [[1, 3, 2], [0, 1, 2]]
    assert all(len(set(row)) == 3 for row in first.tolist())


def test_formal_script_blocks_pseudocold_id_auxiliary_leakage():
    script = (ROOT / "run_coursefit_pseudocold_minimal_seed2025.ps1").read_text(encoding="utf-8")
    assert "-AuxHotOnly $true" in script


def test_coursefit_logging_ignores_exclusion_sentinels():
    values = torch.tensor([[0.2, float("-inf")], [0.4, float("inf")]])
    assert _finite_tensor_mean(values) == pytest.approx(0.3)
