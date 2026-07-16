from pathlib import Path
import inspect
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_new_model_has_an_isolated_entrypoint_and_output_roots():
    source = (ROOT / "learner_guided_cold_refinement.py").read_text(encoding="utf-8")
    runner = (ROOT / "run_learner_guided_full_seed2025.ps1").read_text(encoding="utf-8")

    assert "usim_feedback_fast3_content_delta_recovered_51ea_candidate" in source
    assert "learner_guided_full" in runner
    assert "main_table_51ea12fc_candidate" not in runner


def test_learner_fit_is_finite_nonnegative_and_penalty_monotone():
    from learner_guided_cold_refinement import learner_fit

    similarity = torch.tensor([[-1.0, 0.0, 1.0]])
    concept = torch.tensor([[0.2, 0.2, 0.2]])
    low_gap = torch.zeros_like(similarity)
    high_gap = torch.ones_like(similarity)

    low = learner_fit(similarity, concept, low_gap, low_gap, low_gap)
    high = learner_fit(similarity, concept, high_gap, high_gap, high_gap)

    assert torch.isfinite(low).all()
    assert ((low >= 0.0) & (low <= 1.0)).all()
    assert torch.all(high <= low)


def test_bounded_refinement_preserves_warm_rows_and_caps_displacement():
    from learner_guided_cold_refinement import bounded_learner_refinement

    initial = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor(
        [
            [[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]],
        ]
    )
    user_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    fit = torch.tensor([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
    effective_cold = torch.tensor([False, True, True])

    refined, diagnostics = bounded_learner_refinement(
        initial,
        candidates,
        user_ids,
        fit,
        effective_cold,
        steps=3,
        lr=1.0,
        min_fit=0.05,
        step_cap=0.04,
        total_cap=0.08,
    )

    assert torch.equal(refined[~effective_cold], initial[~effective_cold])
    displacement = torch.linalg.vector_norm(refined - initial, dim=1)
    assert displacement.max().item() <= 0.080001
    assert diagnostics["step_displacement_max"] <= 0.040001
    assert diagnostics["repeated_user_rate"] == 0.0


def test_nonpositive_or_small_fit_stops_row_without_update():
    from learner_guided_cold_refinement import bounded_learner_refinement

    initial = torch.zeros((2, 2))
    candidates = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    user_ids = torch.tensor([[1, 2], [3, 4]])
    fit = torch.tensor([[0.04, 0.03], [0.20, 0.10]])

    refined, diagnostics = bounded_learner_refinement(
        initial,
        candidates,
        user_ids,
        fit,
        torch.tensor([True, True]),
        steps=2,
        min_fit=0.05,
    )

    assert torch.equal(refined[0], initial[0])
    assert not torch.equal(refined[1], initial[1])
    assert diagnostics["stopped_ratio"] > 0.0


def test_selected_users_are_unique_within_each_episode_row():
    from learner_guided_cold_refinement import bounded_learner_refinement

    initial = torch.zeros((1, 2))
    candidates = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    user_ids = torch.tensor([[10, 11, 12]])
    fit = torch.tensor([[0.9, 0.8, 0.7]])

    _, diagnostics = bounded_learner_refinement(
        initial,
        candidates,
        user_ids,
        fit,
        torch.tensor([True]),
        steps=3,
        min_fit=0.0,
    )

    assert diagnostics["selected_user_ids"] == [[10, 11, 12]]
    assert diagnostics["repeated_user_rate"] == 0.0


def test_lira_model_overrides_legacy_episode_without_actor_calls():
    from learner_guided_cold_refinement import LearnerGuidedColdModel

    source = LearnerGuidedColdModel.run_usim_episode.__code__.co_names
    assert "agent" not in source
    assert "bounded_learner_refinement" in source


def test_formal_runner_disables_legacy_objectives_and_uses_three_steps():
    runner = (ROOT / "run_learner_guided_full_seed2025.ps1").read_text(encoding="utf-8")
    required = [
        '-MinimalLiraMode',
        '-UsimSteps $UsimSteps',
        '-RolloutPolicy "course_fit"',
    ]
    for fragment in required:
        assert fragment in runner
    for legacy_flag in ["UsePaac", "UseSageLite", "UseCgrcRecon", "PpoLossWeight"]:
        assert legacy_flag not in runner


def test_static_runner_accepts_explicit_delegate_entrypoint_without_weakening_legacy_guard():
    model_source = (ROOT / "learner_guided_cold_refinement.py").read_text(encoding="utf-8")
    runner = (ROOT / "run_usim_feedback_fast3_content_delta_static.ps1").read_text(encoding="utf-8")

    assert "USIM_STATIC_DELEGATE_ENTRYPOINT = True" in model_source
    assert "USIM_STATIC_DELEGATE_ENTRYPOINT" in runner
    assert '"def run_static_experiment"' in runner
    assert '"_static_split_df"' in runner


def test_lira_disables_auxiliary_computation_and_locks_target_exclusion():
    from learner_guided_cold_refinement import LearnerGuidedColdModel

    assert "_compute_aux_loss" in LearnerGuidedColdModel.__dict__
    runner = (ROOT / "run_learner_guided_full_seed2025.ps1").read_text(encoding="utf-8")
    assert '$env:USIM_FB_COURSE_MATCH_EXCLUDE_TARGET = "1"' in runner


def test_lira_physically_prunes_unused_legacy_modules(monkeypatch):
    from fast3_delta.config import FeedbackConfig
    from learner_guided_cold_refinement import LearnerGuidedColdModel

    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    cfg = FeedbackConfig(n_users=3, n_items=4, content_dim=5)
    model = LearnerGuidedColdModel(cfg, torch.zeros((4, 5)))
    child_names = dict(model.named_children())
    for unused in [
        "agent", "llm_proj", "content_delta", "content_delta_projector",
        "sage_gate_bucket_emb", "sage_gate_mlp", "sage_score_gate_mlp",
        "cgrc_recon_mlp",
    ]:
        assert unused not in child_names


def test_epoch_history_exposes_lira_diagnostics():
    source = (ROOT / "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py").read_text(
        encoding="utf-8"
    )
    for key in [
        "LIRAUpdateActiveRatio",
        "LIRAStoppedRatio",
        "LIRAStepDisplacementMean",
        "LIRATotalDisplacementMean",
        "LIRARepeatedUserRate",
    ]:
        assert f'"{key}"' in source
    assert " | LIRA[active=" in source


def test_runner_parameterizes_steps_for_validation_screening():
    runner = (ROOT / "run_learner_guided_full_seed2025.ps1").read_text(encoding="utf-8")
    assert "[int]$UsimSteps = 3" in runner
    assert "usim_steps = $UsimSteps" in runner
    assert "-UsimSteps $UsimSteps" in runner


def test_zero_step_arm_bypasses_learner_retrieval():
    from learner_guided_cold_refinement import LearnerGuidedColdModel

    source = inspect.getsource(LearnerGuidedColdModel.run_usim_episode)
    zero_step = source.index('if int(getattr(self.cfg, "usim_steps", 3)) <= 0:')
    retrieval = source.index("self._build_user_bank_raw()")
    assert zero_step < retrieval


def test_core_screen_runs_clean_t0_t1_and_t3_in_separate_roots():
    source = (ROOT / "run_learner_guided_seed2025_core_screen.ps1").read_text(encoding="utf-8")
    for name in ["lira_clean_t0_seed2025", "lira_one_step_t1_seed2025", "lira_full_t3_seed2025"]:
        assert name in source
    assert "Steps = 0" in source and "Steps = 1" in source and "Steps = 3" in source


def test_screening_defaults_to_validation_only_and_skips_final_test():
    runner = (ROOT / "run_learner_guided_full_seed2025.ps1").read_text(encoding="utf-8")
    training = (ROOT / "usim_feedback_fast3_content_delta_recovered_51ea_candidate.py").read_text(
        encoding="utf-8"
    )
    assert "[bool]$ValidationOnly = $true" in runner
    assert '$env:USIM_VALIDATION_ONLY = if ($ValidationOnly)' in runner
    assert "[STATIC-VALIDATION-ONLY]" in training
    validation_guard = training.index('if bool(getattr(cfg, "validation_only", False)):')
    test_eval = training.index('print("  [STATIC-TEST] Build eval item bank')
    assert validation_guard < test_eval
