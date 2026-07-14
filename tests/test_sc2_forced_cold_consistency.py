from __future__ import annotations

from pathlib import Path

import pytest
import torch

import usim_feedback_fast3_sc2_consistency as sc2_mod
from usim_feedback_fast3_sc2_consistency import (
    SC2ConsistencyConfig,
    SC2ConsistencyFast3FeedbackUSIM,
    forced_cold_distribution_consistency_loss,
)


def test_identical_logits_have_near_zero_consistency_loss() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    loss = forced_cold_distribution_consistency_loss(
        logits,
        logits.clone(),
        temperature=0.2,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_divergent_logits_only_backpropagate_through_student() -> None:
    teacher = torch.tensor(
        [[3.0, 0.0], [0.0, 3.0]],
        requires_grad=True,
    )
    student = torch.zeros_like(teacher, requires_grad=True)

    loss = forced_cold_distribution_consistency_loss(
        teacher,
        student,
        temperature=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert student.grad is not None
    assert student.grad.abs().sum().item() > 0.0
    assert teacher.grad is None


def test_no_active_rows_return_differentiable_student_zero() -> None:
    teacher = torch.randn(2, 3, requires_grad=True)
    student = torch.randn(2, 3, requires_grad=True)

    loss = forced_cold_distribution_consistency_loss(
        teacher,
        student,
        active_rows=torch.zeros(2, dtype=torch.bool),
        invalid_candidate_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert student.grad is not None
    assert torch.equal(student.grad, torch.zeros_like(student))
    assert teacher.grad is None


def test_masked_candidate_logits_do_not_affect_loss() -> None:
    teacher = torch.tensor([[2.0, -1.0, 0.5]])
    student = torch.tensor([[0.0, 1.0, -0.5]])
    invalid_mask = torch.tensor([[False, True, False]])

    baseline = forced_cold_distribution_consistency_loss(
        teacher,
        student,
        invalid_candidate_mask=invalid_mask,
    )
    changed = forced_cold_distribution_consistency_loss(
        torch.tensor([[2.0, 1_000.0, 0.5]]),
        torch.tensor([[0.0, -1_000.0, -0.5]]),
        invalid_candidate_mask=invalid_mask,
    )

    assert changed.item() == pytest.approx(baseline.item(), abs=1e-7)


@pytest.mark.parametrize(
    ("teacher", "student"),
    [
        (torch.zeros(2), torch.zeros(2)),
        (torch.zeros(2, 3), torch.zeros(2, 2)),
    ],
)
def test_invalid_logit_shapes_raise_value_error(teacher, student) -> None:
    with pytest.raises(ValueError):
        forced_cold_distribution_consistency_loss(teacher, student)


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_nonpositive_temperature_raises_value_error(temperature: float) -> None:
    with pytest.raises(ValueError):
        forced_cold_distribution_consistency_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "active_rows",
    [
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(2),
    ],
)
def test_invalid_active_row_mask_raises_value_error(active_rows) -> None:
    with pytest.raises(ValueError):
        forced_cold_distribution_consistency_loss(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            active_rows=active_rows,
        )


@pytest.mark.parametrize(
    "invalid_candidate_mask",
    [
        torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, 3),
    ],
)
def test_invalid_candidate_mask_raises_value_error(
    invalid_candidate_mask,
) -> None:
    with pytest.raises(ValueError):
        forced_cold_distribution_consistency_loss(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            invalid_candidate_mask=invalid_candidate_mask,
        )


def test_all_masked_active_row_raises_value_error() -> None:
    with pytest.raises(ValueError):
        forced_cold_distribution_consistency_loss(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            active_rows=torch.tensor([True, False]),
            invalid_candidate_mask=torch.tensor(
                [[True, True, True], [False, False, False]],
            ),
        )


def test_sc2_config_reads_isolated_environment(monkeypatch) -> None:
    monkeypatch.setenv("USIM_SC2_CONSISTENCY_WEIGHT", "0.25")
    monkeypatch.setenv("USIM_SC2_CONSISTENCY_TEMP", "0.4")
    monkeypatch.setenv("USIM_SC2_CONSISTENCY_WARM_ONLY", "0")
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")

    cfg = SC2ConsistencyConfig(n_users=2, n_items=3, content_dim=5)

    assert cfg.sc2_consistency_weight == pytest.approx(0.25)
    assert cfg.sc2_consistency_temp == pytest.approx(0.4)
    assert cfg.sc2_consistency_warm_only is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("USIM_SC2_CONSISTENCY_WEIGHT", "-0.1"),
        ("USIM_SC2_CONSISTENCY_TEMP", "0"),
    ],
)
def test_sc2_config_rejects_invalid_loss_controls(monkeypatch, name, value) -> None:
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        SC2ConsistencyConfig(n_users=2, n_items=3, content_dim=5)


def _tiny_sc2_model(monkeypatch, weight="0.25"):
    monkeypatch.setenv("USIM_DISABLE_LLM_SCORE", "1")
    monkeypatch.setenv("USIM_SC2_CONSISTENCY_WEIGHT", weight)
    cfg = SC2ConsistencyConfig(n_users=2, n_items=3, content_dim=5)
    model = SC2ConsistencyFast3FeedbackUSIM(
        cfg,
        torch.zeros((3, 5), dtype=torch.float32),
    )
    model.device = torch.device("cpu")
    return model


def test_sc2_consistency_uses_full_teacher_and_forced_cold_student(monkeypatch) -> None:
    model = _tiny_sc2_model(monkeypatch)
    model.train()
    calls = []

    def fake_get_item_vector(i_idx, llm_s, force_cold=False, disable_id_dropout=False):
        calls.append((force_cold, disable_id_dropout))
        if force_cold:
            vec = torch.tensor(
                [[0.0] + [1.0] * (model.cfg.emb_dim - 1), [1.0] + [0.0] * (model.cfg.emb_dim - 1)],
                requires_grad=True,
            )
        else:
            vec = torch.tensor(
                [[1.0] + [0.0] * (model.cfg.emb_dim - 1), [0.0] + [1.0] * (model.cfg.emb_dim - 1)],
            )
        return vec, vec, vec

    monkeypatch.setattr(model, "get_item_vector", fake_get_item_vector)
    batch = {"u": torch.tensor([0, 1]), "i": torch.tensor([1, 2])}

    loss, diagnostics = model._sc2_consistency_loss(
        batch,
        torch.tensor([10.0, 11.0]),
        llm_s=None,
        user_seen_items=None,
    )

    assert calls == [(False, True), (True, True)]
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert diagnostics["sc2_consistency_active_ratio"] == pytest.approx(1.0)
    assert "sc2_teacher_student_cosine" in diagnostics


def test_forward_adds_weighted_consistency_and_stats(monkeypatch) -> None:
    model = _tiny_sc2_model(monkeypatch, weight="0.25")
    base_loss = torch.tensor(2.0, requires_grad=True)
    consistency = torch.tensor(4.0, requires_grad=True)

    def fake_base_forward(self, batch, pop, llm_s, user_bank_raw=None, user_seen_items=None):
        return base_loss, {"main_loss": 2.0}

    diagnostics = {
        "sc2_consistency_loss": 4.0,
        "sc2_consistency_active_ratio": 1.0,
        "sc2_teacher_student_cosine": 0.5,
    }
    monkeypatch.setattr(sc2_mod.legacy.Fast3FeedbackUSIM, "forward", fake_base_forward)
    monkeypatch.setattr(
        model,
        "_sc2_consistency_loss",
        lambda *args, **kwargs: (consistency, diagnostics),
    )

    total, stats = model.forward(
        {"u": torch.tensor([0]), "i": torch.tensor([1])},
        torch.tensor([2.0]),
        llm_s=None,
    )

    assert total.item() == pytest.approx(3.0)
    assert stats["sc2_consistency_loss"] == pytest.approx(4.0)
    assert stats["sc2_consistency_weighted_loss"] == pytest.approx(1.0)


def test_install_sc2_bindings_replaces_only_config_and_model(monkeypatch) -> None:
    original_main = sc2_mod.legacy.main
    original_config = sc2_mod.legacy.Fast3Config
    original_model = sc2_mod.legacy.Fast3FeedbackUSIM
    try:
        sc2_mod.install_sc2_bindings()

        assert sc2_mod.legacy.Fast3Config is SC2ConsistencyConfig
        assert sc2_mod.legacy.Fast3FeedbackUSIM is SC2ConsistencyFast3FeedbackUSIM
        assert sc2_mod.legacy.main is original_main
    finally:
        sc2_mod.legacy.Fast3Config = original_config
        sc2_mod.legacy.Fast3FeedbackUSIM = original_model


def test_smoke_runner_is_isolated_from_main_table() -> None:
    text = Path("run_sc2_forced_cold_consistency_smoke.ps1").read_text(
        encoding="utf-8",
    )

    assert "usim_feedback_fast3_sc2_consistency.py" in text
    assert "outputs\\sc2_forced_cold_consistency_smoke" in text
    assert "checkpoints\\sc2_forced_cold_consistency_smoke" in text
    assert "Seeds = @(2025)" in text
    assert "Epochs = 1" in text
    assert "SkipAggregate = $true" in text
    assert "[int]$MinFreeGpuMiB = 9000" in text
    assert "[int]$GpuPollSeconds = 30" in text
    assert "nvidia-smi --query-gpu=memory.free" in text
    assert "GPU memory query unavailable; retry" in text
    assert "throw \"Unable to query free GPU memory" not in text
    assert "[int]::TryParse" in text
    assert '$LASTEXITCODE -eq 0 -and' not in text
    assert "aggregate_fast3_static_results.py" not in text
    assert "paper_aaai27" not in text
