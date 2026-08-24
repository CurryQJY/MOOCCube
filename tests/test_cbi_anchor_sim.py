import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

from cbi_anchor_sim import CBIAnchorFast3FeedbackUSIM
from cbi_trust_sim import trust_build_eval_item_vecs, trust_build_eval_pos_item_vecs
from fast3_delta.config import Fast3Config
from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM


def test_anchor_simulator_replaces_any_caller_target_with_initial_cbi(monkeypatch):
    captured = []

    def fake_parent_episode(self, init_item_emb, target_emb=None, **kwargs):
        del self, kwargs
        captured.append(target_emb)
        return init_item_emb, {"rewards": []}, {"steps": 0}

    monkeypatch.setattr(Fast3FeedbackUSIM, "run_usim_episode", fake_parent_episode)
    model = CBIAnchorFast3FeedbackUSIM.__new__(CBIAnchorFast3FeedbackUSIM)
    initial_cbi = torch.tensor([[0.6, 0.8]], requires_grad=True)

    first = model.run_usim_episode(initial_cbi, target_emb=torch.tensor([[1.0, 0.0]]))
    second = model.run_usim_episode(initial_cbi, target_emb=torch.tensor([[0.0, 1.0]]))

    assert first[0] is initial_cbi
    assert second[0] is initial_cbi
    assert len(captured) == 2
    for effective_target in captured:
        assert torch.equal(effective_target, initial_cbi)
        assert effective_target.requires_grad is False
        assert effective_target.data_ptr() == initial_cbi.data_ptr()


def test_anchor_entrypoint_installs_model_eval_hooks_and_resume_reason_bridge():
    entrypoint = importlib.import_module("run_cbi_anchor_sim_seed2025")

    def fake_resume_decision(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(reason="fingerprint match")

    fake_protocol = SimpleNamespace(
        Fast3FeedbackUSIM=object,
        checkpoint_resume_decision=fake_resume_decision,
        build_eval_item_vecs=None,
        build_eval_pos_item_vecs=None,
    )
    fake_eval = SimpleNamespace(build_eval_item_vecs=None, build_eval_pos_item_vecs=None)

    entrypoint.install_protocol(fake_protocol, fake_eval)
    decision = fake_protocol.checkpoint_resume_decision(object(), Fast3Config(2, 3, 5))

    assert fake_protocol.Fast3FeedbackUSIM is CBIAnchorFast3FeedbackUSIM
    assert fake_eval.build_eval_item_vecs is trust_build_eval_item_vecs
    assert fake_eval.build_eval_pos_item_vecs is trust_build_eval_pos_item_vecs
    assert decision.reason == "fingerprint match"
    assert fake_protocol.cfg_reason == "fingerprint match"


def test_anchor_entrypoint_declares_static_runner_delegation():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_anchor_sim_seed2025.py").read_text(encoding="utf-8")

    assert "USIM_STATIC_DELEGATE_ENTRYPOINT = True" in source


def test_anchor_launcher_locks_isolated_reproducible_configuration():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_anchor_sim_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "run_cbi_anchor_sim_seed2025.py"' in source
    assert 'OutputRoot = "outputs\\cbi_anchor_sim_single_seed2025"' in source
    assert 'CheckpointRoot = "checkpoints\\cbi_anchor_sim_single_seed2025"' in source
    assert "Seeds = @(2025)" in source
    assert "    Epochs = 60\n" in source
    assert "    Patience = 10\n" in source
    assert 'EarlyStopAverageMode = "item_macro"' in source
    assert 'EarlyStopScoreMode = "cold_only"' in source
    assert "ContentDeltaMaxNorm = 0.5" in source
    assert "UsimSteps = 5" in source
    assert "AutoResume = $true" in source
    assert "SaveOptState = $true" in source
    assert '"cbi_anchor_sim.py"' in source
    assert '"run_cbi_anchor_sim_seed2025.py"' in source
    assert '"paper_aaai27\\main.tex"' not in source


def test_anchor_three_seed_launcher_is_serial_isolated_and_resumable():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_anchor_sim_3seed_serial.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "run_cbi_anchor_sim_seed2025.py"' in source
    assert 'OutputRoot = "outputs\\cbi_anchor_sim_3seed_serial"' in source
    assert 'CheckpointRoot = "checkpoints\\cbi_anchor_sim_3seed_serial"' in source
    assert '"background_logs\\cbi_anchor_sim_3seed_serial"' in source
    assert "Seeds = @(2026, 2027)" in source
    assert 'aggregate_seeds = @(2025, 2026, 2027)' in source
    assert 'reused_seed = 2025' in source
    assert '"outputs\\cbi_anchor_sim_single_seed2025"' in source
    assert "    Epochs = 60\n" in source
    assert "    Patience = 10\n" in source
    assert "ContentDeltaMaxNorm = 0.5" in source
    assert "UsimSteps = 5" in source
    assert "AutoResume = $true" in source
    assert "ForceFresh = $false" in source
    assert "hard_projection = $false" in source
    assert '"paper_aaai27\\main.tex"' not in source


def test_anchor_aux_screen_locks_three_serial_seed2025_arms():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_anchor_aux_screen_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "run_cbi_anchor_sim_seed2025.py"' in source
    assert '"outputs\\cbi_anchor_aux_screen_seed2025"' in source
    assert '"checkpoints\\cbi_anchor_aux_screen_seed2025"' in source
    assert "AuxWeight = 0.0" in source
    assert "AuxWeight = 0.1" in source
    assert "AuxWeight = 0.3" in source
    assert "Seeds = @(2025)" in source
    assert "Epochs = 30" in source
    assert "Patience = 6" in source
    assert 'execution = "serial"' in source


def test_anchor_aux_queue_waits_for_completed_upstream_and_fails_closed():
    root = Path(__file__).resolve().parents[1]
    source = (root / "wait_cbi_anchor_3seed_then_aux_screen.ps1").read_text(encoding="utf-8")

    assert '"outputs\\cbi_anchor_sim_3seed_serial\\run_manifest.json"' in source
    assert 'status -eq "completed"' in source
    assert 'status -eq "failed"' in source
    assert '"run_cbi_anchor_aux_screen_seed2025.ps1"' in source
    assert "Start-Sleep -Seconds $PollIntervalSec" in source
    assert 'status = "waiting"' in source
