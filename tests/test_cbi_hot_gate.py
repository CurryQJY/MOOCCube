from types import SimpleNamespace
from pathlib import Path
import importlib

import torch
from torch import nn

from cbi_hot_gate import CBIHotGateFast3FeedbackUSIM
from cbi_hot_gate_audit_seed2025 import CBIHotGateAuditFast3FeedbackUSIM
from cbi_trust_sim import trust_build_eval_item_vecs


def test_hot_gate_routes_hot_items_but_cold_items_bypass_gate():
    model = CBIHotGateFast3FeedbackUSIM.__new__(CBIHotGateFast3FeedbackUSIM)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        content_delta_cold_only=False,
        content_delta_train_on_id_dropout=False,
        dropout_prob=0.0,
        disable_llm_score=True,
        llm_weight=0.0,
        llm_cold_only=False,
        llm_hot_only=False,
        content_delta_replace_item=True,
        content_delta_aux_mode="base",
    )
    model.training = False
    model.item_id_emb = nn.Embedding.from_pretrained(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), freeze=False
    )
    model.gate_net = nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())
    model.gate_net[0].weight.data.zero_()
    model.gate_net[0].bias.data.zero_()
    model._content_base_embedding = lambda idx: torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32
    )[idx]
    model._apply_content_delta = lambda content, idx, force_cold=False: content

    fused, id_true, aux_content = model.get_item_vector(
        torch.tensor([0, 1]),
        torch.tensor([-1.0, -1.0]),
        force_cold=torch.tensor([True, False]),
        disable_id_dropout=True,
    )

    assert torch.allclose(fused[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(fused[1], torch.tensor([0.70710677, 0.70710677]), atol=1e-6)
    assert torch.equal(id_true, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(aux_content, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_hot_gate_entrypoint_installs_hot_gate_and_all_item_eval():
    entrypoint = importlib.import_module("run_cbi_hot_gate_seed2025")
    fake_protocol = SimpleNamespace(Fast3FeedbackUSIM=object)
    fake_eval = SimpleNamespace(build_eval_item_vecs=None, build_eval_pos_item_vecs=None)

    entrypoint.install_protocol(fake_protocol, fake_eval)

    assert fake_protocol.Fast3FeedbackUSIM is CBIHotGateFast3FeedbackUSIM
    assert fake_eval.build_eval_item_vecs is trust_build_eval_item_vecs


def test_hot_gate_launcher_locks_single_seed_screen_configuration():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_hot_gate_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "run_cbi_hot_gate_seed2025.py"' in source
    assert 'OutputRoot = "outputs\\cbi_hot_gate_single_seed2025"' in source
    assert 'CheckpointRoot = "checkpoints\\cbi_hot_gate_single_seed2025"' in source
    assert "Seeds = @(2025)" in source
    assert "Epochs = 35" in source
    assert "Patience = 8" in source
    assert "ContentDeltaMaxNorm = 0.5" in source
    assert "AuxWeight = 0.3" in source
    assert "UsePseudoColdTrain = $false" in source
    assert "UsimSteps = 5" in source
    assert 'hot_only_gate = $true' in source


def test_hot_gate_audit_keeps_hot_fusion_scale_before_simulation():
    model = CBIHotGateAuditFast3FeedbackUSIM.__new__(CBIHotGateAuditFast3FeedbackUSIM)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        content_delta_cold_only=False,
        content_delta_train_on_id_dropout=False,
        dropout_prob=0.0,
        disable_llm_score=True,
        llm_weight=0.0,
        llm_cold_only=False,
        llm_hot_only=False,
        content_delta_aux_mode="base",
    )
    model.training = False
    model.item_id_emb = nn.Embedding.from_pretrained(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), freeze=False
    )
    model.gate_net = nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())
    model.gate_net[0].weight.data.zero_()
    model.gate_net[0].bias.data.zero_()
    model._content_base_embedding = lambda idx: torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32
    )[idx]
    model._apply_content_delta = lambda content, idx, force_cold=False: content

    fused, _, _ = model.get_item_vector(
        torch.tensor([0, 1]),
        torch.tensor([-1.0, -1.0]),
        force_cold=torch.tensor([True, False]),
        disable_id_dropout=True,
    )

    assert torch.allclose(fused[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(fused[1], torch.tensor([0.5, 0.5]), atol=1e-6)


def test_hot_gate_audit_launcher_locks_short_fresh_snapshot_screen():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_hot_gate_audit_seed2025.ps1").read_text(encoding="utf-8")

    assert 'ScriptPath = "cbi_hot_gate_audit_seed2025.py"' in source
    assert '$outputRoot = "outputs\\cbi_hot_gate_audit_seed2025"' in source
    assert '$checkpointRoot = "checkpoints\\cbi_hot_gate_audit_seed2025"' in source
    assert "Epochs = 8" in source
    assert "Patience = 8" in source
    assert 'USIM_FB_SNAPSHOT_EPOCHS = "1,2,3,4,5,6,7,8"' in source
    assert 'normalize_hot_fused_before_simulation = $false' in source


def test_hot_gate_cold_delta_launcher_changes_only_delta_scope():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_hot_gate_cold_delta_seed2025.ps1").read_text(encoding="utf-8")

    assert '$outputRoot = "outputs\\cbi_hot_gate_cold_delta_seed2025"' in source
    assert '$checkpointRoot = "checkpoints\\cbi_hot_gate_cold_delta_seed2025"' in source
    assert 'ScriptPath="cbi_hot_gate_cold_delta_seed2025.py"' in source
    assert "ContentDeltaColdOnly=$true" in source
    assert "Epochs=8" in source
    assert 'EarlyStopScoreMode="balanced_rn"' in source
    assert 'USIM_FB_SNAPSHOT_EPOCHS="1,2,3,4,5,6,7,8"' in source


def test_hot_gate_projector_pseudocold_launcher_activates_shared_cold_path():
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_cbi_hot_gate_projector_pseudocold_seed2025.ps1").read_text(
        encoding="utf-8"
    )

    assert '$outputRoot = "outputs\\cbi_hot_gate_projector_pseudocold_seed2025"' in source
    assert 'ScriptPath="cbi_hot_gate_projector_pseudocold_seed2025.py"' in source
    assert 'ContentDeltaMode="projector"' in source
    assert "ContentDeltaColdOnly=$true" in source
    assert "UsePseudoColdTrain=$true" in source
    assert 'PseudoColdMode="batch_random"' in source
    assert "PseudoColdRatio=0.30" in source
    assert "ContentDeltaMaxNorm=0.20" in source
    assert "Epochs=8" in source
    assert 'USIM_FB_SNAPSHOT_EPOCHS="1,2,3,4,5,6,7,8"' in source
