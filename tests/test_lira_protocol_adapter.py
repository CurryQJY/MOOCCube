from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_protocol_adapter_extends_standalone_model_not_legacy_class():
    from lira.model import LIRAModel
    from lira.protocol_adapter import LIRAProtocolAdapter

    assert issubclass(LIRAProtocolAdapter, LIRAModel)
    assert all("Fast3" not in base.__name__ for base in LIRAProtocolAdapter.__mro__)


def test_protocol_adapter_exposes_shared_trainer_and_evaluator_interface():
    from fast3_delta.config import FeedbackConfig
    from lira.protocol_adapter import LIRAProtocolAdapter

    cfg = FeedbackConfig(n_users=4, n_items=5, content_dim=3)
    cfg.usim_steps = 1
    model = LIRAProtocolAdapter(cfg, torch.randn((5, 3)))
    model.device = torch.device("cpu")
    model.set_feedback_item_stats(torch.tensor([4, 3, 2, 1, 0]))
    model.set_user_seen_index({0: {0, 1}, 1: {1}, 2: {2}, 3: {3}})

    assert model.user_emb is model.user_embedding
    assert model.user_proj is model.user_projection
    assert callable(model.get_item_vector)
    assert callable(model.infer_refined_item_vectors)
    assert callable(model.apply_course_rerank)


def test_protocol_adapter_forward_returns_shared_trainer_tuple():
    from fast3_delta.config import FeedbackConfig
    from lira.protocol_adapter import LIRAProtocolAdapter

    cfg = FeedbackConfig(n_users=4, n_items=5, content_dim=3)
    cfg.usim_steps = 1
    cfg.num_candidates = 3
    model = LIRAProtocolAdapter(cfg, torch.randn((5, 3)))
    model.device = torch.device("cpu")
    model.set_feedback_item_stats(torch.tensor([4, 3, 2, 1, 0]))
    model.set_user_seen_index({0: {0}, 1: {1}, 2: {2}, 3: {3}})
    batch = {"u": torch.tensor([0, 1, 2]), "i": torch.tensor([0, 3, 4])}

    loss, diagnostics = model(
        batch,
        torch.tensor([4, 1, 0]),
        torch.zeros(3),
        user_seen_items={0: {0}, 1: {1}, 2: {2}, 3: {3}},
    )

    assert torch.isfinite(loss)
    assert diagnostics["ppo_loss"] == 0.0
    assert diagnostics["aux_loss"] == 0.0
    assert diagnostics["repeated_user_rate"] == 0.0


def test_clean_entry_installs_standalone_adapter_into_shared_protocol():
    source = (ROOT / "lira_entry.py").read_text(encoding="utf-8")
    assert "LIRAProtocolAdapter" in source
    assert "learner_guided_cold_refinement" not in source
    assert "USIM_STATIC_DELEGATE_ENTRYPOINT = True" in source


def test_refined_inference_accepts_shared_evaluator_keyword_contract():
    import inspect
    from lira.protocol_adapter import LIRAProtocolAdapter

    parameters = inspect.signature(LIRAProtocolAdapter.infer_refined_item_vectors).parameters
    assert "llm_s" in parameters
    assert "item_batch" in parameters
    assert "force_cold" in parameters


def test_lira_runtime_controls_propagate_from_environment(monkeypatch):
    from fast3_delta.config import FeedbackConfig
    from lira.protocol_adapter import LIRAProtocolAdapter

    monkeypatch.setenv("LIRA_UPDATE_LR", "0.07")
    monkeypatch.setenv("LIRA_MIN_FIT", "0.15")
    monkeypatch.setenv("LIRA_STEP_CAP", "0.03")
    monkeypatch.setenv("LIRA_TOTAL_CAP", "0.08")
    cfg = FeedbackConfig(n_users=3, n_items=4, content_dim=5)
    cfg.usim_steps = 3
    model = LIRAProtocolAdapter(cfg, torch.zeros((4, 5)))

    assert model.config.update_lr == 0.07
    assert model.config.min_fit == 0.15
    assert model.config.step_cap == 0.03
    assert model.config.total_cap == 0.08
