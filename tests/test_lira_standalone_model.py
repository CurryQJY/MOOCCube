from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_lira_core_files_do_not_import_legacy_model():
    for relative in ["lira/config.py", "lira/refinement.py", "lira/model.py"]:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "usim_feedback_fast3" not in source
        assert "Fast3FeedbackUSIM" not in source
        assert "Actor" not in source
        assert "Critic" not in source
        assert "PPO" not in source


def test_lira_config_contains_only_used_model_controls():
    from lira.config import LIRAConfig

    cfg = LIRAConfig(n_users=3, n_items=4, content_dim=5)
    assert cfg.steps == 3
    assert cfg.step_cap == 0.05
    assert not hasattr(cfg, "ppo_loss_weight")
    assert not hasattr(cfg, "use_sage_lite")
    assert not hasattr(cfg, "llm_weight")


def test_standalone_model_has_only_declared_encoder_modules():
    from lira.config import LIRAConfig
    from lira.model import LIRAModel

    cfg = LIRAConfig(n_users=3, n_items=4, content_dim=5, embedding_dim=8)
    model = LIRAModel(cfg, torch.zeros((4, 5)))

    assert set(dict(model.named_children())) == {
        "user_embedding",
        "item_id_embedding",
        "item_content_embedding",
        "content_projection",
        "user_projection",
        "fusion_gate",
    }


def test_standalone_forward_preserves_warm_rows_and_returns_finite_loss():
    from lira.config import LIRAConfig
    from lira.model import LIRAModel

    cfg = LIRAConfig(n_users=4, n_items=5, content_dim=3, embedding_dim=4, steps=1)
    model = LIRAModel(cfg, torch.randn((5, 3)))
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    effective_cold = torch.tensor([False, True, True])
    candidate_vectors = torch.randn((3, 3, 4))
    candidate_ids = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
    candidate_fit = torch.tensor([[0.9, 0.8, 0.7], [0.8, 0.7, 0.6], [0.7, 0.6, 0.5]])

    output = model(
        users,
        items,
        effective_cold,
        candidate_vectors=candidate_vectors,
        candidate_user_ids=candidate_ids,
        candidate_fit=candidate_fit,
    )

    assert torch.isfinite(output.loss)
    assert output.logits.shape == (3, 3)
    assert torch.equal(output.refined_items[0], output.base_items[0])
    assert output.diagnostics["repeated_user_rate"] == 0.0
    assert torch.isfinite(output.base_loss)
    assert torch.isfinite(output.refinement_loss)
    assert torch.isfinite(output.stability_loss)


def test_dynamic_provider_is_called_again_with_updated_state_and_history():
    from lira.refinement import dynamic_bounded_refinement

    initial = torch.zeros((1, 2))
    calls = []

    def provider(current, selected_history):
        calls.append((current.detach().clone(), [list(row) for row in selected_history]))
        vectors = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        ids = torch.tensor([[10, 11]])
        fit = torch.tensor([[0.9, 0.8]])
        return vectors, ids, fit

    refined, diagnostics = dynamic_bounded_refinement(
        initial,
        torch.tensor([True]),
        provider,
        steps=2,
        update_lr=0.1,
        min_fit=0.05,
        min_gain=0.0,
        step_cap=0.05,
        total_cap=0.10,
    )

    assert len(calls) == 2
    assert not torch.equal(calls[0][0], calls[1][0])
    assert calls[1][1] == [[10]]
    assert diagnostics["selected_user_ids"] == [[10, 11]]
