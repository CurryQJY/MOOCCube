from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_probe_is_validation_only_and_uses_one_frozen_checkpoint():
    source = (ROOT / "lira_validation_inference_steps.py").read_text(encoding="utf-8")
    assert '"evaluation_target": "validation"' in source
    assert 'payload.get("es_best_state") or payload["model_state"]' in source
    assert "test_loader" not in source
    assert "optimizer" not in source
    assert "loss.backward" not in source
