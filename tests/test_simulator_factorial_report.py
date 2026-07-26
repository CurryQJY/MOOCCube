import json

import pandas as pd
import pytest

import simulator_factorial_report as report


def write_audit(path, checkpoint_steps, inference_steps):
    path.write_text(
        json.dumps(
            {
                "checkpoint_usim_steps_values": checkpoint_steps,
                "effective_inference_usim_steps_values": inference_steps,
            }
        ),
        encoding="utf-8",
    )


def test_validate_simulator_step_audit_accepts_t0_training_t5_inference(tmp_path):
    path = tmp_path / "actor_inference_audit.json"
    write_audit(path, [0], [5])

    payload = report.validate_simulator_step_audit(path)

    assert payload["checkpoint_usim_steps_values"] == [0]
    assert payload["effective_inference_usim_steps_values"] == [5]


@pytest.mark.parametrize(
    "checkpoint_steps,inference_steps",
    [([5], [5]), ([0], [0]), ([0, 5], [5])],
)
def test_validate_simulator_step_audit_rejects_wrong_step_provenance(
    tmp_path, checkpoint_steps, inference_steps
):
    path = tmp_path / "actor_inference_audit.json"
    write_audit(path, checkpoint_steps, inference_steps)

    with pytest.raises(ValueError, match="checkpoint T=0.*inference T=5"):
        report.validate_simulator_step_audit(path)


def test_rename_factorial_columns_uses_simulator_semantics():
    source = pd.DataFrame(
        [
            {
                "on_static": 0.2,
                "off_course_fit": 0.3,
                "training_effect_static": 0.01,
                "inference_effect_ppo_off": 0.04,
            }
        ]
    )

    renamed = report.rename_factorial_columns(source)

    assert "t5_training_static" in renamed
    assert "t0_training_course_fit" in renamed
    assert "simulator_training_effect_static" in renamed
    assert "course_fit_effect_t0_training" in renamed
