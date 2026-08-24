import json

import pandas as pd

import usim_feedback_fast3_content_delta as legacy
import usim_feedback_fast3_content_delta_repaired as repaired


def test_static_manifest_records_provenance_fingerprints(tmp_path, monkeypatch):
    cfg = legacy.Fast3Config(2, 2)
    output = tmp_path / "static_protocol_manifest.json"
    monkeypatch.setattr(legacy, "_feedback_output_path", lambda name: str(output))
    provenance = {
        "schema_version": 1,
        "source_manifest_sha256": "source-sha",
        "training_config_sha256": "train-sha",
        "training_config_payload": {"ppo_loss_weight": 1.0},
        "split_sha256": "split-sha",
        "split_payload": {"seed": 2025},
    }

    legacy._write_static_manifest(
        {"split_mode": "strict_item_cold_balanced"},
        {},
        cfg,
        {},
        "data",
        pd.DataFrame({"u_idx": [0, 1], "i_idx": [0, 1]}),
        provenance=provenance,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["provenance"] == provenance


def test_repaired_entrypoint_exposes_resume_decision_wrapper():
    assert callable(repaired.repaired_checkpoint_resume_decision)
