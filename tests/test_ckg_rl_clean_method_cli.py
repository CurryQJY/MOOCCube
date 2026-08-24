"""CLI checks for the clean CKG-RL method contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_clean_method_cli_writes_contract_json(tmp_path):
    output = tmp_path / "contract.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ckg_rl_clean_method_cli.py"),
            "--output",
            str(output),
            "--max-policy-delta",
            "0.35",
            "--stability-anchor-count",
            "128",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert json.loads(result.stdout)["status"] == "ok"
    assert payload["method"] == "clean_ckg_rl"
    assert payload["evaluation"]["ranking_bank"] == "single_unified_catalog_bank"
    assert payload["policy_bounds"]["max_policy_delta"] == 0.35
    assert payload["policy_bounds"]["stability_anchor_count"] == 128


def test_clean_method_cli_rejects_legacy_flags(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ckg_rl_clean_method_cli.py"),
            "--output",
            str(tmp_path / "bad.json"),
            "--legacy-dual-vector-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "legacy dual-vector" in result.stderr
