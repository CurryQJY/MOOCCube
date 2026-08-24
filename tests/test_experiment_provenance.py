import json
from pathlib import Path

import pytest

from fast3_delta.provenance import (
    ProvenanceCorruptionError,
    build_source_manifest,
    compare_source_manifests,
    create_provenance_snapshot,
    verify_provenance_snapshot,
)


def _source_tree(tmp_path):
    root = tmp_path / "repo"
    entrypoint = root / "train.py"
    module = root / "fast3_delta" / "config.py"
    runner = root / "run.ps1"
    module.parent.mkdir(parents=True)
    entrypoint.write_text("print('v1')\n", encoding="utf-8")
    module.write_text("VALUE = 1\n", encoding="utf-8")
    runner.write_text("Write-Host run\n", encoding="utf-8")
    return root, entrypoint, runner


def test_create_provenance_snapshot_is_immutable(tmp_path):
    root, entrypoint, runner = _source_tree(tmp_path)
    output = tmp_path / "output"

    first = create_provenance_snapshot(output, root, entrypoint, runner)
    entrypoint.write_text("print('v2')\n", encoding="utf-8")
    second = create_provenance_snapshot(output, root, entrypoint, runner)

    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]
    assert (output / "provenance" / "source" / "train.py").read_text(encoding="utf-8") == "print('v1')\n"


def test_source_manifest_covers_entrypoint_modules_and_runner(tmp_path):
    root, entrypoint, runner = _source_tree(tmp_path)

    manifest = build_source_manifest(root, entrypoint, runner)

    assert set(manifest["files"]) == {"train.py", "fast3_delta/config.py", "run.ps1"}
    assert all(record["sha256"] for record in manifest["files"].values())


def test_compare_source_manifests_reports_added_removed_and_modified():
    expected = {"files": {"same.py": {"sha256": "a"}, "old.py": {"sha256": "b"}, "changed.py": {"sha256": "c"}}}
    current = {"files": {"same.py": {"sha256": "a"}, "new.py": {"sha256": "d"}, "changed.py": {"sha256": "e"}}}

    assert compare_source_manifests(expected, current) == {
        "added": ["new.py"],
        "removed": ["old.py"],
        "modified": ["changed.py"],
    }


def test_verify_snapshot_detects_corruption(tmp_path):
    root, entrypoint, runner = _source_tree(tmp_path)
    output = tmp_path / "output"
    create_provenance_snapshot(output, root, entrypoint, runner)
    (output / "provenance" / "source" / "train.py").write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(ProvenanceCorruptionError, match="train.py"):
        verify_provenance_snapshot(output)
