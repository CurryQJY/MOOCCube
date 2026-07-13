import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


PROVENANCE_SCHEMA_VERSION = 1


class ProvenanceCorruptionError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), payload


def _relative_source_files(source_root, entrypoint, runner_path):
    root = Path(source_root).resolve()
    selected = [Path(entrypoint).resolve()]
    module_root = root / "fast3_delta"
    if module_root.exists():
        selected.extend(sorted(module_root.rglob("*.py")))
    if runner_path:
        runner = Path(runner_path).resolve()
        if runner.exists():
            selected.append(runner)
    unique = {}
    for path in selected:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = "external/" + path.name
        unique[relative] = path
    return root, unique


def build_source_manifest(source_root, entrypoint, runner_path=""):
    _, files = _relative_source_files(source_root, entrypoint, runner_path)
    records = {
        relative: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for relative, path in sorted(files.items())
    }
    return {"schema_version": PROVENANCE_SCHEMA_VERSION, "files": records}


def _manifest_path(output_dir):
    return Path(output_dir) / "provenance" / "source_manifest.json"


def verify_provenance_snapshot(output_dir):
    manifest_path = _manifest_path(output_dir)
    if not manifest_path.exists():
        raise ProvenanceCorruptionError(f"missing provenance manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_dir = manifest_path.parent / "source"
    for relative, record in manifest.get("files", {}).items():
        captured = source_dir / Path(relative)
        if not captured.exists() or sha256_file(captured) != record.get("sha256"):
            raise ProvenanceCorruptionError(f"provenance snapshot corrupted: {relative}")
    manifest_sha = sha256_file(manifest_path)
    return {"source_manifest": manifest, "source_manifest_sha256": manifest_sha, "provenance_dir": str(manifest_path.parent)}


def create_provenance_snapshot(output_dir, source_root, entrypoint, runner_path=""):
    manifest_path = _manifest_path(output_dir)
    if manifest_path.exists():
        return verify_provenance_snapshot(output_dir)

    provenance_dir = manifest_path.parent
    temp_dir = provenance_dir.with_name(provenance_dir.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    source_dir = temp_dir / "source"
    root, files = _relative_source_files(source_root, entrypoint, runner_path)
    manifest = build_source_manifest(root, entrypoint, runner_path)
    source_dir.mkdir(parents=True, exist_ok=False)
    try:
        for relative, source in sorted(files.items()):
            destination = source_dir / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (temp_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        provenance_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, provenance_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return verify_provenance_snapshot(output_dir)


def compare_source_manifests(expected, current):
    old = expected.get("files", {})
    new = current.get("files", {})
    old_paths = set(old)
    new_paths = set(new)
    return {
        "added": sorted(new_paths - old_paths),
        "removed": sorted(old_paths - new_paths),
        "modified": sorted(path for path in old_paths & new_paths if old[path].get("sha256") != new[path].get("sha256")),
    }


def write_resume_source_audit(output_dir, source_root, entrypoint, runner_path=""):
    current = build_source_manifest(source_root, entrypoint, runner_path)
    audit_dir = Path(output_dir) / "provenance" / "resume_checks"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / time.strftime("%Y%m%d_%H%M%S_source_manifest.json")
    audit_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    return current, str(audit_path)


def build_runtime_metadata(source_root, normalized_command=None):
    root = str(Path(source_root).resolve())
    def _git(*args):
        try:
            return subprocess.check_output(["git", "-C", root, *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "normalized_command": normalized_command or "",
    }


def build_split_fingerprint(split_info, exports=None):
    exports = exports or {}
    payload = {key: split_info[key] for key in sorted(split_info) if key not in {"created_at", "created_at_unix"}}
    artifacts = {}
    for name in ("train_split", "val_split", "test_split", "split_assignments"):
        path = exports.get(name)
        if path and os.path.exists(path):
            artifacts[name] = sha256_file(path)
    payload["artifact_sha256"] = artifacts
    return stable_fingerprint(payload)
