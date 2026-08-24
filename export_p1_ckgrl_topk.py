import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def make_read_only_torch_save(checkpoint_dir, real_save):
    checkpoint_root = Path(checkpoint_dir).resolve()

    def guarded_save(obj, path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            target = Path(path).resolve()
            if target.is_relative_to(checkpoint_root):
                print(f">> P1 READ-ONLY: blocked checkpoint write {target}")
                return None
        return real_save(obj, path, *args, **kwargs)

    return guarded_save


def build_runtime_environment(
    manifest_path,
    checkpoint_dir,
    output_dir,
    topk_output,
    top_k=20,
    model_label="ckg_rl",
):
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    env = {str(key): str(value) for key, value in (payload.get("env") or {}).items()}
    seed = int((payload.get("split") or {}).get("seed"))
    env.update(
        {
            "USIM_STATIC_SEED": str(seed),
            "USIM_SEED": str(seed),
            "USIM_FB_CKPT_DIR": str(checkpoint_dir),
            "USIM_FB_OUTPUT_DIR": str(output_dir),
            "USIM_FB_SAVE_CKPT": "1",
            "USIM_FB_AUTO_RESUME": "1",
            "USIM_FB_FORCE_FRESH": "0",
            "P1_TOPK_EXPORT_PATH": str(topk_output),
            "P1_TOPK_EXPORT_K": str(int(top_k)),
            "P1_TOPK_EXPORT_MODEL": str(model_label),
        }
    )
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk-output", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-label", default="ckg_rl")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    topk_output = Path(args.topk_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_runtime_environment(
        manifest_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        topk_output=topk_output,
        top_k=args.top_k,
        model_label=args.model_label,
    )
    os.environ.update(env)

    checkpoint_paths = [
        path for path in (checkpoint_dir / "latest.pt", checkpoint_dir / "finished.pt")
        if path.exists()
    ]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No latest.pt or finished.pt in {checkpoint_dir}")
    before_hashes = {str(path): _sha256(path) for path in checkpoint_paths}

    import usim_feedback_fast3_content_delta_recovered_51ea_candidate as legacy

    real_save = torch.save
    guarded_save = make_read_only_torch_save(checkpoint_dir, real_save)

    def block_feedback_checkpoint(ckpt_dir, state, snapshot_name=None):
        name = snapshot_name or "latest.pt"
        target = Path(ckpt_dir) / name
        print(f">> P1 READ-ONLY: blocked checkpoint write {target}")
        return str(target)

    legacy.torch.save = guarded_save
    legacy._save_feedback_checkpoint = block_feedback_checkpoint
    legacy.main()

    after_hashes = {str(path): _sha256(path) for path in checkpoint_paths}
    if before_hashes != after_hashes:
        raise RuntimeError("Checkpoint hash changed during read-only export")
    if not topk_output.exists():
        raise FileNotFoundError(f"Top-K export was not produced: {topk_output}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = {
        "model": str(args.model_label),
        "seed": int(manifest["split"]["seed"]),
        "top_k": int(args.top_k),
        "source_manifest": str(manifest_path),
        "checkpoint_hashes": before_hashes,
        "topk_output": str(topk_output),
        "record_count": sum(1 for _ in topk_output.open("r", encoding="utf-8")),
    }
    (output_dir / "p1_topk_export_manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
