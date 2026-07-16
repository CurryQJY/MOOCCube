import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def build_cgrc_runtime_environment(
    *,
    seed: int,
    split_dir,
    checkpoint_dir,
    output_dir,
    topk_output,
    top_k: int = 20,
    analysis_split: str = "test",
) -> dict[str, str]:
    seed = int(seed)
    if analysis_split not in {"validation", "test"}:
        raise ValueError(f"unsupported analysis split: {analysis_split}")
    return {
        "USIM_DATA_DIR": "processed_data_hin_clean_pop5",
        "USIM_COLD_THRESHOLD": "1",
        "USIM_STATIC_TEST_HISTORY": "train_only",
        "USIM_EVAL_N_NEG": "200",
        "USIM_STATIC_SPLIT_DIR": str(split_dir),
        "USIM_BASELINE_OUTPUT_DIR": str(output_dir),
        "USIM_STATIC_SEED": str(seed),
        "CGRC_PAPER_STATIC_EPOCHS": "50",
        "CGRC_PAPER_BATCH_SIZE": "4096",
        "CGRC_PAPER_EVAL_N_NEG": "200",
        "CGRC_PAPER_COLD_THRESHOLD": "1",
        "CGRC_PAPER_BEST_AVERAGE_MODE": "item_macro",
        "CGRC_PAPER_RUN_SAMPLED_EVAL": "0",
        "CGRC_PAPER_MASK_RHO": "0.3",
        "CGRC_PAPER_RECON_TOPK": "20",
        "CGRC_PAPER_LAMBDA_E": "1.0",
        "CGRC_PAPER_TAU": "0.5",
        "CGRC_PAPER_STATIC_SEED": str(seed),
        "CGRC_PAPER_SEED": str(seed),
        "CGRC_PAPER_CKPT_DIR": str(checkpoint_dir),
        "CGRC_PAPER_SAVE_CKPT": "0",
        "CGRC_PAPER_AUTO_RESUME": "1",
        "CGRC_PAPER_FORCE_FRESH": "0",
        "CGRC_PAPER_SAVE_OPT_STATE": "0",
        "CGRC_PAPER_EXPORT_TOPK_PATH": str(topk_output),
        "CGRC_PAPER_EXPORT_TOPK_K": str(int(top_k)),
        "CGRC_PAPER_EVAL_SPLIT": analysis_split,
    }


def build_export_manifest(
    *,
    seed: int,
    top_k: int,
    checkpoint_paths,
    checkpoint_sha256_before: dict[str, str],
    checkpoint_sha256_after: dict[str, str],
    split_paths,
    script_paths,
    topk_output,
    native_result,
    record_count: int,
    analysis_split: str = "test",
) -> dict:
    if analysis_split not in {"validation", "test"}:
        raise ValueError(f"unsupported analysis split: {analysis_split}")
    before = dict(checkpoint_sha256_before)
    after = dict(checkpoint_sha256_after)
    if before != after:
        raise RuntimeError("checkpoint changed during export")

    checkpoint_bindings = []
    for raw_path in checkpoint_paths:
        path = Path(raw_path).resolve()
        key = str(path)
        current = _sha256(path)
        if before.get(key) != current or after.get(key) != current:
            raise RuntimeError(f"checkpoint hash does not bind current file: {path}")
        checkpoint_bindings.append(
            {
                "path": key,
                "size": int(path.stat().st_size),
                "sha256_before": before[key],
                "sha256_after": after[key],
            }
        )

    topk_binding = _file_binding(Path(topk_output))
    native_binding = _file_binding(Path(native_result))
    actual_count = sum(1 for _ in Path(topk_output).open("r", encoding="utf-8"))
    if actual_count != int(record_count):
        raise RuntimeError(
            f"Top-K record count changed while building manifest: {actual_count} != {record_count}"
        )
    native_rows = json.loads(Path(native_result).read_text(encoding="utf-8"))
    if not native_rows or int(native_rows[0]["count_full_cold"]) != actual_count:
        raise RuntimeError("native cold-split coverage does not match Top-K export")
    native_split = str(native_rows[0].get("evaluation_split", "test"))
    if native_split != analysis_split:
        raise RuntimeError(
            f"native evaluation split mismatch: {native_split} != {analysis_split}"
        )

    return {
        "schema_version": 1,
        "model": "cgrc",
        "seed": int(seed),
        "analysis_split": analysis_split,
        "top_k": int(top_k),
        "record_count": actual_count,
        "target_course_count": int(native_rows[0].get("count_full_cold_item_macro", 0)),
        "restored_state": "latest.pt:best_state",
        "checkpoints": checkpoint_bindings,
        "split_files": [_file_binding(Path(path)) for path in split_paths],
        "script_files": [_file_binding(Path(path)) for path in script_paths],
        "topk_output": topk_binding,
        "native_result": native_binding,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--analysis-split", choices=("validation", "test"), default="test")
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    split_dir = args.split_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    topk_output = args.topk_output.resolve()
    manifest_output = (
        args.manifest_output.resolve()
        if args.manifest_output
        else topk_output.parent / "export_manifest.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [checkpoint_dir / "latest.pt", checkpoint_dir / "best.pt"]
    missing = [path for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing CGRC checkpoint files: {missing}")
    before = {str(path.resolve()): _sha256(path) for path in checkpoint_paths}

    environment = build_cgrc_runtime_environment(
        seed=args.seed,
        split_dir=split_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        topk_output=topk_output,
        top_k=args.top_k,
        analysis_split=args.analysis_split,
    )
    os.environ.update(environment)

    import cgrc_paper_static_hin

    cgrc_paper_static_hin.main()

    after = {str(path.resolve()): _sha256(path) for path in checkpoint_paths}
    native_result = output_dir / "cgrc_paper_static_result.json"
    if not topk_output.is_file() or not native_result.is_file():
        raise FileNotFoundError("CGRC replay did not produce required outputs")
    native_rows = json.loads(native_result.read_text(encoding="utf-8"))
    native_split = str(native_rows[0].get("evaluation_split", "test")) if native_rows else ""
    if native_split != args.analysis_split:
        raise RuntimeError(
            f"CGRC replay evaluated {native_split!r}, expected {args.analysis_split!r}"
        )

    split_names = (
        "static_protocol_manifest.json",
        "static_split_assignments.csv",
        "static_split_counts.csv",
        "static_split_sources.csv",
        "static_split_summary.json",
        "static_train.pkl",
        "static_val.pkl",
        "static_test.pkl",
    )
    split_paths = [split_dir / name for name in split_names]
    script_paths = [
        Path(__file__),
        root / "cgrc_paper_static_hin.py",
        root / "hin_eval_common.py",
        root / "ranking_topk_export.py",
        root / "baseline_checkpoint.py",
    ]
    record_count = sum(1 for _ in topk_output.open("r", encoding="utf-8"))
    manifest = build_export_manifest(
        seed=args.seed,
        top_k=args.top_k,
        checkpoint_paths=checkpoint_paths,
        checkpoint_sha256_before=before,
        checkpoint_sha256_after=after,
        split_paths=split_paths,
        script_paths=script_paths,
        topk_output=topk_output,
        native_result=native_result,
        record_count=record_count,
        analysis_split=args.analysis_split,
    )
    _write_json_atomic(manifest_output, manifest)
    print(f"[P1-CGRC] wrote {manifest_output}", flush=True)


if __name__ == "__main__":
    main()
