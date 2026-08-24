"""Replay the frozen CBI seed-2025 checkpoint with all items USIM-refined."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import fast3_delta.eval as eval_mod
from fast3_delta.config import Fast3Config
from fast3_delta.course_artifacts import _empty_course_stats, build_course_artifacts
from fast3_delta.eval import evaluate_usim, prepare_llm_scores
from fast3_delta.static_protocol import StreamDataset, add_user_seen_from_df, apply_train_popularity
from usim_feedback_fast3_content_delta import (
    Fast3FeedbackUSIM,
    _resolve_torch_device,
    load_llm_scores_for_stream,
)


DEFAULT_MANIFEST = Path(
    "outputs/cbi_faithful_single_seed2025/"
    "strict_item_cold_balanced_thr1_seed_2025/static_protocol_manifest.json"
)
DEFAULT_CHECKPOINT = Path(
    "checkpoints/cbi_faithful_single_seed2025/"
    "strict_item_cold_balanced_thr1_seed_2025/finished.pt"
)
DEFAULT_OUTPUT = Path("outputs/cbi_faithful_seed2025_eval_all_refined")
USIM_PREFIX = "USIM_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_usim_env(manifest_env: dict) -> None:
    for key in list(os.environ):
        if key.startswith(USIM_PREFIX):
            os.environ.pop(key, None)
    for key, value in manifest_env.items():
        if key.startswith(USIM_PREFIX):
            os.environ[key] = str(value)
    os.environ["USIM_STATIC"] = "1"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cached_bank_positive_vectors(item_bank: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
    """Return positive rows from the exact bank used for candidate ranking."""
    return item_bank.index_select(0, item_idx.to(device=item_bank.device, dtype=torch.long))


def build_all_refined_item_bank(
    model,
    device: torch.device,
    llm_scores=None,
    item_batch: int = 1024,
):
    """Refine strict-cold and warm items with their matching force-cold semantics."""
    if getattr(model, "item_popularity", None) is None:
        raise RuntimeError("all-refined evaluation requires model.item_popularity")

    n_items = int(model.cfg.n_items)
    all_idx = torch.arange(n_items, device=device)
    popularity = model.item_popularity.to(device=device).float().view(-1)
    cold_mask = popularity < float(model.cfg.cold_threshold)
    cold_idx = all_idx[cold_mask]
    hot_idx = all_idx[~cold_mask]
    bank = torch.empty((n_items, int(model.cfg.emb_dim)), device=device)

    cached_user_bank = None
    if getattr(model.cfg, "candidate_strategy", "") == "retrieve_sample":
        cached_user_bank = model._build_user_bank_raw()

    def refine(idx: torch.Tensor, force_cold: bool) -> None:
        if idx.numel() == 0:
            return
        llm_batch = torch.full((idx.numel(),), -1.0, dtype=torch.float32, device=device)
        refined = model.infer_refined_item_vectors(
            idx,
            llm_s=llm_batch,
            item_batch=item_batch,
            force_cold=force_cold,
            user_bank_raw=cached_user_bank,
        )
        bank[idx] = F.normalize(refined, dim=1)

    refine(cold_idx, True)
    refine(hot_idx, False)
    return bank, {
        "cold_items": int(cold_idx.numel()),
        "hot_items": int(hot_idx.numel()),
        "total_items": n_items,
    }


@contextlib.contextmanager
def _cached_positive_bank(item_bank: torch.Tensor):
    original = eval_mod.build_eval_pos_item_vecs

    def from_cache(model, item_idx, llm_s, pop_sel, eval_type, **kwargs):
        del model, llm_s, pop_sel, eval_type, kwargs
        return cached_bank_positive_vectors(item_bank, item_idx)

    eval_mod.build_eval_pos_item_vecs = from_cache
    try:
        yield
    finally:
        eval_mod.build_eval_pos_item_vecs = original


def _load_splits(manifest: dict):
    exports = manifest["exports"]
    return tuple(
        pd.read_pickle(exports[key])
        for key in ("train_split", "val_split", "test_split")
    )


def _build_loader(test_df, llm_scores, batch_size: int):
    return DataLoader(
        StreamDataset(test_df, llm_scores),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: (
            {
                "u": torch.stack([item["u"] for item in batch]),
                "i": torch.stack([item["i"] for item in batch]),
            },
            torch.stack([item["pop"] for item in batch]),
            torch.stack([item["llm"] for item in batch]),
        ),
    )


def _model_parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _metric_row(eval_type: str, metrics: dict, count: int) -> dict:
    return {
        "eval_type": eval_type,
        "item_count": int(count),
        **{key: float(metrics[key]) for key in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")},
    }


def _read_original_metrics(path: Path) -> dict:
    row = pd.read_csv(path).iloc[0]
    return {
        "cold": {
            "R@5": float(row["full_cold_item_macro_r5"]),
            "R@10": float(row["full_cold_item_macro_r10"]),
            "R@20": float(row["full_cold_item_macro_r20"]),
            "N@5": float(row["full_cold_item_macro_n5"]),
            "N@10": float(row["full_cold_item_macro_n10"]),
            "N@20": float(row["full_cold_item_macro_n20"]),
        },
        "hot": {
            "R@5": float(row["full_hot_item_macro_r5"]),
            "R@10": float(row["full_hot_item_macro_r10"]),
            "R@20": float(row["full_hot_item_macro_r20"]),
            "N@5": float(row["full_hot_item_macro_n5"]),
            "N@10": float(row["full_hot_item_macro_n10"]),
            "N@20": float(row["full_hot_item_macro_n20"]),
        },
    }


def _write_comparison(path: Path, original: dict, current: dict) -> None:
    lines = [
        "# CBI seed-2025: all-item refined checkpoint replay",
        "",
        "| Split | Metric | Original | All refined | Absolute change | Relative change |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("cold", "hot"):
        for metric in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20"):
            before = original[split][metric]
            after = current[split][metric]
            delta = after - before
            relative = delta / before if before else 0.0
            lines.append(
                f"| {split.title()} | {metric} | {before:.6f} | {after:.6f} | "
                f"{delta:+.6f} | {relative:+.2%} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    # 先保存先修开关（_reset_usim_env 会清掉所有 USIM_ 变量，需在其后恢复）
    _prereq_target = os.environ.get("USIM_PREREQ_TARGET", "0")
    _prereq_path = os.environ.get("USIM_PREREQ_TARGET_PATH", "")
    _reset_usim_env(manifest.get("env", {}))
    # 恢复先修开关（本实验的唯一变量，主表manifest里没有此键）
    os.environ["USIM_PREREQ_TARGET"] = _prereq_target
    if _prereq_path:
        os.environ["USIM_PREREQ_TARGET_PATH"] = _prereq_path
    seed = int(manifest["split"]["seed"])
    _set_seed(seed)

    data_dir = Path(manifest["data"]["data_dir"])
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    frame = pd.read_pickle(data_dir / "stream_data.pkl")
    content = torch.load(data_dir / "content_emb.pt", map_location="cpu", weights_only=False)
    llm_scores, llm_path, _ = load_llm_scores_for_stream(
        str(data_dir),
        frame,
        cold_threshold=int(manifest["split"]["cold_threshold"]),
        n_users=meta["n_users"],
        n_items=meta["n_items"],
        fallback_data_dirs=["processed_data"],
        verbose=False,
    )

    cfg = Fast3Config(meta["n_users"], meta["n_items"], content.shape[1])
    llm_scores, llm_summary = prepare_llm_scores(llm_scores, cfg)
    cfg.llm_bank_mode = llm_summary["mode"]
    device = _resolve_torch_device()

    train_df, val_df, test_df = _load_splits(manifest)
    train_df, val_df, test_df, train_pop = apply_train_popularity(train_df, val_df, test_df, cfg)
    artifact_df = frame if manifest["split"].get("artifact_source") == "all_metadata" else train_df
    if cfg.feedback_load_course_artifacts:
        course_artifacts, course_stats = build_course_artifacts(
            artifact_df,
            cfg.n_items,
            relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
        )
    else:
        course_artifacts, course_stats = None, _empty_course_stats(cfg.n_items)

    model = Fast3FeedbackUSIM(cfg, content).to(device)
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(train_pop)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_state = checkpoint.get("es_best_state")
    if best_state is None:
        raise RuntimeError("checkpoint does not contain es_best_state")
    incompatible = model.load_state_dict(best_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    train_seen = add_user_seen_from_df({}, train_df)
    if manifest["split"].get("test_history_policy") == "train_val":
        train_seen = add_user_seen_from_df(train_seen, val_df)
    model.set_user_seen_index(train_seen)
    loader = _build_loader(test_df, llm_scores, batch_size=int(cfg.batch_size))

    before_digest = _model_parameter_digest(model)
    bank_started = time.perf_counter()
    with torch.no_grad():
        item_bank, bank_stats = build_all_refined_item_bank(
            model,
            device,
            llm_scores=llm_scores,
            item_batch=1024,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bank_seconds = time.perf_counter() - bank_started

    ranking_started = time.perf_counter()
    with _cached_positive_bank(item_bank), torch.no_grad():
        cold_metrics, cold_count = evaluate_usim(
            model,
            loader,
            device,
            llm_scores,
            k_list=[5, 10, 20],
            n_neg=cfg.eval_n_neg,
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            all_item_vecs=item_bank,
            average_mode="item_macro",
        )
        hot_metrics, hot_count = evaluate_usim(
            model,
            loader,
            device,
            llm_scores,
            k_list=[5, 10, 20],
            n_neg=cfg.eval_n_neg,
            eval_type="hot",
            full_ranking=True,
            user_seen_items=train_seen,
            all_item_vecs=item_bank,
            average_mode="item_macro",
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ranking_seconds = time.perf_counter() - ranking_started
    after_digest = _model_parameter_digest(model)
    if before_digest != after_digest:
        raise RuntimeError("model parameters changed during evaluation")

    rows = [
        _metric_row("cold", cold_metrics, cold_count),
        _metric_row("hot", hot_metrics, hot_count),
    ]
    current = {
        row["eval_type"]: {key: row[key] for key in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")}
        for row in rows
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_refined_fullrank.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    original_path = Path(manifest["exports"]["final_fullrank"])
    original = _read_original_metrics(original_path)
    comparison_path = output_dir / "comparison.md"
    _write_comparison(comparison_path, original, current)

    result = {
        "schema_version": 1,
        "experiment": "cbi_faithful_seed2025_eval_all_refined",
        "evaluation_only": True,
        "seed": seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_best_epoch": checkpoint.get("es_best", {}).get("epoch"),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_result": str(original_path.resolve()),
        "script_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "torch": torch.__version__,
        "llm_score_path": llm_path,
        "course_stats": course_stats,
        "semantics": {
            "cold_force_cold": True,
            "hot_force_cold": False,
            "cold_refined": True,
            "hot_refined": True,
            "deterministic_rollout": True,
            "target_emb": None,
            "shared_candidate_and_positive_bank": True,
            "usim_steps": int(cfg.usim_steps),
        },
        "bank_stats": bank_stats,
        "parameter_digest_before": before_digest,
        "parameter_digest_after": after_digest,
        "timing_seconds": {
            "item_bank": bank_seconds,
            "ranking": ranking_seconds,
            "total": time.perf_counter() - started,
        },
        "metrics": {row["eval_type"]: row for row in rows},
        "original_metrics": original,
    }
    (output_dir / "all_refined_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_evaluation(args.manifest, args.checkpoint, args.output_dir)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {args.output_dir / 'all_refined_fullrank.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'all_refined_manifest.json'}", flush=True)
    print(f"wrote {args.output_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
