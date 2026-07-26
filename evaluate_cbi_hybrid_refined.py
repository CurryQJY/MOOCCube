"""Evaluate frozen TDInit checkpoints with Cold refined and Hot left unrefined."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from cbi_anchor_sim import CBIAnchorFast3FeedbackUSIM
from evaluate_cbi_all_refined_seed2025 import (
    _build_loader,
    _cached_positive_bank,
    _load_splits,
    _metric_row,
    _model_parameter_digest,
    _read_original_metrics,
    _reset_usim_env,
    _set_seed,
    _sha256,
)
from fast3_delta.config import Fast3Config
from fast3_delta.course_artifacts import _empty_course_stats, build_course_artifacts
from fast3_delta.eval import evaluate_usim, prepare_llm_scores
from fast3_delta.static_protocol import add_user_seen_from_df, apply_train_popularity
from usim_feedback_fast3_content_delta import _resolve_torch_device, load_llm_scores_for_stream


METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
SUFFIXES = {
    "R@5": "r5",
    "R@10": "r10",
    "R@20": "r20",
    "N@5": "n5",
    "N@10": "n10",
    "N@20": "n20",
}


def build_cold_refined_hot_base_bank(
    model,
    device: torch.device,
    llm_scores=None,
    item_batch: int = 1024,
):
    """Build one bank where strict-cold items are simulated and Hot items are base vectors."""
    del llm_scores
    if getattr(model, "item_popularity", None) is None:
        raise RuntimeError("hybrid evaluation requires model.item_popularity")

    n_items = int(model.cfg.n_items)
    all_idx = torch.arange(n_items, device=device)
    popularity = model.item_popularity.to(device=device).float().view(-1)
    cold_mask = popularity < float(model.cfg.cold_threshold)
    cold_idx = all_idx[cold_mask]
    hot_idx = all_idx[~cold_mask]
    bank = torch.empty((n_items, int(model.cfg.emb_dim)), device=device)
    batch_size = max(1, int(item_batch))

    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            idx = all_idx[start : start + batch_size]
            llm_batch = torch.full((idx.numel(),), -1.0, dtype=torch.float32, device=device)
            force_cold = cold_mask.index_select(0, idx)
            base, _, _ = model.get_item_vector(
                idx,
                llm_batch,
                force_cold=force_cold,
                disable_id_dropout=True,
            )
            bank[idx] = F.normalize(base, dim=1)

        if cold_idx.numel() > 0:
            cached_user_bank = None
            if getattr(model.cfg, "candidate_strategy", "") == "retrieve_sample":
                cached_user_bank = model._build_user_bank_raw()
            llm_batch = torch.full(
                (cold_idx.numel(),), -1.0, dtype=torch.float32, device=device
            )
            refined = model.infer_refined_item_vectors(
                cold_idx,
                llm_s=llm_batch,
                item_batch=batch_size,
                force_cold=True,
                user_bank_raw=cached_user_bank,
            )
            bank[cold_idx] = F.normalize(refined, dim=1)

    return bank, {
        "cold_items": int(cold_idx.numel()),
        "hot_items": int(hot_idx.numel()),
        "total_items": n_items,
        "cold_refined": True,
        "hot_refined": False,
    }


def _default_manifest(root: Path, seed: int) -> Path:
    return (
        root
        / "outputs"
        / "cbi_anchor_sim_3seed_serial"
        / f"strict_item_cold_balanced_thr1_seed_{seed}"
        / "static_protocol_manifest.json"
    )


def _default_checkpoint(root: Path, seed: int) -> Path:
    checkpoint_root = (
        root / "checkpoints" / "cbi_anchor_sim_single_seed2025"
        if seed == 2025
        else root / "checkpoints" / "cbi_anchor_sim_3seed_serial"
    )
    return (
        checkpoint_root
        / f"strict_item_cold_balanced_thr1_seed_{seed}"
        / "finished.pt"
    )


def run_seed(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict:
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    _reset_usim_env(manifest.get("env", {}))
    seed = int(manifest["split"]["seed"])
    _set_seed(seed)

    data_dir = Path(manifest["data"]["data_dir"])
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    frame = pd.read_pickle(data_dir / "stream_data.pkl")
    content = torch.load(
        data_dir / "content_emb.pt", map_location="cpu", weights_only=False
    )
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
    train_df, val_df, test_df, train_pop = apply_train_popularity(
        train_df, val_df, test_df, cfg
    )
    artifact_df = (
        frame if manifest["split"].get("artifact_source") == "all_metadata" else train_df
    )
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

    model = CBIAnchorFast3FeedbackUSIM(cfg, content).to(device)
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(train_pop)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_state = checkpoint.get("es_best_state")
    if best_state is None:
        raise RuntimeError(f"checkpoint lacks es_best_state: {checkpoint_path}")
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
    item_bank, bank_stats = build_cold_refined_hot_base_bank(
        model, device, llm_scores=llm_scores, item_batch=1024
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
        raise RuntimeError("model parameters changed during hybrid evaluation")

    rows = [
        _metric_row("cold", cold_metrics, cold_count),
        _metric_row("hot", hot_metrics, hot_count),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "hybrid_fullrank.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    original_path = Path(manifest["exports"]["final_fullrank"])
    result = {
        "schema_version": 1,
        "experiment": "cbi_anchor_cold_refined_hot_base",
        "evaluation_only": True,
        "seed": seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_best_epoch": checkpoint.get("es_best", {}).get("epoch"),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_all_refined_result": str(original_path.resolve()),
        "script_sha256": _sha256(Path(__file__)),
        "device": str(device),
        "torch": torch.__version__,
        "llm_score_path": llm_path,
        "course_stats": course_stats,
        "semantics": {
            "model_space": "single_tdinit_checkpoint",
            "cold_refined": True,
            "hot_refined": False,
            "hot_representation": "same_checkpoint_base_content_plus_delta",
            "cross_checkpoint_vector_mixing": False,
            "deterministic_cold_rollout": True,
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
        "all_refined_metrics": _read_original_metrics(original_path),
    }
    (output_dir / "hybrid_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def _aggregate(results: list[dict], output_root: Path) -> None:
    detail_rows = []
    for result in results:
        row = {"seed": result["seed"]}
        for split in ("cold", "hot"):
            metrics = result["metrics"][split]
            for metric in METRICS:
                row[f"full_{split}_item_macro_{SUFFIXES[metric]}"] = metrics[metric]
            row[f"full_{split}_item_macro_count"] = metrics["item_count"]
        cold_count = float(row["full_cold_item_macro_count"])
        hot_count = float(row["full_hot_item_macro_count"])
        for metric in METRICS:
            suffix = SUFFIXES[metric]
            row[f"full_overall_item_macro_{suffix}"] = (
                row[f"full_cold_item_macro_{suffix}"] * cold_count
                + row[f"full_hot_item_macro_{suffix}"] * hot_count
            ) / (cold_count + hot_count)
        detail_rows.append(row)

    detail_path = output_root / "hybrid_3seed_runs_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = {"runs": len(detail_rows), "seeds": ",".join(str(r["seed"]) for r in detail_rows)}
    metric_fields = [key for key in detail_rows[0] if key not in {"seed"} and not key.endswith("count")]
    for field in metric_fields:
        values = [float(row[field]) for row in detail_rows]
        summary[f"{field}_mean"] = statistics.mean(values)
        summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    with (output_root / "hybrid_3seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs" / "cbi_anchor_hybrid_eval_3seed",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        manifest = _default_manifest(root, seed)
        checkpoint = _default_checkpoint(root, seed)
        seed_output = args.output_root / f"strict_item_cold_balanced_thr1_seed_{seed}"
        print(f"[HYBRID-EVAL] seed={seed}", flush=True)
        result = run_seed(manifest, checkpoint, seed_output)
        results.append(result)
        print(json.dumps(result["metrics"], ensure_ascii=False), flush=True)
    _aggregate(results, args.output_root)
    print(f"wrote {args.output_root / 'hybrid_3seed_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
