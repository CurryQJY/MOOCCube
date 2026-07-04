import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fast3_delta.course_artifacts import build_course_artifacts, _empty_course_stats
from fast3_delta.eval import build_eval_item_vecs, evaluate_usim, prepare_llm_scores
from fast3_delta.static_protocol import (
    StreamDataset,
    add_user_seen_from_df,
    apply_train_popularity,
)
from usim_feedback_fast3_content_delta import (
    Fast3FeedbackUSIM,
    _resolve_torch_device,
    load_llm_scores_for_stream,
)
from fast3_delta.config import Fast3Config


USIM_PREFIX = "USIM_"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_usim_env(manifest_env):
    for key in list(os.environ):
        if key.startswith(USIM_PREFIX):
            os.environ.pop(key, None)
    for key, value in manifest_env.items():
        if key.startswith(USIM_PREFIX):
            os.environ[key] = str(value)
    os.environ["USIM_STATIC"] = "1"


def _load_split(manifest):
    exports = manifest.get("exports", {})
    required = {
        "train_split": "static_train.pkl",
        "val_split": "static_val.pkl",
        "test_split": "static_test.pkl",
    }
    splits = {}
    for key, fallback in required.items():
        path = exports.get(key)
        if not path:
            out_dir = manifest.get("env", {}).get("USIM_FB_OUTPUT_DIR")
            if out_dir:
                path = str(Path(out_dir) / fallback)
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Missing split artifact for {key}: {path}")
        splits[key] = pd.read_pickle(path)
    return splits["train_split"], splits["val_split"], splits["test_split"]


def _load_checkpoint(path):
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        finished = ckpt_path / "finished.pt"
        latest = ckpt_path / "latest.pt"
        ckpt_path = finished if finished.exists() else latest
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu")
    model_state = state.get("model_state", state)
    return ckpt_path, state, model_state


def profile_one(eval_manifest_path, ckpt_path, seed, dataset, method, source_note):
    manifest_path = Path(eval_manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    _reset_usim_env(manifest.get("env", {}))

    data_dir = os.environ.get("USIM_DATA_DIR", manifest.get("data", {}).get("data_dir", "processed_data_hin"))
    with open(Path(data_dir) / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_pickle(Path(data_dir) / "stream_data.pkl")
    content_emb = torch.load(Path(data_dir) / "content_emb.pt", map_location="cpu")
    llm_scores, llm_path, _ = load_llm_scores_for_stream(
        data_dir,
        df,
        cold_threshold=int(os.environ.get("USIM_COLD_THRESHOLD", "5")),
        n_users=meta.get("n_users"),
        n_items=meta.get("n_items"),
        fallback_data_dirs=["processed_data"],
        verbose=False,
    )

    cfg = Fast3Config(meta["n_users"], meta["n_items"], content_emb.shape[1])
    llm_scores, llm_summary = prepare_llm_scores(llm_scores, cfg)
    cfg.llm_bank_mode = llm_summary["mode"]
    device = _resolve_torch_device()

    train_df, val_df, test_df = _load_split(manifest)
    train_df, val_df, test_df, item_train_pop = apply_train_popularity(train_df, val_df, test_df, cfg)

    artifact_source = manifest.get("split", {}).get("artifact_source")
    if not artifact_source:
        artifact_source = "all_metadata" if cfg.prereq_graph_source == "concept" else "train"
    artifact_df = df if artifact_source == "all_metadata" else train_df
    if cfg.feedback_load_course_artifacts:
        course_artifacts, _ = build_course_artifacts(
            artifact_df,
            cfg.n_items,
            relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
        )
    else:
        course_artifacts, _ = None, _empty_course_stats(cfg.n_items)

    model = Fast3FeedbackUSIM(cfg, content_emb).to(device)
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_train_pop)

    resolved_ckpt_path, ckpt_state, model_state = _load_checkpoint(ckpt_path)
    incompatible = model.load_state_dict(model_state, strict=False)
    model.eval()

    test_history_policy = manifest.get("split", {}).get(
        "test_history_policy",
        os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only"),
    )
    train_seen = add_user_seen_from_df({}, train_df)
    if test_history_policy == "train_val":
        train_seen = add_user_seen_from_df(train_seen, val_df)
    model.set_user_seen_index(train_seen)

    loader = DataLoader(
        StreamDataset(test_df, llm_scores),
        batch_size=2048,
        shuffle=False,
        collate_fn=lambda batch: (
            {"u": torch.stack([item["u"] for item in batch]), "i": torch.stack([item["i"] for item in batch])},
            torch.stack([item["pop"] for item in batch]),
            torch.stack([item["llm"] for item in batch]),
        ),
    )

    k_list = [5, 10, 20]

    with torch.no_grad():
        warm_bank = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
        evaluate_usim(
            model,
            loader,
            device,
            llm_scores,
            k_list=k_list,
            n_neg=cfg.eval_n_neg,
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            all_item_vecs=warm_bank,
            average_mode="item_macro",
        )
        _sync(device)

        t0 = time.perf_counter()
        item_bank = build_eval_item_vecs(model, device, llm_scores, item_batch=1024)
        _sync(device)
        t1 = time.perf_counter()
        cold_metrics, cold_count = evaluate_usim(
            model,
            loader,
            device,
            llm_scores,
            k_list=k_list,
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
            k_list=k_list,
            n_neg=cfg.eval_n_neg,
            eval_type="hot",
            full_ranking=True,
            user_seen_items=train_seen,
            all_item_vecs=item_bank,
            average_mode="item_macro",
        )
        _sync(device)
        t2 = time.perf_counter()

    return {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "source_note": source_note,
        "eval_manifest": str(manifest_path),
        "checkpoint": str(resolved_ckpt_path),
        "checkpoint_status": ckpt_state.get("status") if isinstance(ckpt_state, dict) else None,
        "checkpoint_epoch": ckpt_state.get("next_epoch") if isinstance(ckpt_state, dict) else None,
        "llm_score_path": llm_path,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "precompute_s": float(t1 - t0),
        "ranking_s": float(t2 - t1),
        "final_infer_s": float(t2 - t0),
        "cold_item_count": int(cold_count),
        "hot_item_count": int(hot_count),
        "cold_N@10": None if not cold_metrics else float(cold_metrics.get("N@10", 0.0)),
        "hot_N@10": None if not hot_metrics else float(hot_metrics.get("N@10", 0.0)),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _mean(values):
    return sum(values) / max(1, len(values))


def _std(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="CKG-RL")
    parser.add_argument("--source-note", default="")
    parser.add_argument("--eval-manifest", action="append", required=True)
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument("--seed", action="append", required=True, type=int)
    parser.add_argument("--out-dir", default="outputs/runtime_profile")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if not (len(args.eval_manifest) == len(args.ckpt) == len(args.seed)):
        raise ValueError("--eval-manifest, --ckpt, and --seed must have the same count")

    rows = []
    for manifest, ckpt, seed in zip(args.eval_manifest, args.ckpt, args.seed):
        print(f"[profile] {args.dataset} {args.method} seed={seed}", flush=True)
        row = profile_one(manifest, ckpt, seed, args.dataset, args.method, args.source_note)
        rows.append(row)
        print(
            f"[profile] seed={seed} final={row['final_infer_s']:.3f}s "
            f"pre={row['precompute_s']:.3f}s rank={row['ranking_s']:.3f}s",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    base = f"ckg_final_inference_profile_{args.dataset.lower()}{tag}_{stamp}"
    csv_path = out_dir / f"{base}.csv"
    json_path = out_dir / f"{base}.json"
    summary_path = out_dir / f"{base}_summary.csv"

    fieldnames = [
        "dataset",
        "method",
        "seed",
        "source_note",
        "final_infer_s",
        "precompute_s",
        "ranking_s",
        "cold_item_count",
        "hot_item_count",
        "cold_N@10",
        "hot_N@10",
        "device",
        "checkpoint",
        "eval_manifest",
        "checkpoint_status",
        "checkpoint_epoch",
        "llm_score_path",
        "torch",
        "cuda",
        "missing_keys",
        "unexpected_keys",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["missing_keys"] = ";".join(flat["missing_keys"])
            flat["unexpected_keys"] = ";".join(flat["unexpected_keys"])
            writer.writerow(flat)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    final_values = [row["final_infer_s"] for row in rows]
    pre_values = [row["precompute_s"] for row in rows]
    rank_values = [row["ranking_s"] for row in rows]
    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "n_seeds": len(rows),
        "seeds": ",".join(str(row["seed"]) for row in rows),
        "source_note": args.source_note,
        "final_infer_mean_s": _mean(final_values),
        "final_infer_std_s": _std(final_values),
        "precompute_mean_s": _mean(pre_values),
        "precompute_std_s": _std(pre_values),
        "ranking_mean_s": _mean(rank_values),
        "ranking_std_s": _std(rank_values),
    }
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"[profile] wrote {csv_path}")
    print(f"[profile] wrote {json_path}")
    print(f"[profile] wrote {summary_path}")
    print(
        f"[profile] summary final={summary['final_infer_mean_s']:.3f}"
        f"+/-{summary['final_infer_std_s']:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
