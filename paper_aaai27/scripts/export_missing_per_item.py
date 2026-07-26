from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_aaai27" / "figures"
RUN_ROOT = ROOT / "outputs" / "significance_per_item_exports"
CKPT_ROOT = ROOT / "checkpoints" / "significance_per_item_exports"
AUDIT_CSV = FIG_DIR / "significance_main_missing_inputs.csv"
SUMMARY_CSV = FIG_DIR / "significance_export_queue.csv"
QUEUE_LOG = FIG_DIR / "significance_export_queue.log"
METRIC_TOL = 5e-5


@dataclass(frozen=True)
class JobConfig:
    script: str
    result_file: str
    per_item_file: str
    split_dir: Path
    data_dir: str
    relation_dir: str
    env: dict[str, str]
    ckpt_dir: Path | None = None
    teacher_ckpt_dir: Path | None = None
    needs_checkpoint: bool = False


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def log(message: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def first_candidate(raw: str) -> Path:
    first = raw.split(" | ")[0].strip()
    return ROOT / first


def split_name(seed: int) -> str:
    return f"strict_item_cold_balanced_thr1_seed_{seed}"


def checkpoint_epoch(path: Path | None) -> int | None:
    if path is None:
        return None
    ckpt = path / "latest.pt"
    if not ckpt.exists():
        return None
    code = (
        "import sys, torch; "
        "p=sys.argv[1]; "
        "x=torch.load(p, map_location='cpu', weights_only=False); "
        "print(int(x.get('epoch', 0)))"
    )
    proc = subprocess.run(
        [str(ROOT / "py.bat"), "-c", code, str(ckpt)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        log(f"WARN failed to read checkpoint epoch: {rel(ckpt)} | {proc.stderr.strip()}")
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def metric_block_from_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data[0] if isinstance(data, list) and data else data
    if not isinstance(row, dict):
        return {}
    block = row.get("full_cold_item_macro", {})
    if isinstance(block, dict):
        return {k: float(v) for k, v in block.items() if isinstance(v, (int, float))}
    return {}


def result_metrics(path: Path) -> tuple[float, float]:
    if path.suffix.lower() == ".json":
        block = metric_block_from_json(path)
        return float(block.get("R@10", math.nan)), float(block.get("N@10", math.nan))
    frame = pd.read_csv(path)
    if frame.empty:
        return math.nan, math.nan
    r_cols = ["full_cold_item_macro_r10", "full_cold_item_macro_R10", "R@10"]
    n_cols = ["full_cold_item_macro_n10", "full_cold_item_macro_N10", "N@10"]
    r10 = next((float(frame[c].iloc[0]) for c in r_cols if c in frame.columns), math.nan)
    n10 = next((float(frame[c].iloc[0]) for c in n_cols if c in frame.columns), math.nan)
    return r10, n10


def per_item_metrics(path: Path) -> tuple[int, float, float]:
    frame = pd.read_csv(path)
    if frame.empty:
        return 0, math.nan, math.nan
    return (
        int(len(frame)),
        float(pd.to_numeric(frame["R@10"], errors="coerce").mean()),
        float(pd.to_numeric(frame["N@10"], errors="coerce").mean()),
    )


def same_metrics(primary_result: Path, temp_result: Path, temp_per_item: Path) -> tuple[bool, str]:
    primary_r10, primary_n10 = result_metrics(primary_result)
    temp_r10, temp_n10 = result_metrics(temp_result)
    count, item_r10, item_n10 = per_item_metrics(temp_per_item)
    diffs = {
        "tmp_vs_primary_R@10": abs(temp_r10 - primary_r10),
        "tmp_vs_primary_N@10": abs(temp_n10 - primary_n10),
        "item_vs_primary_R@10": abs(item_r10 - primary_r10),
        "item_vs_primary_N@10": abs(item_n10 - primary_n10),
    }
    ok = all(not math.isnan(v) and v <= METRIC_TOL for v in diffs.values())
    msg = (
        f"primary=({primary_r10:.8f},{primary_n10:.8f}) "
        f"temp=({temp_r10:.8f},{temp_n10:.8f}) "
        f"per_item_mean=({item_r10:.8f},{item_n10:.8f}) count={count} "
        + " ".join(f"{k}={v:.2e}" for k, v in diffs.items())
    )
    return ok, msg


def common_env(dataset: str, seed: int, cfg: JobConfig, out_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "USIM_DATA_DIR": cfg.data_dir,
            "USIM_RELATION_DIR": cfg.relation_dir,
            "USIM_STATIC_SPLIT_DIR": str(cfg.split_dir),
            "USIM_BASELINE_OUTPUT_DIR": str(out_dir),
            "USIM_STATIC_SEED": str(seed),
            "USIM_SEED": str(seed),
            "USIM_COLD_THRESHOLD": "1",
            "USIM_STATIC_TEST_HISTORY": "train_only",
            "USIM_EVAL_N_NEG": "200",
            "USIM_RUN_SAMPLED_EVAL": "0",
            "BASELINE_EARLY_STOP_AVG_MODE": "item_macro",
            "BASELINE_EARLY_STOP_AVERAGE_MODE": "item_macro",
            "USIM_EARLY_STOP_AVG_MODE": "item_macro",
            "BASELINE_BEST_METRIC": "cold",
            "BASELINE_SAVE_CKPT": "1",
            "BASELINE_SAVE_OPT_STATE": "1",
            "BASELINE_AUTO_RESUME": "1",
            "BASELINE_FORCE_FRESH": "0",
        }
    )
    if dataset == "COCO":
        env["USIM_PREREQ_GRAPH_SOURCE"] = "concept"
    env.update(cfg.env)
    if cfg.ckpt_dir is not None:
        env["BASELINE_CKPT_DIR"] = str(cfg.ckpt_dir)
    return env


def junyi_checkpoint(seed: int, name: str) -> Path:
    if seed == 2025:
        return ROOT / "checkpoints" / "junyi" / name / split_name(seed)
    return ROOT / "checkpoints" / "junyi" / "main_table_strictfix" / name / split_name(seed)


def mooccube_split(seed: int) -> Path:
    return ROOT / "outputs" / "content_delta_pop5" / "static_item_cold_balanced" / split_name(seed)


def junyi_split(seed: int) -> Path:
    if seed == 2025:
        return ROOT / "outputs" / "junyi" / "official_prereq_seed2025" / split_name(seed)
    return ROOT / "outputs" / "junyi" / "main_table_3seed" / split_name(seed)


def coco_split(seed: int) -> Path:
    return ROOT / "outputs" / "coco" / "single_seed_triage" / "ours_full" / split_name(seed)


def configure_job(dataset: str, method: str, seed: int, primary_result: Path) -> JobConfig | None:
    if dataset == "Junyi":
        split = junyi_split(seed)
        base = {
            "Popularity": JobConfig(
                "popularity_static.py",
                "popularity_static_result.json",
                "per_item_full_cold_popularity_static.csv",
                split,
                "processed_data_junyi",
                "processed_data_junyi\\relations",
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "POP_DEVICE": "cpu",
                    "POP_BATCH_SIZE": "512",
                    "POP_COLD_THRESHOLD": "1",
                    "POP_EVAL_N_NEG": "200",
                    "POP_STATIC_SEED": str(seed),
                    "POP_SEED": str(seed),
                },
            ),
            "BPR": ("bpr_static_fair.py", "bpr_static_result.json", "per_item_full_cold_bpr_static.csv", "bpr_compare", "BPR", {"BPR_BATCH_SIZE": "4096", "BPR_EVAL_INTERVAL": "10"}),
            "LightGCN": ("lightgcn_static_hin_fair.py", "lightgcn_static_result.json", "per_item_full_cold_lightgcn_static.csv", "lightgcn_compare", "LIGHTGCN", {"LIGHTGCN_BATCH_SIZE": "4096", "LIGHTGCN_EVAL_INTERVAL": "10"}),
            "DropoutNet": ("dropoutnet_official_static_hin.py", "dropoutnet_official_static_result.json", "per_item_full_cold_dropoutnet_official_static.csv", "dropoutnet_compare", "DROPOUT_OFFICIAL", {"DROPOUT_OFFICIAL_BATCH_SIZE": "4096", "DROPOUT_OFFICIAL_EVAL_INTERVAL": "5", "DROPOUT_OFFICIAL_ITEM_DROPOUT": "0.5", "DROPOUT_OFFICIAL_USER_DROPOUT": "0.0"}),
            "CCFCRec": ("ccfc_static_hin.py", "ccfcrec_static_result.json", "per_item_full_cold_ccfcrec_static.csv", "ccfcrec_compare", "CCFCREC", {"CCFCREC_BATCH_SIZE": "4096", "CCFCREC_EVAL_BATCH_SIZE": "4096", "CCFCREC_EVAL_INTERVAL": "5", "CCFCREC_EVAL_ITEM_MODE": "mixed"}),
            "ALDI": ("aldi_static_hin.py", "aldi_static_result.json", "per_item_full_cold_aldi_static.csv", "aldi_compare", "ALDI", {"ALDI_BATCH_SIZE": "4096", "ALDI_EVAL_BATCH_SIZE": "4096", "ALDI_EVAL_INTERVAL": "5", "ALDI_TEACHER_EVAL_INTERVAL": "20"}),
        }
        if method == "Popularity":
            return base[method]
        if method not in base:
            return None
        script, result_file, per_item_file, ckpt_name, prefix, extra = base[method]
        ckpt = junyi_checkpoint(seed, ckpt_name)
        teacher_ckpt = ckpt / "teacher" if method in {"ALDI", "DropoutNet"} else None
        env = dict(extra)
        env.update({f"{prefix}_CKPT_DIR": str(ckpt), f"{prefix}_AUTO_RESUME": "1", f"{prefix}_FORCE_FRESH": "0"})
        epoch = checkpoint_epoch(ckpt)
        if epoch:
            if method == "BPR":
                env["BPR_STATIC_EPOCHS"] = str(epoch)
            elif method == "LightGCN":
                env["LIGHTGCN_STATIC_EPOCHS"] = str(epoch)
            elif method == "DropoutNet":
                env["DROPOUT_OFFICIAL_STATIC_EPOCHS"] = str(epoch)
            elif method == "CCFCRec":
                env["CCFCREC_STATIC_EPOCHS"] = str(epoch)
            elif method == "ALDI":
                env["ALDI_STATIC_EPOCHS"] = str(epoch)
        teacher_epoch = checkpoint_epoch(teacher_ckpt)
        if teacher_epoch and method == "ALDI":
            env["ALDI_TEACHER_EPOCHS"] = str(teacher_epoch)
            env["ALDI_TEACHER_CKPT_DIR"] = str(teacher_ckpt)
        if teacher_epoch and method == "DropoutNet":
            env["DROPOUT_OFFICIAL_TEACHER_EPOCHS"] = str(teacher_epoch)
            env["DROPOUT_OFFICIAL_TEACHER_CKPT_DIR"] = str(teacher_ckpt)
        env[f"{prefix}_STATIC_SEED"] = str(seed)
        env[f"{prefix}_SEED"] = str(seed)
        env[f"{prefix}_COLD_THRESHOLD"] = "1"
        env[f"{prefix}_EVAL_N_NEG"] = "200"
        return JobConfig(script, result_file, per_item_file, split, "processed_data_junyi", "processed_data_junyi\\relations", env, ckpt, teacher_ckpt, True)

    if dataset == "COCO":
        split = coco_split(seed)
        specs = {
            "Popularity": JobConfig(
                "popularity_static.py",
                "popularity_static_result.json",
                "per_item_full_cold_popularity_static.csv",
                split,
                "processed_data_coco",
                "processed_data_coco\\relations",
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "POP_DEVICE": "cpu",
                    "POP_BATCH_SIZE": "512",
                    "POP_COLD_THRESHOLD": "1",
                    "POP_EVAL_N_NEG": "200",
                    "POP_STATIC_SEED": str(seed),
                    "POP_SEED": str(seed),
                },
            ),
            "BPR": ("bpr_static_fair.py", "bpr_static_result.json", "per_item_full_cold_bpr_static.csv", "bpr_lightweight", "BPR", {"BPR_BATCH_SIZE": "4096", "BPR_EVAL_INTERVAL": "5", "BPR_EMB_DIM": "64"}),
            "LightGCN": ("lightgcn_static_hin_fair.py", "lightgcn_static_result.json", "per_item_full_cold_lightgcn_static.csv", "lightgcn_lightweight", "LIGHTGCN", {"LIGHTGCN_BATCH_SIZE": "2048", "LIGHTGCN_EVAL_INTERVAL": "5", "LIGHTGCN_EMB_DIM": "64", "LIGHTGCN_N_LAYERS": "1", "LIGHTGCN_CONTENT_WEIGHT": "0.35"}),
            "DropoutNet": ("drop_static_hin.py", "drop_static_result.json", "per_item_full_cold_drop_static.csv", "dropoutnet_lightweight", "DROPOUT", {"DROPOUT_BATCH_SIZE": "512", "DROPOUT_EVAL_INTERVAL": "5"}),
            "ALDI": ("aldi_static_hin.py", "aldi_static_result.json", "per_item_full_cold_aldi_static.csv", "aldi_lightweight", "ALDI", {"ALDI_BATCH_SIZE": "1024", "ALDI_EVAL_BATCH_SIZE": "1024", "ALDI_EVAL_INTERVAL": "5", "ALDI_EMB_DIM": "64", "ALDI_HIDDEN_DIM": "64", "ALDI_TEACHER_EVAL_INTERVAL": "5"}),
        }
        if method == "Popularity":
            return specs[method]
        if method not in specs:
            return None
        script, result_file, per_item_file, ckpt_name, prefix, extra = specs[method]
        ckpt = ROOT / "checkpoints" / "coco" / "single_seed_triage" / ckpt_name / split_name(seed)
        teacher_ckpt = ckpt / "teacher" if method == "ALDI" else None
        env = dict(extra)
        env.update({f"{prefix}_CKPT_DIR": str(ckpt), f"{prefix}_AUTO_RESUME": "1", f"{prefix}_FORCE_FRESH": "0"})
        epoch = checkpoint_epoch(ckpt)
        if epoch:
            if method == "BPR":
                env["BPR_STATIC_EPOCHS"] = str(epoch)
            elif method == "LightGCN":
                env["LIGHTGCN_STATIC_EPOCHS"] = str(epoch)
            elif method == "DropoutNet":
                env["DROPOUT_STATIC_EPOCHS"] = str(epoch)
            elif method == "ALDI":
                env["ALDI_STATIC_EPOCHS"] = str(epoch)
        teacher_epoch = checkpoint_epoch(teacher_ckpt)
        if teacher_epoch and method == "ALDI":
            env["ALDI_TEACHER_EPOCHS"] = str(teacher_epoch)
            env["ALDI_TEACHER_CKPT_DIR"] = str(teacher_ckpt)
        env[f"{prefix}_STATIC_SEED"] = str(seed)
        env[f"{prefix}_SEED"] = str(seed)
        env[f"{prefix}_COLD_THRESHOLD"] = "1"
        env[f"{prefix}_EVAL_N_NEG"] = "200"
        return JobConfig(script, result_file, per_item_file, split, "processed_data_coco", "processed_data_coco\\relations", env, ckpt, teacher_ckpt, True)

    if dataset == "MOOCCube":
        split = mooccube_split(seed)
        if method == "Popularity":
            return JobConfig(
                "popularity_static.py",
                "popularity_static_result.json",
                "per_item_full_cold_popularity_static.csv",
                split,
                "processed_data_hin_clean_pop5",
                "processed_data_hin_clean_pop5\\relations",
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "POP_DEVICE": "cpu",
                    "POP_BATCH_SIZE": "512",
                    "POP_COLD_THRESHOLD": "1",
                    "POP_EVAL_N_NEG": "200",
                    "POP_STATIC_SEED": str(seed),
                    "POP_SEED": str(seed),
                },
            )
        retrain_root = CKPT_ROOT / "mooccube" / method.lower() / split_name(seed)
        primary_name = primary_result.name
        if method == "BPR":
            return JobConfig("bpr_static_fair.py", primary_name, "per_item_full_cold_bpr_static.csv", split, "processed_data_hin_clean_pop5", "processed_data_hin_clean_pop5\\relations", {"BPR_STATIC_EPOCHS": "200", "BPR_EVAL_INTERVAL": "5", "BPR_BATCH_SIZE": "4096", "BPR_CKPT_DIR": str(retrain_root), "BPR_STATIC_SEED": str(seed), "BPR_SEED": str(seed), "BPR_COLD_THRESHOLD": "1", "BPR_EVAL_N_NEG": "200"}, retrain_root, None, True)
        if method == "LightGCN":
            return JobConfig("lightgcn_static_hin_fair.py", primary_name, "per_item_full_cold_lightgcn_static.csv", split, "processed_data_hin_clean_pop5", "processed_data_hin_clean_pop5\\relations", {"LIGHTGCN_STATIC_EPOCHS": "100", "LIGHTGCN_EVAL_INTERVAL": "5", "LIGHTGCN_BATCH_SIZE": "4096", "LIGHTGCN_CKPT_DIR": str(retrain_root), "LIGHTGCN_STATIC_SEED": str(seed), "LIGHTGCN_SEED": str(seed), "LIGHTGCN_COLD_THRESHOLD": "1", "LIGHTGCN_EVAL_N_NEG": "200"}, retrain_root, None, True)
        if method == "DropoutNet":
            return JobConfig("dropoutnet_official_static_hin.py", primary_name, "per_item_full_cold_dropoutnet_official_static.csv", split, "processed_data_hin_clean_pop5", "processed_data_hin_clean_pop5\\relations", {"DROPOUT_OFFICIAL_TEACHER_EPOCHS": "80", "DROPOUT_OFFICIAL_STATIC_EPOCHS": "120", "DROPOUT_OFFICIAL_EVAL_INTERVAL": "5", "DROPOUT_OFFICIAL_BATCH_SIZE": "4096", "DROPOUT_OFFICIAL_CKPT_DIR": str(retrain_root), "DROPOUT_OFFICIAL_TEACHER_CKPT_DIR": str(retrain_root / "teacher"), "DROPOUT_OFFICIAL_STATIC_SEED": str(seed), "DROPOUT_OFFICIAL_SEED": str(seed), "DROPOUT_OFFICIAL_COLD_THRESHOLD": "1", "DROPOUT_OFFICIAL_EVAL_N_NEG": "200"}, retrain_root, retrain_root / "teacher", True)
        if method == "CCFCRec":
            return JobConfig("ccfc_static_hin.py", primary_name, "per_item_full_cold_ccfcrec_static.csv", split, "processed_data_hin_clean_pop5", "processed_data_hin_clean_pop5\\relations", {"CCFCREC_STATIC_EPOCHS": "80", "CCFCREC_EVAL_INTERVAL": "5", "CCFCREC_BATCH_SIZE": "4096", "CCFCREC_EVAL_BATCH_SIZE": "4096", "CCFCREC_EVAL_ITEM_MODE": "mixed", "CCFCREC_CKPT_DIR": str(retrain_root), "CCFCREC_STATIC_SEED": str(seed), "CCFCREC_SEED": str(seed), "CCFCREC_COLD_THRESHOLD": "1", "CCFCREC_EVAL_N_NEG": "200"}, retrain_root, None, True)
        if method == "ALDI":
            return JobConfig("aldi_official_static_hin.py", primary_name, "per_item_full_cold_aldi_official_static.csv", split, "processed_data_hin_clean_pop5", "processed_data_hin_clean_pop5\\relations", {"ALDI_OFFICIAL_STATIC_EPOCHS": "100", "ALDI_OFFICIAL_STATIC_SEED": str(seed), "ALDI_OFFICIAL_SEED": str(seed), "ALDI_OFFICIAL_CKPT_DIR": str(retrain_root), "ALDI_OFFICIAL_AUTO_RESUME": "1", "ALDI_OFFICIAL_FORCE_FRESH": "0"}, retrain_root, None, True)
    return None


def run_job(row: dict[str, str], allow_retrain: bool, dry_run: bool) -> dict[str, str]:
    dataset = row["dataset"]
    method = row["method"]
    seed = int(row["seed"])
    primary_result = Path(row["result_source"])
    if not primary_result.is_absolute():
        primary_result = ROOT / primary_result
    target_per_item = first_candidate(row["per_item_candidates"])
    cfg = configure_job(dataset, method, seed, primary_result)
    status = {
        "dataset": dataset,
        "method": method,
        "seed": str(seed),
        "target_per_item": rel(target_per_item),
        "status": "pending",
        "message": "",
    }
    if cfg is None:
        status["status"] = "skipped"
        status["message"] = "no runner mapping"
        return status
    if cfg.needs_checkpoint and cfg.ckpt_dir is not None and not (cfg.ckpt_dir / "latest.pt").exists():
        if not allow_retrain:
            status["status"] = "skipped"
            status["message"] = f"checkpoint missing: {rel(cfg.ckpt_dir)}"
            return status
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if cfg.teacher_ckpt_dir is not None:
            cfg.teacher_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RUN_ROOT / dataset.lower() / method.lower().replace("-", "_") / split_name(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = common_env(dataset, seed, cfg, out_dir)
    temp_result = out_dir / cfg.result_file
    temp_per_item = out_dir / cfg.per_item_file
    log_path = out_dir / "run.log"
    cmd = [str(ROOT / "py.bat"), "-u", cfg.script]
    log(f"START {dataset}/{method}/seed{seed} -> {rel(out_dir)}")
    if dry_run:
        status["status"] = "dry_run"
        status["message"] = " ".join(cmd)
        return status
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        status["status"] = "failed"
        status["message"] = f"exit={proc.returncode}; log={rel(log_path)}"
        log(f"FAIL {dataset}/{method}/seed{seed} | {status['message']}")
        return status
    if not temp_result.exists() or not temp_per_item.exists():
        status["status"] = "failed"
        status["message"] = f"missing temp artifacts; result={temp_result.exists()} per_item={temp_per_item.exists()} log={rel(log_path)}"
        log(f"FAIL {dataset}/{method}/seed{seed} | {status['message']}")
        return status
    ok, message = same_metrics(primary_result, temp_result, temp_per_item)
    if not ok:
        status["status"] = "mismatch"
        status["message"] = message + f"; log={rel(log_path)}"
        log(f"MISMATCH {dataset}/{method}/seed{seed} | {message}")
        return status
    target_per_item.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temp_per_item, target_per_item)
    status["status"] = "copied"
    status["message"] = message
    log(f"DONE {dataset}/{method}/seed{seed} | copied={rel(target_per_item)} | {message}")
    return status


def read_rows(datasets: set[str], methods: set[str], seeds: set[int], include_ready_alt: bool) -> list[dict[str, str]]:
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Missing audit CSV: {AUDIT_CSV}")
    rows: list[dict[str, str]] = []
    with AUDIT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status"] != "missing_per_item" and not (include_ready_alt and row["status"] == "ready_alt"):
                continue
            if datasets and row["dataset"] not in datasets:
                continue
            if methods and row["method"] not in methods:
                continue
            if seeds and int(row["seed"]) not in seeds:
                continue
            rows.append(row)
    return rows


def parse_csv_set(values: Iterable[str], cast=str) -> set:
    out = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.add(cast(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["Junyi", "COCO", "MOOCCube"])
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", default=[])
    parser.add_argument("--allow-retrain", action="store_true")
    parser.add_argument("--include-ready-alt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    datasets = parse_csv_set(args.datasets)
    methods = parse_csv_set(args.methods)
    seeds = parse_csv_set(args.seeds, int)
    rows = read_rows(datasets, methods, seeds, args.include_ready_alt)
    if args.limit > 0:
        rows = rows[: args.limit]
    log(
        "QUEUE START "
        f"jobs={len(rows)} datasets={sorted(datasets)} methods={sorted(methods) or 'all'} "
        f"seeds={sorted(seeds) or 'all'} allow_retrain={args.allow_retrain} dry_run={args.dry_run}"
    )
    results = []
    for row in rows:
        results.append(run_job(row, allow_retrain=args.allow_retrain, dry_run=args.dry_run))
        pd.DataFrame(results).to_csv(SUMMARY_CSV, index=False)
    log(f"QUEUE END jobs={len(rows)} summary={rel(SUMMARY_CSV)}")
    if results:
        print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
