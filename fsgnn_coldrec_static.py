"""Official ColdRec FS-GNN adapter for the shared static item-cold protocol.

This file deliberately keeps the FS-GNN implementation inside the ColdRec
checkout unchanged. It only:
1. Converts the project static split into ColdRec's data layout.
2. Calls ColdRec's own Config -> model_factory -> model.run() path.
3. Maps ColdRec internal embeddings back to project ids for the paper evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hin_data_common import (
    InteractionDataset,
    add_user_seen_from_df,
    build_user_seen,
    clone_user_seen,
    collate_interactions,
    load_hin_processed,
    setup_seed,
)
from hin_eval_common import evaluate_embedding_ranker, print_final_report


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_COLDREC_ROOT = REPO_ROOT / ".runtime_tmp" / "ColdRec"
METRICS = [f"{metric}@{k}" for metric in ("R", "N") for k in (5, 10, 20)]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_first(names: Sequence[str], default: str = "") -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _result_path(default_filename: str) -> str:
    output_dir = _env_first(
        [
            "FSGNN_BASELINE_OUTPUT_DIR",
            "USIM_BASELINE_OUTPUT_DIR",
            "FSGNN_STATIC_SPLIT_DIR",
            "USIM_STATIC_SPLIT_DIR",
        ]
    )
    if not output_dir:
        return default_filename
    os.makedirs(output_dir, exist_ok=True)
    return str(Path(output_dir) / default_filename)


def _safe_dataset_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "mooccube_static"


def load_split_cold_threshold(split_dir: str, fallback: int) -> int:
    if split_dir:
        summary_path = Path(split_dir) / "static_split_summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if "cold_threshold" in summary:
                return int(summary["cold_threshold"])

        counts_path = Path(split_dir) / "static_split_counts.csv"
        if counts_path.exists():
            counts = pd.read_csv(counts_path)
            if "cold_threshold" in counts.columns and len(counts) > 0:
                values = counts["cold_threshold"].dropna().unique()
                if len(values) > 0:
                    return int(values[0])
    return int(fallback)


def load_static_split(
    df: pd.DataFrame,
    split_dir: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_dir:
        split_path = Path(split_dir)
        paths = [
            split_path / "static_train.pkl",
            split_path / "static_val.pkl",
            split_path / "static_test.pkl",
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Static split directory is set but split files are missing: "
                + ", ".join(missing)
            )
        train_df = pd.read_pickle(paths[0]).copy()
        val_df = pd.read_pickle(paths[1]).copy()
        test_df = pd.read_pickle(paths[2]).copy()
        print(
            f"Loaded shared static split from {split_path}: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
            flush=True,
        )
        return train_df, val_df, test_df

    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Invalid split ratio, require 0 < train,val and train+val < 1")
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"Split too small: total={n_total}, train={n_train}, val={n_val}, test={n_test}"
        )
    return (
        df_shuffled.iloc[:n_train].copy(),
        df_shuffled.iloc[n_train:n_train + n_val].copy(),
        df_shuffled.iloc[n_train + n_val:].copy(),
    )


def _pairs(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": df["u_idx"].astype(np.int64).to_numpy(),
            "item": df["i_idx"].astype(np.int64).to_numpy(),
        }
    )


def _unique_int_array(values: Iterable[int]) -> np.ndarray:
    return np.asarray(sorted({int(v) for v in values}), dtype=np.int32)


def _write_pairs(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _pairs(df).to_csv(path, index=False)


def export_coldrec_dataset(
    *,
    coldrec_root: Path,
    dataset_name: str,
    meta: Dict,
    content_emb: torch.Tensor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_threshold: int,
    source_data_dir: str,
    split_dir: str,
) -> Path:
    """Write the current split in the exact file layout ColdRec main.py loads."""
    dataset_name = _safe_dataset_name(dataset_name)
    dataset_dir = Path(coldrec_root) / "data" / dataset_name
    cold_dir = dataset_dir / "cold_item"
    cold_dir.mkdir(parents=True, exist_ok=True)

    val_cold = val_df.loc[val_df["popularity"] < cold_threshold].copy()
    val_warm = val_df.loc[val_df["popularity"] >= cold_threshold].copy()
    test_cold = test_df.loc[test_df["popularity"] < cold_threshold].copy()
    test_warm = test_df.loc[test_df["popularity"] >= cold_threshold].copy()

    _write_pairs(cold_dir / "warm_train.csv", train_df)
    _write_pairs(cold_dir / "warm_val.csv", val_warm)
    _write_pairs(cold_dir / "warm_test.csv", test_warm)
    _write_pairs(cold_dir / "cold_item_val.csv", val_cold)
    _write_pairs(cold_dir / "cold_item_test.csv", test_cold)
    _write_pairs(cold_dir / "overall_val.csv", val_df)
    _write_pairs(cold_dir / "overall_test.csv", test_df)

    warm_user = _unique_int_array(train_df["u_idx"].tolist())
    warm_item = _unique_int_array(train_df["i_idx"].tolist())
    cold_item = _unique_int_array(
        list(val_cold["i_idx"].astype(int).tolist()) + list(test_cold["i_idx"].astype(int).tolist())
    )
    info = {
        "user_num": int(meta["n_users"]),
        "item_num": int(meta["n_items"]),
        "warm_user": warm_user,
        "warm_item": warm_item,
        "cold_user": np.empty(0, dtype=np.int32),
        "cold_item": cold_item,
    }
    with (cold_dir / "info_dict.pkl").open("wb") as f:
        pickle.dump(info, f, protocol=4)

    content_np = content_emb.detach().cpu().float().numpy()
    expected_items = int(meta["n_items"])
    if content_np.shape[0] != expected_items:
        raise ValueError(
            f"content_emb rows ({content_np.shape[0]}) must equal meta n_items ({expected_items})"
        )
    np.save(dataset_dir / f"{dataset_name}_item_content.npy", content_np)

    manifest = {
        "dataset": dataset_name,
        "source_data_dir": source_data_dir,
        "split_dir": split_dir,
        "cold_threshold": int(cold_threshold),
        "train_rows": int(len(train_df)),
        "val_cold_rows": int(len(val_cold)),
        "val_warm_rows": int(len(val_warm)),
        "test_cold_rows": int(len(test_cold)),
        "test_warm_rows": int(len(test_warm)),
        "warm_user_count": int(len(warm_user)),
        "warm_item_count": int(len(warm_item)),
        "cold_item_count": int(len(cold_item)),
        "note": "Adapter-generated ColdRec view; FS-GNN source code is not modified.",
    }
    with (dataset_dir / "static_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return dataset_dir


def restore_original_order_embeddings(
    *,
    mapped_user_emb: torch.Tensor,
    mapped_item_emb: torch.Tensor,
    id2user: Dict[int, int],
    id2item: Dict[int, int],
    n_users: int,
    n_items: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ColdRec's contiguous internal rows back to project id rows."""
    user_emb = torch.zeros((int(n_users), mapped_user_emb.shape[1]), dtype=mapped_user_emb.detach().cpu().dtype)
    item_emb = torch.zeros((int(n_items), mapped_item_emb.shape[1]), dtype=mapped_item_emb.detach().cpu().dtype)
    mapped_user_cpu = mapped_user_emb.detach().cpu()
    mapped_item_cpu = mapped_item_emb.detach().cpu()

    for mapped_id, source_id in id2user.items():
        mapped_id = int(mapped_id)
        source_id = int(source_id)
        if 0 <= mapped_id < mapped_user_cpu.shape[0] and 0 <= source_id < n_users:
            user_emb[source_id] = mapped_user_cpu[mapped_id]
    for mapped_id, source_id in id2item.items():
        mapped_id = int(mapped_id)
        source_id = int(source_id)
        if 0 <= mapped_id < mapped_item_cpu.shape[0] and 0 <= source_id < n_items:
            item_emb[source_id] = mapped_item_cpu[mapped_id]
    return user_emb, item_emb


def train_cold_item_ids(train_df: pd.DataFrame, n_items: int) -> torch.Tensor:
    warm = {int(x) for x in train_df["i_idx"].astype(int).tolist()}
    cold = [item_id for item_id in range(int(n_items)) if item_id not in warm]
    return torch.as_tensor(cold, dtype=torch.long)


def apply_cold_item_score_bias(
    scores: torch.Tensor,
    cold_item_ids: torch.Tensor,
    bias: float,
    cand_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if float(bias) == 0.0 or cold_item_ids.numel() == 0:
        return scores
    adjusted = scores.clone()
    cold_ids = cold_item_ids.to(device=scores.device, dtype=torch.long)
    cold_ids = cold_ids[cold_ids >= 0]
    if cold_ids.numel() == 0:
        return adjusted

    if cand_idx is None:
        cold_ids = cold_ids[cold_ids < adjusted.size(1)]
        if cold_ids.numel() > 0:
            adjusted[:, cold_ids] = adjusted[:, cold_ids] + float(bias)
        return adjusted

    cand = cand_idx.to(device=scores.device, dtype=torch.long)
    max_id = int(max(cold_ids.max().item(), cand.max().item())) if cand.numel() else int(cold_ids.max().item())
    lookup = torch.zeros(max_id + 1, dtype=torch.bool, device=scores.device)
    lookup[cold_ids[cold_ids <= max_id]] = True
    valid = cand >= 0
    mask = torch.zeros_like(cand, dtype=torch.bool)
    mask[valid] = lookup[cand[valid]]
    adjusted[mask] = adjusted[mask] + float(bias)
    return adjusted


def _cold_bias_adjust_fn(cold_item_ids: torch.Tensor, bias: float):
    if float(bias) == 0.0:
        return None

    def _adjust(scores, uid_t, i, pop_sel, cand_idx):
        del uid_t, i, pop_sel
        return apply_cold_item_score_bias(scores, cold_item_ids, bias, cand_idx)

    return _adjust


def _parse_float_grid(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("FSGNN_COLD_BIAS_GRID must contain at least one float")
    return values


@contextmanager
def _coldrec_runtime(coldrec_root: Path):
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    root_str = str(coldrec_root.resolve())
    sys.path.insert(0, root_str)
    os.chdir(root_str)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        try:
            sys.path.remove(root_str)
        except ValueError:
            pass


def _load_coldrec_main(coldrec_root: Path):
    main_path = coldrec_root / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing ColdRec main.py: {main_path}")
    module_name = "coldrec_official_main_for_fsgnn"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ColdRec main.py from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_value(cwd: Path, args: Sequence[str]) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return res.stdout.strip()


def _coldrec_git_info(coldrec_root: Path) -> Dict[str, object]:
    tracked_status = _git_value(coldrec_root, ["status", "--short", "--untracked-files=no"])
    runtime_status = _git_value(coldrec_root, ["status", "--short"])
    return {
        "official_repo": _git_value(coldrec_root, ["remote", "get-url", "origin"])
        or "https://github.com/YuanchenBei/ColdRec",
        "official_commit": _git_value(coldrec_root, ["rev-parse", "--short", "HEAD"]),
        "official_tree_clean": tracked_status == "",
        "official_status_short": tracked_status or "",
        "official_runtime_status_short": runtime_status or "",
    }


def _build_coldrec_argv(dataset_name: str, cfg) -> list[str]:
    argv = [
        "main.py",
        "--dataset",
        dataset_name,
        "--model",
        "FSGNN",
        "--cold_object",
        "item",
        "--epochs",
        str(cfg.epochs),
        "--topN",
        cfg.topn,
        "--bs",
        str(cfg.batch_size),
        "--emb_size",
        str(cfg.emb_size),
        "--lr",
        str(cfg.lr),
        "--reg",
        str(cfg.reg),
        "--runs",
        "1",
        "--seed",
        str(cfg.seed),
        "--use_gpu",
        "true" if cfg.use_gpu else "false",
        "--save_emb",
        "true",
        "--gpu_id",
        str(cfg.gpu_id),
        "--early_stop",
        str(cfg.early_stop),
        "--eval_every",
        str(cfg.eval_every),
        "--result_file",
        str((cfg.output_dir / "coldrec_native_fsgnn_result.txt").resolve()) if cfg.output_dir else "",
        "--result_overwrite",
    ]
    extra = shlex.split(cfg.extra_args) if cfg.extra_args else []
    return [arg for arg in argv if arg != ""] + extra


def run_coldrec_fsgnn(coldrec_root: Path, dataset_name: str, cfg):
    """Execute ColdRec source in-process so id2user/id2item stay accessible."""
    (coldrec_root / "emb").mkdir(parents=True, exist_ok=True)
    with _coldrec_runtime(coldrec_root):
        coldrec_main = _load_coldrec_main(coldrec_root)
        sys.argv = _build_coldrec_argv(dataset_name, cfg)
        args = coldrec_main.parse_args()
        config = coldrec_main.Config(args)
        coldrec_main.set_seed(args.seed, args.use_gpu)
        model = coldrec_main.model_factory(config)
        print(
            "Executing ColdRec official source: "
            f"{coldrec_root / 'main.py'} -> model.run() | dataset={dataset_name}",
            flush=True,
        )
        model.run()
        return model, args, config


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        split_dir = _env_first(["FSGNN_STATIC_SPLIT_DIR", "USIM_STATIC_SPLIT_DIR"], args.split_dir)
        fallback_threshold = _env_int("FSGNN_COLD_THRESHOLD", _env_int("USIM_COLD_THRESHOLD", args.cold_threshold))
        self.data_dir = _env_first(["FSGNN_DATA_DIR", "USIM_DATA_DIR"], args.data_dir)
        self.split_dir = split_dir
        self.cold_threshold = load_split_cold_threshold(split_dir, fallback_threshold)
        self.seed = _env_int("FSGNN_SEED", _env_int("USIM_STATIC_SEED", args.seed))
        self.static_seed = _env_int("FSGNN_STATIC_SEED", self.seed)
        self.train_ratio = _env_float("FSGNN_STATIC_TRAIN_RATIO", args.train_ratio)
        self.val_ratio = _env_float("FSGNN_STATIC_VAL_RATIO", args.val_ratio)
        self.coldrec_root = Path(_env_first(["FSGNN_COLDREC_ROOT"], args.coldrec_root)).resolve()
        self.dataset_name = _safe_dataset_name(
            _env_first(["FSGNN_COLDREC_DATASET"], args.dataset_name)
            or f"fsgnn_{Path(self.data_dir).name}_{Path(split_dir).name if split_dir else 'seed_' + str(self.static_seed)}"
        )
        output_dir_raw = _env_first(["FSGNN_BASELINE_OUTPUT_DIR", "USIM_BASELINE_OUTPUT_DIR"], args.output_dir)
        self.output_dir = Path(output_dir_raw).resolve() if output_dir_raw else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = _env_int("FSGNN_EPOCHS", args.epochs)
        if self.epochs < 1:
            raise ValueError("FSGNN_EPOCHS must be >= 1 because ColdRec FS-GNN saves best embeddings after validation")
        self.emb_size = _env_int("FSGNN_EMB_SIZE", args.emb_size)
        self.batch_size = _env_int("FSGNN_BATCH_SIZE", args.batch_size)
        self.lr = _env_float("FSGNN_LR", args.lr)
        self.reg = _env_float("FSGNN_REG", args.reg)
        self.topn = os.environ.get("FSGNN_TOPN", args.topn).strip()
        self.use_gpu = _env_bool("FSGNN_USE_GPU", args.use_gpu)
        self.gpu_id = _env_int("FSGNN_GPU_ID", args.gpu_id)
        self.early_stop = _env_int("FSGNN_EARLY_STOP", args.early_stop)
        self.eval_every = _env_int("FSGNN_EVAL_EVERY", args.eval_every)
        self.extra_args = os.environ.get("FSGNN_COLDREC_EXTRA_ARGS", args.extra_args).strip()
        self.eval_batch_size = _env_int("FSGNN_EVAL_BATCH_SIZE", args.eval_batch_size)
        self.eval_n_neg = _env_int("FSGNN_EVAL_N_NEG", _env_int("USIM_EVAL_N_NEG", args.eval_n_neg))
        self.run_sampled_eval = _env_bool("FSGNN_RUN_SAMPLED_EVAL", args.run_sampled_eval)
        self.cold_score_bias = _env_float("FSGNN_COLD_SCORE_BIAS", args.cold_score_bias)
        self.auto_calibrate_cold_bias = _env_bool(
            "FSGNN_AUTO_CALIBRATE_COLD_BIAS",
            args.auto_calibrate_cold_bias,
        )
        self.cold_bias_grid = _parse_float_grid(
            os.environ.get("FSGNN_COLD_BIAS_GRID", args.cold_bias_grid)
        )
        self.test_history_policy = os.environ.get("FSGNN_STATIC_TEST_HISTORY", os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only")).strip().lower()
        if self.test_history_policy not in {"train_only", "train_val"}:
            raise ValueError("FSGNN_STATIC_TEST_HISTORY/USIM_STATIC_TEST_HISTORY must be train_only or train_val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    parser.add_argument("--split-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--coldrec-root", default=str(DEFAULT_COLDREC_ROOT))
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reg", type=float, default=0.0005)
    parser.add_argument("--topn", default="5,10,20")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-use-gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-n-neg", type=int, default=200)
    parser.add_argument("--run-sampled-eval", action="store_true", default=False)
    parser.add_argument("--cold-score-bias", type=float, default=0.0)
    parser.add_argument("--auto-calibrate-cold-bias", action="store_true", default=False)
    parser.add_argument("--cold-bias-grid", default="0,5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--extra-args", default="")
    return parser.parse_args()


def tune_cold_score_bias(
    *,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    cfg: Config,
    n_items: int,
    device: torch.device,
    cold_item_ids: torch.Tensor,
) -> Tuple[float, Dict[str, float]]:
    val_loader = DataLoader(
        InteractionDataset(val_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    train_seen = build_user_seen(train_df)
    user_emb_dev = user_emb.to(device)
    item_emb_dev = item_emb.to(device)
    get_user_fn = lambda batch: user_emb_dev[batch["u"]]

    best_bias = float(cfg.cold_score_bias)
    best_metrics: Dict[str, float] = {}
    best_score = -float("inf")
    for bias in cfg.cold_bias_grid:
        metrics, n_eval = evaluate_embedding_ranker(
            val_loader,
            device,
            n_items,
            cfg.cold_threshold,
            get_user_fn,
            item_emb_dev,
            k_list=(5, 10, 20),
            n_neg=cfg.eval_n_neg,
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            normalize_user=False,
            average_mode="item_macro",
            score_adjust_fn=_cold_bias_adjust_fn(cold_item_ids, float(bias)),
        )
        metrics = metrics or {}
        score = float(metrics.get("N@10", 0.0))
        if n_eval > 0 and score > best_score:
            best_score = score
            best_bias = float(bias)
            best_metrics = metrics
    return best_bias, best_metrics


def _evaluate(
    *,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Config,
    n_items: int,
    device: torch.device,
    cold_item_ids: torch.Tensor,
    cold_score_bias: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], int, int, int, int, int, int]:
    test_loader = DataLoader(
        InteractionDataset(test_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    user_emb_dev = user_emb.to(device)
    item_emb_dev = item_emb.to(device)
    get_user_fn = lambda batch: user_emb_dev[batch["u"]]

    train_seen = build_user_seen(train_df)
    test_seen = clone_user_seen(train_seen)
    if cfg.test_history_policy == "train_val":
        add_user_seen_from_df(test_seen, val_df)
    score_adjust_fn = _cold_bias_adjust_fn(cold_item_ids, cold_score_bias)

    sample_cold: Dict[str, float] = {}
    sample_hot: Dict[str, float] = {}
    n_sc = 0
    n_sh = 0
    if cfg.run_sampled_eval:
        sample_cold_raw, n_sc = evaluate_embedding_ranker(
            test_loader,
            device,
            n_items,
            cfg.cold_threshold,
            get_user_fn,
            item_emb_dev,
            k_list=(5, 10, 20),
            n_neg=cfg.eval_n_neg,
            eval_type="cold",
            full_ranking=False,
            user_seen_items=test_seen,
            normalize_user=False,
            score_adjust_fn=score_adjust_fn,
        )
        sample_hot_raw, n_sh = evaluate_embedding_ranker(
            test_loader,
            device,
            n_items,
            cfg.cold_threshold,
            get_user_fn,
            item_emb_dev,
            k_list=(5, 10, 20),
            n_neg=cfg.eval_n_neg,
            eval_type="hot",
            full_ranking=False,
            user_seen_items=test_seen,
            normalize_user=False,
            score_adjust_fn=score_adjust_fn,
        )
        sample_cold = sample_cold_raw or {}
        sample_hot = sample_hot_raw or {}

    full_cold, n_fc = evaluate_embedding_ranker(
        test_loader,
        device,
        n_items,
        cfg.cold_threshold,
        get_user_fn,
        item_emb_dev,
        k_list=(5, 10, 20),
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=test_seen,
        normalize_user=False,
        average_mode="interaction",
        score_adjust_fn=score_adjust_fn,
    )
    full_hot, n_fh = evaluate_embedding_ranker(
        test_loader,
        device,
        n_items,
        cfg.cold_threshold,
        get_user_fn,
        item_emb_dev,
        k_list=(5, 10, 20),
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        full_ranking=True,
        user_seen_items=test_seen,
        normalize_user=False,
        average_mode="interaction",
        score_adjust_fn=score_adjust_fn,
    )
    full_cold_item, n_fc_item = evaluate_embedding_ranker(
        test_loader,
        device,
        n_items,
        cfg.cold_threshold,
        get_user_fn,
        item_emb_dev,
        k_list=(5, 10, 20),
        n_neg=cfg.eval_n_neg,
        eval_type="cold",
        full_ranking=True,
        user_seen_items=test_seen,
        normalize_user=False,
        average_mode="item_macro",
        score_adjust_fn=score_adjust_fn,
        export_item_metrics_path=_result_path("per_item_full_cold_fsgnn_coldrec_static.csv"),
    )
    full_hot_item, n_fh_item = evaluate_embedding_ranker(
        test_loader,
        device,
        n_items,
        cfg.cold_threshold,
        get_user_fn,
        item_emb_dev,
        k_list=(5, 10, 20),
        n_neg=cfg.eval_n_neg,
        eval_type="hot",
        full_ranking=True,
        user_seen_items=test_seen,
        normalize_user=False,
        average_mode="item_macro",
        score_adjust_fn=score_adjust_fn,
        export_item_metrics_path=_result_path("per_item_full_hot_fsgnn_coldrec_static.csv"),
    )
    return (
        sample_cold,
        sample_hot,
        full_cold or {},
        full_hot or {},
        full_cold_item or {},
        full_hot_item or {},
        n_sc,
        n_sh,
        n_fc,
        n_fh,
        n_fc_item,
        n_fh_item,
    )


def main() -> None:
    cfg = Config(parse_args())
    if not cfg.coldrec_root.exists():
        raise FileNotFoundError(f"Missing ColdRec source checkout: {cfg.coldrec_root}")

    setup_seed(cfg.seed)
    print(f"Loading data from {cfg.data_dir} ...", flush=True)
    meta, df, content_emb = load_hin_processed(cfg.data_dir)
    train_df, val_df, test_df = load_static_split(
        df,
        cfg.split_dir,
        seed=cfg.static_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    dataset_dir = export_coldrec_dataset(
        coldrec_root=cfg.coldrec_root,
        dataset_name=cfg.dataset_name,
        meta=meta,
        content_emb=content_emb,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cold_threshold=cfg.cold_threshold,
        source_data_dir=cfg.data_dir,
        split_dir=cfg.split_dir,
    )
    print(f"Wrote ColdRec dataset view to {dataset_dir}", flush=True)

    model, coldrec_args, _ = run_coldrec_fsgnn(cfg.coldrec_root, cfg.dataset_name, cfg)
    user_emb, item_emb = restore_original_order_embeddings(
        mapped_user_emb=model.user_emb,
        mapped_item_emb=model.item_emb,
        id2user=model.data.id2user,
        id2item=model.data.id2item,
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
    )

    checkpoint_path = _result_path("fsgnn_coldrec_mapped_embeddings.pt")
    torch.save(
        {
            "user_emb": user_emb,
            "item_emb": item_emb,
            "id2user": dict(model.data.id2user),
            "id2item": dict(model.data.id2item),
            "coldrec_args": vars(coldrec_args),
            "dataset": cfg.dataset_name,
            "source": "ColdRec official source embeddings mapped back to project ids.",
        },
        checkpoint_path,
    )
    print(f"Saved mapped embedding checkpoint: {checkpoint_path}", flush=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu) else "cpu")
    cold_item_ids = train_cold_item_ids(train_df, int(meta["n_items"]))
    selected_cold_bias = float(cfg.cold_score_bias)
    calibration_val_metrics: Dict[str, float] = {}
    if cfg.auto_calibrate_cold_bias:
        selected_cold_bias, calibration_val_metrics = tune_cold_score_bias(
            user_emb=user_emb,
            item_emb=item_emb,
            val_df=val_df,
            train_df=train_df,
            cfg=cfg,
            n_items=int(meta["n_items"]),
            device=device,
            cold_item_ids=cold_item_ids,
        )
        print(
            f"Cold score calibration: selected_bias={selected_cold_bias:.4f} "
            f"val_cold_item_macro_N@10={calibration_val_metrics.get('N@10', 0.0):.4f}",
            flush=True,
        )
    elif selected_cold_bias != 0.0:
        print(f"Cold score calibration: fixed_bias={selected_cold_bias:.4f}", flush=True)

    (
        sample_cold,
        sample_hot,
        full_cold,
        full_hot,
        full_cold_item,
        full_hot_item,
        n_sc,
        n_sh,
        n_fc,
        n_fh,
        n_fc_item,
        n_fh_item,
    ) = _evaluate(
        user_emb=user_emb,
        item_emb=item_emb,
        test_df=test_df,
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        n_items=int(meta["n_items"]),
        device=device,
        cold_item_ids=cold_item_ids,
        cold_score_bias=selected_cold_bias,
    )

    if cfg.run_sampled_eval:
        print_final_report(
            eval_n_neg=cfg.eval_n_neg,
            metrics_keys=METRICS,
            sample_cold=sample_cold,
            sample_hot=sample_hot,
            full_cold=full_cold,
            full_hot=full_hot,
            count_sample_cold=n_sc,
            count_sample_hot=n_sh,
            count_full_cold=n_fc,
            count_full_hot=n_fh,
            title="ColdRec FS-GNN Static HIN",
        )
    else:
        print("\n" + "=" * 76)
        print("         FINAL REPORT: full ranking only (ColdRec FS-GNN Static HIN)")
        print("=" * 76)
        print(f"{'Metric':<10} | {'Full Cold':<12} | {'Full Hot':<12}")
        print("-" * 76)
        for metric in METRICS:
            print(f"{metric:<10} | {full_cold.get(metric, 0.0):<12.4f} | {full_hot.get(metric, 0.0):<12.4f}")
        print("-" * 76)
        print(f"Full samples/items: Cold={n_fc}, Hot={n_fh}")
        print("=" * 76)

    git_info = _coldrec_git_info(cfg.coldrec_root)
    best_epoch = None
    best_native = {}
    if getattr(model, "bestPerformance", None):
        best_epoch = int(model.bestPerformance[0])
        best_native = dict(model.bestPerformance[1])
    official_user_emb = cfg.coldrec_root / "emb" / f"{cfg.dataset_name}_cold_item_FSGNN_user_emb.pt"
    official_item_emb = cfg.coldrec_root / "emb" / f"{cfg.dataset_name}_cold_item_FSGNN_item_emb.pt"

    result = {
        "model": "FSGNN-coldrec-official",
        "model_display": "FS-GNN",
        "source": "ColdRec official source executed in-process via main.py Config/model_factory/model.run; adapter only converts data and evaluates mapped embeddings.",
        "official_source_dir": str(cfg.coldrec_root),
        "official_code": str(cfg.coldrec_root),
        **git_info,
        "paper": "Feature-Structure Adaptive Completion Graph Neural Network for Cold-Start Recommendation",
        "paper_venue": "AAAI 2025",
        "protocol": "static_item_cold",
        "score_function": "raw_dot_product" if selected_cold_bias == 0.0 else "raw_dot_product_plus_train_cold_bias",
        "coldrec_dataset": cfg.dataset_name,
        "coldrec_dataset_dir": str(dataset_dir),
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item,
        "full_hot_item_macro": full_hot_item,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "count_full_cold_item_macro": n_fc_item,
        "count_full_hot_item_macro": n_fh_item,
        "best_epoch": best_epoch,
        "best_metric": "ColdRec native overall validation NDCG at max(topN)",
        "coldrec_native_best_performance": best_native,
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "seed": cfg.seed,
        "cold_threshold": cfg.cold_threshold,
        "emb_dim": cfg.emb_size,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "reg": cfg.reg,
        "topN": cfg.topn,
        "extra_args": cfg.extra_args,
        "test_history_policy": cfg.test_history_policy,
        "cold_score_bias": selected_cold_bias,
        "auto_calibrate_cold_bias": bool(cfg.auto_calibrate_cold_bias),
        "cold_bias_grid": cfg.cold_bias_grid,
        "calibration_val_cold_item_macro": calibration_val_metrics,
        "train_cold_candidate_count": int(cold_item_ids.numel()),
        "mapped_embedding_checkpoint": checkpoint_path,
        "official_user_embedding_checkpoint": str(official_user_emb),
        "official_item_embedding_checkpoint": str(official_item_emb),
        "official_embedding_checkpoints_present": bool(official_user_emb.exists() and official_item_emb.exists()),
        "note": "ColdRec native cold metrics mask warm items; main-table metrics above are recomputed with the paper full-catalog evaluator.",
    }
    result_path = _result_path("fsgnn_coldrec_static_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
