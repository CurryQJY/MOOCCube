"""Strict static item-cold adapter for the released ColdRec GAR model.

ColdRec's GAR learner, objective, optimizer, and pairwise sampler remain in the
source checkout. This module owns only data conversion, validation selection,
and external protocol-controlled evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fsgnn_coldrec_static import (
    DEFAULT_COLDREC_ROOT,
    _coldrec_git_info,
    _coldrec_runtime,
    _env_bool,
    _env_first,
    _env_float,
    _env_int,
    _load_coldrec_main,
    _safe_dataset_name,
    export_coldrec_dataset,
    load_split_cold_threshold,
    load_static_split,
    restore_original_order_embeddings,
)
from hin_data_common import (
    InteractionDataset,
    add_user_seen_from_df,
    build_user_seen,
    clone_user_seen,
    collate_interactions,
    load_hin_processed,
    setup_seed,
)
from hin_eval_common import evaluate_embedding_ranker


@dataclass
class Config:
    data_dir: str
    split_dir: str
    cold_threshold: int
    seed: int
    static_seed: int
    coldrec_root: Path
    dataset_name: str
    output_dir: Path
    epochs: int
    emb_size: int
    batch_size: int
    lr: float
    reg: float
    topn: str
    use_gpu: bool
    gpu_id: int
    early_stop: int
    eval_every: int
    extra_args: str
    eval_batch_size: int
    eval_n_neg: int
    run_sampled_eval: bool
    test_history_policy: str
    backbone: str
    alpha: float
    beta: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        split_dir = _env_first(["GAR_COLDREC_STATIC_SPLIT_DIR"], args.split_dir)
        fallback_threshold = _env_int("GAR_COLDREC_COLD_THRESHOLD", args.cold_threshold)
        seed = _env_int("GAR_COLDREC_SEED", args.seed)
        static_seed = _env_int("GAR_COLDREC_STATIC_SEED", seed)
        data_dir = _env_first(["GAR_COLDREC_DATA_DIR"], args.data_dir)
        coldrec_root = Path(
            _env_first(["GAR_COLDREC_ROOT"], args.coldrec_root)
        ).resolve()
        dataset_name = _safe_dataset_name(
            _env_first(["GAR_COLDREC_DATASET"], args.dataset_name)
            or f"gar_{Path(data_dir).name}_seed{static_seed}"
        )
        output_dir = Path(
            _env_first(["GAR_COLDREC_OUTPUT_DIR"], args.output_dir)
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        history = os.environ.get(
            "GAR_COLDREC_TEST_HISTORY", args.test_history
        ).strip().lower()
        if history not in {"train_only", "train_val"}:
            raise ValueError("GAR_COLDREC_TEST_HISTORY must be train_only or train_val")
        epochs = _env_int("GAR_COLDREC_EPOCHS", args.epochs)
        if epochs < 1:
            raise ValueError("GAR_COLDREC_EPOCHS must be >= 1")
        return cls(
            data_dir=data_dir,
            split_dir=split_dir,
            cold_threshold=load_split_cold_threshold(split_dir, fallback_threshold),
            seed=seed,
            static_seed=static_seed,
            coldrec_root=coldrec_root,
            dataset_name=dataset_name,
            output_dir=output_dir,
            epochs=epochs,
            emb_size=_env_int("GAR_COLDREC_EMB_SIZE", args.emb_size),
            batch_size=_env_int("GAR_COLDREC_BATCH_SIZE", args.batch_size),
            lr=_env_float("GAR_COLDREC_LR", args.lr),
            reg=_env_float("GAR_COLDREC_REG", args.reg),
            topn=os.environ.get("GAR_COLDREC_TOPN", args.topn).strip(),
            use_gpu=_env_bool("GAR_COLDREC_USE_GPU", args.use_gpu),
            gpu_id=_env_int("GAR_COLDREC_GPU_ID", args.gpu_id),
            early_stop=_env_int("GAR_COLDREC_EARLY_STOP", args.early_stop),
            eval_every=_env_int("GAR_COLDREC_EVAL_EVERY", args.eval_every),
            extra_args=os.environ.get(
                "GAR_COLDREC_EXTRA_ARGS", args.extra_args
            ).strip(),
            eval_batch_size=_env_int(
                "GAR_COLDREC_EVAL_BATCH_SIZE", args.eval_batch_size
            ),
            eval_n_neg=_env_int("GAR_COLDREC_EVAL_N_NEG", args.eval_n_neg),
            run_sampled_eval=_env_bool(
                "GAR_COLDREC_RUN_SAMPLED_EVAL", args.run_sampled_eval
            ),
            test_history_policy=history,
            backbone=os.environ.get(
                "GAR_COLDREC_BACKBONE", args.backbone
            ).strip(),
            alpha=_env_float("GAR_COLDREC_ALPHA", args.alpha),
            beta=_env_float("GAR_COLDREC_BETA", args.beta),
        )


def _build_coldrec_argv(dataset_name: str, cfg: Config) -> list[str]:
    argv = [
        "main.py",
        "--dataset",
        dataset_name,
        "--model",
        "GAR",
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
        "--backbone",
        cfg.backbone,
        "--alpha",
        str(cfg.alpha),
        "--beta",
        str(cfg.beta),
        "--result_file",
        str((cfg.output_dir / "coldrec_native_gar_result.txt").resolve()),
        "--result_overwrite",
    ]
    return argv + (shlex.split(cfg.extra_args) if cfg.extra_args else [])


def require_mf_embeddings(
    coldrec_root: Path,
    dataset_name: str,
    backbone: str = "MF",
) -> Tuple[Path, Path]:
    emb_dir = Path(coldrec_root) / "emb"
    user_path = emb_dir / f"{dataset_name}_cold_item_{backbone}_user_emb.pt"
    item_path = emb_dir / f"{dataset_name}_cold_item_{backbone}_item_emb.pt"
    missing = [str(path) for path in (user_path, item_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing matching ColdRec MF embeddings: " + ", ".join(missing))
    return user_path, item_path


def assert_strict_cold_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_threshold: int,
) -> Dict[str, int]:
    train_items = {int(item) for item in train_df["i_idx"].tolist()}
    heldout = pd.concat([val_df, test_df], ignore_index=True)
    heldout_cold = {
        int(item)
        for item in heldout.loc[
            heldout["popularity"] < int(cold_threshold), "i_idx"
        ].tolist()
    }
    overlap = train_items.intersection(heldout_cold)
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(
            "Strict split violation: held-out cold items appear in train: "
            + ", ".join(str(item) for item in sample)
        )
    return {
        "heldout_cold_item_count": len(heldout_cold),
        "train_overlap_count": len(overlap),
    }


def bind_strict_validation_callback(
    trainer,
    evaluate_fn: Callable[[object], Tuple[Optional[Dict[str, float]], int]],
):
    """Select GAR embeddings by full-catalog cold item-macro validation N@10."""

    trainer.strict_validation_history = []

    def strict_fast_evaluation(self, epoch: int, valid_type: str = "all"):
        del valid_type
        metrics, item_count = evaluate_fn(self)
        metrics = dict(metrics or {})
        score = float(metrics.get("N@10", float("nan")))
        finite = item_count > 0 and math.isfinite(score)
        improved = False
        if finite:
            if not self.bestPerformance or score > float(self.bestPerformance[1]["NDCG"]):
                performance = {
                    "Hit Ratio": float(metrics.get("R@10", 0.0)),
                    "Precision": float(metrics.get("R@10", 0.0)),
                    "Recall": float(metrics.get("R@10", 0.0)),
                    "NDCG": score,
                    **metrics,
                }
                self.bestPerformance = [int(epoch) + 1, performance]
                self.save()
                improved = True

        if getattr(self, "early_stop_flag", False):
            if improved:
                self.early_stop_patience = self.max_early_stop_patience
            else:
                self.early_stop_patience -= 1

        record = {
            "epoch": int(epoch) + 1,
            "item_count": int(item_count),
            "improved": improved,
            **metrics,
        }
        self.strict_validation_history.append(record)
        print(
            "Strict validation: "
            f"epoch={epoch + 1} cold_items={item_count} "
            f"N@10={score:.6f} improved={improved}",
            flush=True,
        )
        return metrics

    trainer.fast_evaluation = MethodType(strict_fast_evaluation, trainer)
    return trainer


def make_strict_validation_evaluator(
    *,
    val_df: pd.DataFrame,
    train_seen: Dict[int, set],
    cfg: Config,
    n_users: int,
    n_items: int,
):
    loader = DataLoader(
        InteractionDataset(val_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )

    def evaluate(trainer):
        device = getattr(
            trainer,
            "device",
            torch.device(
                f"cuda:{cfg.gpu_id}"
                if cfg.use_gpu and torch.cuda.is_available()
                else "cpu"
            ),
        )
        user_emb, item_emb = restore_original_order_embeddings(
            mapped_user_emb=trainer.user_emb,
            mapped_item_emb=trainer.item_emb,
            id2user=trainer.data.id2user,
            id2item=trainer.data.id2item,
            n_users=n_users,
            n_items=n_items,
        )
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)
        metrics, item_count = evaluate_embedding_ranker(
            loader,
            device,
            n_items,
            cfg.cold_threshold,
            lambda batch: user_emb[batch["u"]],
            item_emb,
            k_list=(5, 10, 20),
            n_neg=cfg.eval_n_neg,
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            normalize_user=False,
            average_mode="item_macro",
        )
        return metrics or {}, item_count

    return evaluate


def build_result_payload(
    *,
    cfg: Config,
    git_info: Dict[str, object],
    strict_audit: Dict[str, int],
    best_epoch: Optional[int],
    best_val_n10: Optional[float],
    full_cold: Dict[str, float],
    full_hot: Dict[str, float],
    full_cold_item: Dict[str, float],
    full_hot_item: Dict[str, float],
    counts: Dict[str, int],
    device: str,
    elapsed_seconds: float,
    per_item_cold_path: Path,
    per_item_hot_path: Path,
) -> Dict[str, object]:
    return {
        "model": "GAR-coldrec-source-strict",
        "model_display": "GAR (ColdRec source, strict adapter)",
        "paper": "Generative Adversarial Framework for Cold-Start Item Recommendation",
        "paper_venue": "SIGIR 2022",
        "source": (
            "ColdRec GAR source instantiated through Config/model_factory; "
            "adapter changes data loading, validation selection, and final evaluation only."
        ),
        **git_info,
        "source_model_file": str(cfg.coldrec_root / "model" / "GAR.py"),
        "source_model_unchanged": True,
        "source_training_method": "ColdRec model.GAR.GAR.train",
        "protocol": "static_item_cold_balanced",
        "candidate_mode": "full_catalog",
        "checkpoint_metric": "validation_full_cold_item_macro.N@10",
        "item_macro_metrics": True,
        "train_history_masking": True,
        "train_only_interaction_evidence": True,
        "test_history_policy": cfg.test_history_policy,
        "strict_split_dir": cfg.split_dir,
        "source_data_dir": cfg.data_dir,
        "strict_audit": strict_audit,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item,
        "full_hot_item_macro": full_hot_item,
        "counts": counts,
        "best_epoch": best_epoch,
        "best_val_full_cold_item_macro_n10": best_val_n10,
        "seed": cfg.seed,
        "static_seed": cfg.static_seed,
        "cold_threshold": cfg.cold_threshold,
        "epochs": cfg.epochs,
        "emb_size": cfg.emb_size,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "reg": cfg.reg,
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "backbone": cfg.backbone,
        "device": device,
        "cuda_used": device.startswith("cuda"),
        "elapsed_seconds": float(elapsed_seconds),
        "per_item_full_cold_path": str(per_item_cold_path),
        "per_item_full_hot_path": str(per_item_hot_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="processed_data_hin_clean_pop5")
    parser.add_argument(
        "--split-dir",
        default=(
            "outputs/content_delta_pop5/static_item_cold_balanced/"
            "strict_item_cold_balanced_thr1_seed_2025"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "paper_aaai27/baseline_sources/_gar_coldrec_strict/"
            "mooccube_seed2025_single"
        ),
    )
    parser.add_argument("--coldrec-root", default=str(DEFAULT_COLDREC_ROOT))
    parser.add_argument("--dataset-name", default="gar_mooccube_seed2025")
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--topn", default="5,10,20")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-use-gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-n-neg", type=int, default=200)
    parser.add_argument("--run-sampled-eval", action="store_true", default=False)
    parser.add_argument("--test-history", default="train_only")
    parser.add_argument("--backbone", default="MF")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _evaluate_final(
    *,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Config,
    n_items: int,
    device: torch.device,
):
    loader = DataLoader(
        InteractionDataset(test_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    get_user = lambda batch: user_emb[batch["u"]]
    test_seen = clone_user_seen(build_user_seen(train_df))
    if cfg.test_history_policy == "train_val":
        add_user_seen_from_df(test_seen, val_df)

    def evaluate(eval_type: str, average_mode: str, export_path: Optional[Path] = None):
        result, count = evaluate_embedding_ranker(
            loader,
            device,
            n_items,
            cfg.cold_threshold,
            get_user,
            item_emb,
            k_list=(5, 10, 20),
            n_neg=cfg.eval_n_neg,
            eval_type=eval_type,
            full_ranking=True,
            user_seen_items=test_seen,
            normalize_user=False,
            average_mode=average_mode,
            export_item_metrics_path=str(export_path) if export_path else None,
        )
        return result or {}, count

    cold_path = cfg.output_dir / "per_item_full_cold_gar_coldrec.csv"
    hot_path = cfg.output_dir / "per_item_full_hot_gar_coldrec.csv"
    full_cold, n_full_cold = evaluate("cold", "interaction")
    full_hot, n_full_hot = evaluate("hot", "interaction")
    full_cold_item, n_full_cold_item = evaluate("cold", "item_macro", cold_path)
    full_hot_item, n_full_hot_item = evaluate("hot", "item_macro", hot_path)
    return (
        full_cold,
        full_hot,
        full_cold_item,
        full_hot_item,
        {
            "full_cold": n_full_cold,
            "full_hot": n_full_hot,
            "full_cold_item": n_full_cold_item,
            "full_hot_item": n_full_hot_item,
        },
        cold_path,
        hot_path,
    )


def run_coldrec_gar(
    *,
    cfg: Config,
    val_df: pd.DataFrame,
    train_seen: Dict[int, set],
    n_users: int,
    n_items: int,
):
    require_mf_embeddings(cfg.coldrec_root, cfg.dataset_name, cfg.backbone)
    (cfg.coldrec_root / "emb").mkdir(parents=True, exist_ok=True)
    with _coldrec_runtime(cfg.coldrec_root):
        coldrec_main = _load_coldrec_main(cfg.coldrec_root)
        sys.argv = _build_coldrec_argv(cfg.dataset_name, cfg)
        coldrec_args = coldrec_main.parse_args()
        coldrec_config = coldrec_main.Config(coldrec_args)
        if cfg.use_gpu and coldrec_config.device.type != "cuda":
            raise RuntimeError("GAR GPU run requested but ColdRec resolved a non-CUDA device")
        coldrec_main.set_seed(coldrec_args.seed, coldrec_args.use_gpu)
        model = coldrec_main.model_factory(coldrec_config)
        evaluate = make_strict_validation_evaluator(
            val_df=val_df,
            train_seen=train_seen,
            cfg=cfg,
            n_users=n_users,
            n_items=n_items,
        )
        bind_strict_validation_callback(model, evaluate)
        print(
            "Executing released ColdRec GAR.train() with strict validation callback | "
            f"dataset={cfg.dataset_name} device={coldrec_config.device}",
            flush=True,
        )
        model.train()
        return model, coldrec_args, coldrec_config


def main() -> None:
    args = parse_args()
    cfg = Config.from_args(args)
    if not cfg.coldrec_root.exists():
        raise FileNotFoundError(f"Missing ColdRec checkout: {cfg.coldrec_root}")
    setup_seed(cfg.seed)
    meta, df, content_emb = load_hin_processed(cfg.data_dir)
    train_df, val_df, test_df = load_static_split(
        df,
        cfg.split_dir,
        seed=cfg.static_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    strict_audit = assert_strict_cold_disjoint(
        train_df, val_df, test_df, cfg.cold_threshold
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
    generated_manifest = dataset_dir / "static_manifest.json"
    with generated_manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.update(
        {
            "model": "GAR",
            "candidate_mode": "full_catalog",
            "checkpoint_metric": "validation_full_cold_item_macro.N@10",
            "train_history_masking": True,
            "train_only_interaction_evidence": True,
            "strict_audit": strict_audit,
            "note": "Adapter-generated ColdRec view; GAR source model is unchanged.",
        }
    )
    with generated_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    shutil.copy2(generated_manifest, cfg.output_dir / "coldrec_static_manifest.json")
    split_manifest = Path(cfg.split_dir) / "static_protocol_manifest.json"
    if split_manifest.exists():
        shutil.copy2(split_manifest, cfg.output_dir / "static_protocol_manifest.json")
    if args.prepare_only:
        print(f"Prepared ColdRec GAR dataset: {dataset_dir}", flush=True)
        return

    mf_user_path, mf_item_path = require_mf_embeddings(
        cfg.coldrec_root, cfg.dataset_name, cfg.backbone
    )
    train_seen = build_user_seen(train_df)
    started = time.perf_counter()
    model, coldrec_args, coldrec_config = run_coldrec_gar(
        cfg=cfg,
        val_df=val_df,
        train_seen=train_seen,
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
    )
    elapsed = time.perf_counter() - started
    user_emb, item_emb = restore_original_order_embeddings(
        mapped_user_emb=model.user_emb,
        mapped_item_emb=model.item_emb,
        id2user=model.data.id2user,
        id2item=model.data.id2item,
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
    )
    (
        full_cold,
        full_hot,
        full_cold_item,
        full_hot_item,
        counts,
        cold_path,
        hot_path,
    ) = _evaluate_final(
        user_emb=user_emb,
        item_emb=item_emb,
        test_df=test_df,
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        n_items=int(meta["n_items"]),
        device=coldrec_config.device,
    )
    if counts["full_cold_item"] < 1:
        raise RuntimeError("GAR strict evaluation produced no cold course-macro items")
    best_epoch = int(model.bestPerformance[0]) if model.bestPerformance else None
    best_val_n10 = (
        float(model.bestPerformance[1]["NDCG"]) if model.bestPerformance else None
    )
    result = build_result_payload(
        cfg=cfg,
        git_info=_coldrec_git_info(cfg.coldrec_root),
        strict_audit=strict_audit,
        best_epoch=best_epoch,
        best_val_n10=best_val_n10,
        full_cold=full_cold,
        full_hot=full_hot,
        full_cold_item=full_cold_item,
        full_hot_item=full_hot_item,
        counts=counts,
        device=str(coldrec_config.device),
        elapsed_seconds=elapsed,
        per_item_cold_path=cold_path,
        per_item_hot_path=hot_path,
    )
    result.update(
        {
            "coldrec_dataset": cfg.dataset_name,
            "coldrec_dataset_dir": str(dataset_dir),
            "mf_user_embedding": str(mf_user_path),
            "mf_item_embedding": str(mf_item_path),
            "coldrec_args": {
                key: value
                for key, value in vars(coldrec_args).items()
                if isinstance(value, (str, int, float, bool)) or value is None
            },
            "strict_validation_history": list(model.strict_validation_history),
            "epochs_ran": int(getattr(model, "epochs_ran", 0)),
        }
    )
    result_path = cfg.output_dir / "gar_coldrec_strict_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    report_path = cfg.output_dir / "gar_coldrec_strict_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# GAR ColdRec Strict Single-Seed Report\n\n")
        f.write(f"- Seed: `{cfg.seed}`\n")
        f.write(f"- Device: `{coldrec_config.device}`\n")
        f.write(f"- Best epoch: `{best_epoch}`\n")
        f.write(f"- Validation cold item-macro N@10: `{best_val_n10}`\n")
        f.write(f"- Strict train overlap: `{strict_audit['train_overlap_count']}`\n")
        f.write(f"- Test cold courses: `{counts['full_cold_item']}`\n\n")
        f.write("## Full-Catalog Cold Course-Macro\n\n")
        for key in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20"):
            f.write(f"- {key}: `{full_cold_item.get(key, 0.0):.6f}`\n")
        f.write("\nSource model and loss are unchanged; only data, validation, and evaluation are adapted.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {result_path}", flush=True)
    print(f"Saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
