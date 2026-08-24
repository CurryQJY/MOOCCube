"""Official ColdRec M2VAE adapter for the shared static item-cold protocol.

The M2VAE implementation remains inside the ColdRec checkout unchanged. This
adapter only converts the project split to ColdRec's data layout, executes
ColdRec's Config -> model_factory -> model.run() path, and evaluates the mapped
embeddings with the paper's full-catalog evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fsgnn_coldrec_static import (
    DEFAULT_COLDREC_ROOT,
    METRICS,
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
from hin_eval_common import evaluate_embedding_ranker, print_final_report


def _result_path(default_filename: str) -> str:
    output_dir = _env_first(["M2VAE_BASELINE_OUTPUT_DIR", "M2VAE_STATIC_SPLIT_DIR"])
    if not output_dir:
        return default_filename
    os.makedirs(output_dir, exist_ok=True)
    return str(Path(output_dir) / default_filename)


def _cfg_result_path(cfg: "Config", default_filename: str) -> str:
    if cfg.output_dir is None:
        return default_filename
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return str(cfg.output_dir / default_filename)


@dataclass
class Config:
    data_dir: str
    split_dir: str
    cold_threshold: int
    seed: int
    static_seed: int
    coldrec_root: Path
    dataset_name: str
    output_dir: Optional[Path]
    epochs: int
    emb_size: int
    batch_size: int
    lr: float
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
    positive_number: int
    negative_number: int
    self_neg_number: int
    attr_present_dim: int
    implicit_dim: int
    cat_implicit_dim: int
    tau: float
    weight_decay: float
    kld_weight: float
    recon_weight: float
    decouple_weight: float
    pretrain: bool
    pretrain_update: bool
    attr_mask_neg1: bool
    backbone: str = "MF"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        split_dir = _env_first(["M2VAE_STATIC_SPLIT_DIR"], args.split_dir)
        fallback_threshold = _env_int("M2VAE_COLD_THRESHOLD", args.cold_threshold)
        data_dir = _env_first(["M2VAE_DATA_DIR"], args.data_dir)
        seed = _env_int("M2VAE_SEED", args.seed)
        static_seed = _env_int("M2VAE_STATIC_SEED", seed)
        coldrec_root = Path(_env_first(["M2VAE_COLDREC_ROOT"], args.coldrec_root)).resolve()
        dataset_name = _safe_dataset_name(
            _env_first(["M2VAE_COLDREC_DATASET"], args.dataset_name)
            or f"m2vae_{Path(data_dir).name}_{Path(split_dir).name if split_dir else 'seed_' + str(static_seed)}"
        )
        output_dir_raw = _env_first(["M2VAE_BASELINE_OUTPUT_DIR"], args.output_dir)
        output_dir = Path(output_dir_raw).resolve() if output_dir_raw else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        test_history_policy = os.environ.get("M2VAE_STATIC_TEST_HISTORY", args.test_history).strip().lower()
        if test_history_policy not in {"train_only", "train_val"}:
            raise ValueError("M2VAE_STATIC_TEST_HISTORY must be train_only or train_val")
        epochs = _env_int("M2VAE_EPOCHS", args.epochs)
        if epochs < 1:
            raise ValueError("M2VAE_EPOCHS must be >= 1 because ColdRec saves best embeddings after validation")

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
            emb_size=_env_int("M2VAE_EMB_SIZE", args.emb_size),
            batch_size=_env_int("M2VAE_BATCH_SIZE", args.batch_size),
            lr=_env_float("M2VAE_LR", args.lr),
            topn=os.environ.get("M2VAE_TOPN", args.topn).strip(),
            use_gpu=_env_bool("M2VAE_USE_GPU", args.use_gpu),
            gpu_id=_env_int("M2VAE_GPU_ID", args.gpu_id),
            early_stop=_env_int("M2VAE_EARLY_STOP", args.early_stop),
            eval_every=_env_int("M2VAE_EVAL_EVERY", args.eval_every),
            extra_args=os.environ.get("M2VAE_COLDREC_EXTRA_ARGS", args.extra_args).strip(),
            eval_batch_size=_env_int("M2VAE_EVAL_BATCH_SIZE", args.eval_batch_size),
            eval_n_neg=_env_int("M2VAE_EVAL_N_NEG", args.eval_n_neg),
            run_sampled_eval=_env_bool("M2VAE_RUN_SAMPLED_EVAL", args.run_sampled_eval),
            test_history_policy=test_history_policy,
            positive_number=_env_int("M2VAE_POSITIVE_NUMBER", args.positive_number),
            negative_number=_env_int("M2VAE_NEGATIVE_NUMBER", args.negative_number),
            self_neg_number=_env_int("M2VAE_SELF_NEG_NUMBER", args.self_neg_number),
            attr_present_dim=_env_int("M2VAE_ATTR_PRESENT_DIM", args.attr_present_dim),
            implicit_dim=_env_int("M2VAE_IMPLICIT_DIM", args.implicit_dim),
            cat_implicit_dim=_env_int("M2VAE_CAT_IMPLICIT_DIM", args.cat_implicit_dim),
            tau=_env_float("M2VAE_TAU", args.tau),
            weight_decay=_env_float("M2VAE_WEIGHT_DECAY", args.weight_decay),
            kld_weight=_env_float("M2VAE_KLD_WEIGHT", args.kld_weight),
            recon_weight=_env_float("M2VAE_RECON_WEIGHT", args.recon_weight),
            decouple_weight=_env_float("M2VAE_DECOUPLE_WEIGHT", args.decouple_weight),
            pretrain=_env_bool("M2VAE_PRETRAIN", args.pretrain),
            pretrain_update=_env_bool("M2VAE_PRETRAIN_UPDATE", args.pretrain_update),
            attr_mask_neg1=_env_bool("M2VAE_ATTR_MASK_NEG1", args.attr_mask_neg1),
            backbone=os.environ.get("M2VAE_BACKBONE", args.backbone).strip(),
        )


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--topn", default="5,10,20")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-use-gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-n-neg", type=int, default=200)
    parser.add_argument("--run-sampled-eval", action="store_true", default=False)
    parser.add_argument("--test-history", default="train_only")
    parser.add_argument("--positive-number", type=int, default=10)
    parser.add_argument("--negative-number", type=int, default=40)
    parser.add_argument("--self-neg-number", type=int, default=40)
    parser.add_argument("--attr-present-dim", type=int, default=64)
    parser.add_argument("--implicit-dim", type=int, default=64)
    parser.add_argument("--cat-implicit-dim", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--kld-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--decouple-weight", type=float, default=100.0)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain-update", action="store_true", default=False)
    parser.add_argument("--attr-mask-neg1", action="store_true", default=False)
    parser.add_argument("--backbone", default="MF")
    parser.add_argument("--extra-args", default="")
    return parser.parse_args()


def _build_coldrec_argv(dataset_name: str, cfg: Config) -> list[str]:
    argv = [
        "main.py",
        "--dataset",
        dataset_name,
        "--model",
        "M2VAE",
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
        "--result_file",
        str((cfg.output_dir / "coldrec_native_m2vae_result.txt").resolve()) if cfg.output_dir else "",
        "--result_overwrite",
        "--positive_number",
        str(cfg.positive_number),
        "--negative_number",
        str(cfg.negative_number),
        "--self_neg_number",
        str(cfg.self_neg_number),
        "--attr_present_dim",
        str(cfg.attr_present_dim),
        "--implicit_dim",
        str(cfg.implicit_dim),
        "--cat_implicit_dim",
        str(cfg.cat_implicit_dim),
        "--tau",
        str(cfg.tau),
        "--m2vae_weight_decay",
        str(cfg.weight_decay),
        "--m2vae_kld_weight",
        str(cfg.kld_weight),
        "--m2vae_recon_weight",
        str(cfg.recon_weight),
        "--m2vae_decouple_weight",
        str(cfg.decouple_weight),
    ]
    if cfg.pretrain:
        argv.append("--m2vae_pretrain")
    if cfg.pretrain_update:
        argv.append("--m2vae_pretrain_update")
    if cfg.attr_mask_neg1:
        argv.append("--m2vae_attr_mask_neg1")
    extra = shlex.split(cfg.extra_args) if cfg.extra_args else []
    return [arg for arg in argv if arg != ""] + extra


def run_coldrec_m2vae(coldrec_root: Path, dataset_name: str, cfg: Config):
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
            f"{coldrec_root / 'main.py'} -> model.run() | model=M2VAE | dataset={dataset_name}",
            flush=True,
        )
        model.run()
        return model, args, config


def train_cold_item_ids(train_df: pd.DataFrame, n_items: int) -> torch.Tensor:
    warm = {int(x) for x in train_df["i_idx"].astype(int).tolist()}
    cold = [item_id for item_id in range(int(n_items)) if item_id not in warm]
    return torch.as_tensor(cold, dtype=torch.long)


def export_m2vae_coldrec_dataset(
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
    dataset_dir = export_coldrec_dataset(
        coldrec_root=coldrec_root,
        dataset_name=dataset_name,
        meta=meta,
        content_emb=content_emb,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cold_threshold=cold_threshold,
        source_data_dir=source_data_dir,
        split_dir=split_dir,
    )
    manifest_path = dataset_dir / "static_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["note"] = "Adapter-generated ColdRec view; M2VAE source code is not modified."
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return dataset_dir


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
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    int,
    int,
    int,
    int,
    int,
    int,
]:
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
        export_item_metrics_path=_cfg_result_path(cfg, "per_item_full_cold_m2vae_coldrec_static.csv"),
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
        export_item_metrics_path=_cfg_result_path(cfg, "per_item_full_hot_m2vae_coldrec_static.csv"),
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
    args = parse_args()
    cfg = Config.from_args(args)
    if not cfg.coldrec_root.exists():
        raise FileNotFoundError(f"Missing ColdRec source checkout: {cfg.coldrec_root}")

    setup_seed(cfg.seed)
    print(f"Loading data from {cfg.data_dir} ...", flush=True)
    meta, df, content_emb = load_hin_processed(cfg.data_dir)
    train_df, val_df, test_df = load_static_split(
        df,
        cfg.split_dir,
        seed=cfg.static_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    dataset_dir = export_m2vae_coldrec_dataset(
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

    model, coldrec_args, _ = run_coldrec_m2vae(cfg.coldrec_root, cfg.dataset_name, cfg)
    user_emb, item_emb = restore_original_order_embeddings(
        mapped_user_emb=model.user_emb,
        mapped_item_emb=model.item_emb,
        id2user=model.data.id2user,
        id2item=model.data.id2item,
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
    )

    checkpoint_path = _cfg_result_path(cfg, "m2vae_coldrec_mapped_embeddings.pt")
    torch.save(
        {
            "user_emb": user_emb,
            "item_emb": item_emb,
            "id2user": dict(model.data.id2user),
            "id2item": dict(model.data.id2item),
            "coldrec_args": vars(coldrec_args),
            "dataset": cfg.dataset_name,
            "source": "ColdRec official M2VAE source embeddings mapped back to project ids.",
        },
        checkpoint_path,
    )
    print(f"Saved mapped embedding checkpoint: {checkpoint_path}", flush=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu) else "cpu")
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
            title="ColdRec M2VAE Static HIN",
        )
    else:
        print("\n" + "=" * 76)
        print("         FINAL REPORT: full ranking only (ColdRec M2VAE Static HIN)")
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
    official_user_emb = cfg.coldrec_root / "emb" / f"{cfg.dataset_name}_cold_item_M2VAE_user_emb.pt"
    official_item_emb = cfg.coldrec_root / "emb" / f"{cfg.dataset_name}_cold_item_M2VAE_item_emb.pt"

    result = {
        "model": "M2VAE-coldrec-official",
        "model_display": "M2VAE",
        "source": "ColdRec official source executed in-process via main.py Config/model_factory/model.run; adapter only converts data and evaluates mapped embeddings.",
        "official_source_dir": str(cfg.coldrec_root),
        "official_code": str(cfg.coldrec_root),
        **git_info,
        "paper": "M2VAE: Multi-Modal Multi-View Variational Autoencoder for Cold-start Item Recommendation",
        "paper_venue": "AAAI 2026",
        "protocol": "static_item_cold",
        "score_function": "raw_dot_product",
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
        "topN": cfg.topn,
        "extra_args": cfg.extra_args,
        "test_history_policy": cfg.test_history_policy,
        "m2vae_positive_number": cfg.positive_number,
        "m2vae_negative_number": cfg.negative_number,
        "m2vae_self_neg_number": cfg.self_neg_number,
        "m2vae_attr_present_dim": cfg.attr_present_dim,
        "m2vae_implicit_dim": cfg.implicit_dim,
        "m2vae_cat_implicit_dim": cfg.cat_implicit_dim,
        "m2vae_tau": cfg.tau,
        "m2vae_weight_decay": cfg.weight_decay,
        "m2vae_kld_weight": cfg.kld_weight,
        "m2vae_recon_weight": cfg.recon_weight,
        "m2vae_decouple_weight": cfg.decouple_weight,
        "mapped_embedding_checkpoint": str(checkpoint_path),
        "official_user_embedding_checkpoint": str(official_user_emb),
        "official_item_embedding_checkpoint": str(official_item_emb),
        "official_embedding_checkpoints_present": bool(official_user_emb.exists() and official_item_emb.exists()),
        "note": "ColdRec native cold metrics mask warm items; main-table metrics above are recomputed with the paper full-catalog evaluator.",
    }
    result_path = _cfg_result_path(cfg, "m2vae_coldrec_static_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
