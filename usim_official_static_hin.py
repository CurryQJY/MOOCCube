"""
Official USIM adapter for the project static item-cold protocol.

This runner keeps the USIM algorithm implementation in USIM-main unchanged:
  - warm_model.bprmf.BPRMF
  - cold_model.USIM.USIM

The code here only adapts the current processed dataframe/split artifacts to
the official repository's expected warm-backbone + content-mapper + RL rollout
pipeline, then evaluates with the project's full-ranking evaluator.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hin_data_common import (
    InteractionDataset,
    add_user_seen_from_df,
    build_user_seen,
    clone_user_seen,
    collate_interactions,
    load_hin_processed,
    setup_seed,
    static_result_path,
    static_split_df,
)
from hin_eval_common import evaluate_embedding_ranker, print_final_report
from lightgcn_static_hin import prepare_train_cache, sample_negatives


REPO_ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO_DIR = REPO_ROOT / "USIM-main"


def _env_bool(name: str, default: bool = False) -> bool:
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


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _import_official_classes():
    if not OFFICIAL_REPO_DIR.exists():
        raise FileNotFoundError(f"Missing official USIM repository: {OFFICIAL_REPO_DIR}")

    repo_str = str(OFFICIAL_REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    loaded_utils = sys.modules.get("utils")
    if loaded_utils is not None:
        utils_path = getattr(loaded_utils, "__file__", "") or ""
        if "USIM-main" not in utils_path.replace("\\", "/"):
            del sys.modules["utils"]

    from cold_model.USIM import USIM as OfficialUSIM  # type: ignore
    from warm_model.bprmf import BPRMF  # type: ignore

    return BPRMF, OfficialUSIM


def load_split_cold_threshold(split_dir: str, fallback: int) -> int:
    """Read the protocol cold threshold from split artifacts when available."""
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


def cold_item_ids_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_threshold: int,
) -> torch.Tensor:
    """Return unique eval cold item ids under the project's row-level cold flag."""
    del train_df  # The strict split defines eval-cold items from val/test rows.
    cold_parts = []
    for df in (val_df, test_df):
        if len(df) == 0:
            continue
        cold_parts.append(df.loc[df["popularity"] < cold_threshold, "i_idx"].to_numpy(np.int64))
    if not cold_parts:
        return torch.empty(0, dtype=torch.long)
    cold = np.unique(np.concatenate(cold_parts).astype(np.int64, copy=False))
    return torch.as_tensor(cold, dtype=torch.long)


def _tensor_to_int_set(values: Optional[torch.Tensor]) -> set:
    if values is None:
        return set()
    if values.numel() == 0:
        return set()
    return {int(x) for x in values.detach().cpu().tolist()}


def build_official_rl_records(
    train_df: pd.DataFrame,
    content_emb: torch.Tensor,
    excluded_item_ids: Optional[torch.Tensor] = None,
) -> List[Dict[str, object]]:
    """Group project interactions into official USIM item -> users records."""
    excluded = _tensor_to_int_set(excluded_item_ids)
    if excluded:
        train_df = train_df.loc[~train_df["i_idx"].isin(excluded)].copy()

    grouped = (
        train_df.groupby("i_idx", sort=True)["u_idx"]
        .agg(lambda s: sorted({int(x) for x in s.tolist()}))
        .reset_index()
    )

    records: List[Dict[str, object]] = []
    for row in grouped.itertuples(index=False):
        item_id = int(row.i_idx)
        if item_id < 0 or item_id >= int(content_emb.shape[0]):
            raise IndexError(f"Item id {item_id} is outside content_emb shape {tuple(content_emb.shape)}")
        users = list(row.u_idx)
        if not users:
            continue
        records.append(
            {
                "item": item_id,
                "user": users,
                "item_content": content_emb[item_id].detach().cpu().float().clone(),
            }
        )
    return records


class OfficialRLBatch:
    """Minimal adapter matching the official interaction object's API."""

    def __init__(self, records: Sequence[Dict[str, object]]):
        self.interaction = {
            "item": torch.as_tensor([int(r["item"]) for r in records], dtype=torch.long),
            "user": [
                torch.as_tensor(list(r["user"]), dtype=torch.long)  # type: ignore[arg-type]
                for r in records
            ],
            "item_content": torch.stack(
                [torch.as_tensor(r["item_content"], dtype=torch.float32) for r in records],
                dim=0,
            ),
        }

    def __getitem__(self, key: str):
        return self.interaction[key]

    def __len__(self) -> int:
        return int(self.interaction["item"].numel())

    def to(self, device: torch.device) -> "OfficialRLBatch":
        self.interaction["item"] = self.interaction["item"].to(device)
        self.interaction["item_content"] = self.interaction["item_content"].to(device)
        self.interaction["user"] = [u.to(device) for u in self.interaction["user"]]
        return self


def iter_official_rl_batches(
    records: Sequence[Dict[str, object]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[OfficialRLBatch]:
    order = list(range(len(records)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield OfficialRLBatch([records[i] for i in idx])


@dataclass
class Config:
    n_users: int
    n_items: int
    content_dim: int
    cold_threshold: int

    def __post_init__(self) -> None:
        self.seed = _env_int("USIM_OFFICIAL_SEED", _env_int("USIM_STATIC_SEED", 2025))
        self.static_seed = _env_int("USIM_STATIC_SEED", self.seed)

        self.emb_dim = _env_int("USIM_OFFICIAL_EMB_DIM", _env_int("USIM_EMB_DIM", 128))
        self.backbone_epochs = _env_int("USIM_OFFICIAL_BACKBONE_EPOCHS", 20)
        self.mapper_epochs = _env_int("USIM_OFFICIAL_MLP_EPOCHS", 20)
        self.rl_epochs = _env_int("USIM_OFFICIAL_RL_EPOCHS", 50)

        self.backbone_lr = _env_float("USIM_OFFICIAL_BACKBONE_LR", 1e-3)
        self.mapper_lr = _env_float("USIM_OFFICIAL_MLP_LR", 1e-3)

        self.backbone_batch_size = _env_int("USIM_OFFICIAL_BACKBONE_BATCH_SIZE", 4096)
        self.mapper_batch_size = _env_int("USIM_OFFICIAL_MLP_BATCH_SIZE", 4096)
        # Official USIM builds action masks over all users. MOOCCube has nearly
        # 200k users, so the original script's 128 batch default is too large
        # for this adapter's full action space on common GPUs.
        self.rl_batch_size = _env_int("USIM_OFFICIAL_RL_BATCH_SIZE", 8)
        self.eval_batch_size = _env_int("USIM_OFFICIAL_EVAL_BATCH_SIZE", 2048)

        self.max_rl_batches = _env_int("USIM_OFFICIAL_MAX_RL_BATCHES", 0)
        self.max_backbone_interactions = _env_int("USIM_OFFICIAL_MAX_BACKBONE_INTERACTIONS", 0)
        self.max_mapper_interactions = _env_int("USIM_OFFICIAL_MAX_MLP_INTERACTIONS", 0)
        self.max_rl_items = _env_int("USIM_OFFICIAL_MAX_RL_ITEMS", 0)

        self.max_time = _env_int("USIM_OFFICIAL_MAX_TIME", 3)
        self.transition_rate = _env_float("USIM_OFFICIAL_TRANSITION_RATE", 0.1)
        self.k = _env_int("USIM_OFFICIAL_K", 10)
        self.weight = _env_float("USIM_OFFICIAL_REWARD_WEIGHT", 0.5)
        self.reward_cost = _env_float("USIM_OFFICIAL_REWARD_COST", 0.01)

        self.eval_n_neg = _env_int("USIM_OFFICIAL_EVAL_N_NEG", _env_int("USIM_EVAL_N_NEG", 200))
        self.run_sampled_eval = _env_bool("USIM_OFFICIAL_RUN_SAMPLED_EVAL", False)
        self.best_average_mode = os.environ.get("USIM_OFFICIAL_BEST_AVG_MODE", "item_macro").strip().lower()
        if self.best_average_mode not in {"interaction", "item_macro"}:
            raise ValueError("USIM_OFFICIAL_BEST_AVG_MODE must be interaction or item_macro")

        self.train_ratio = _env_float("USIM_STATIC_TRAIN_RATIO", 0.8)
        self.val_ratio = _env_float("USIM_STATIC_VAL_RATIO", 0.1)
        self.test_history_policy = os.environ.get("USIM_STATIC_TEST_HISTORY", "train_only").strip().lower()
        self.exclude_eval_cold_from_train = _env_bool("USIM_OFFICIAL_EXCLUDE_EVAL_COLD_FROM_TRAIN", True)

        self.ckpt_dir = os.environ.get("USIM_OFFICIAL_CKPT_DIR", "").strip()
        default_ckpt_enabled = bool(self.ckpt_dir)
        self.save_ckpt = _env_bool("USIM_OFFICIAL_SAVE_CKPT", default_ckpt_enabled)
        self.auto_resume = _env_bool("USIM_OFFICIAL_AUTO_RESUME", default_ckpt_enabled)
        self.force_fresh = _env_bool("USIM_OFFICIAL_FORCE_FRESH", False)
        self.save_opt_state = _env_bool("USIM_OFFICIAL_SAVE_OPT_STATE", True)


def _official_args(cfg: Config, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        factor_num=cfg.emb_dim,
        content_dim=cfg.content_dim,
        device=device,
        max_time=cfg.max_time,
        transition_rate=cfg.transition_rate,
        k=cfg.k,
        weight=cfg.weight,
        reward_cost=cfg.reward_cost,
    )


def _limit_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def train_official_backbone(
    warm_model: torch.nn.Module,
    train_df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
) -> None:
    if cfg.backbone_epochs <= 0:
        print("Official BPRMF backbone: skipped (USIM_OFFICIAL_BACKBONE_EPOCHS=0)")
        return

    ckpt_dir = Path(cfg.ckpt_dir) if cfg.ckpt_dir else None
    final_path = ckpt_dir / "backbone_final.pt" if ckpt_dir else None
    latest_path = ckpt_dir / "backbone_latest.pt" if ckpt_dir else None
    if final_path and cfg.auto_resume and not cfg.force_fresh and final_path.exists():
        ckpt = _torch_load(final_path, map_location=device)
        warm_model.load_state_dict(ckpt["model_state"])
        print(f"Official BPRMF backbone: loaded final checkpoint {final_path}")
        return

    train_df = _limit_df(train_df, cfg.max_backbone_interactions, cfg.seed)
    users_np, pos_np, user_rows, user_neg_pool = prepare_train_cache(train_df, cfg.n_items)
    users = torch.as_tensor(users_np, dtype=torch.long, device=device)
    pos = torch.as_tensor(pos_np, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(warm_model.parameters(), lr=cfg.backbone_lr)
    n_train = int(users.numel())
    start_epoch = 0
    if latest_path and cfg.auto_resume and not cfg.force_fresh and latest_path.exists():
        ckpt = _torch_load(latest_path, map_location=device)
        warm_model.load_state_dict(ckpt["model_state"])
        opt_state = ckpt.get("optimizer_state")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Official BPRMF backbone: resume epoch={start_epoch} from {latest_path}")

    for epoch in range(start_epoch, cfg.backbone_epochs):
        warm_model.train()
        neg_np = sample_negatives(pos_np, user_rows, user_neg_pool, cfg.n_items)
        neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)
        perm = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.backbone_batch_size):
            idx = perm[start:start + cfg.backbone_batch_size]
            batch = torch.stack([users[idx], pos[idx], neg[idx]], dim=1)
            loss = warm_model.calculate_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
            n_batches += 1
        print(
            f"Official BPRMF epoch [{epoch + 1}/{cfg.backbone_epochs}] "
            f"loss={loss_sum / max(1, n_batches):.4f}",
            flush=True,
        )
        if latest_path and cfg.save_ckpt:
            _save_checkpoint(
                latest_path,
                {
                    "stage": "backbone",
                    "epoch": epoch + 1,
                    "n_epochs": cfg.backbone_epochs,
                    "model_state": warm_model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if cfg.save_opt_state else None,
                },
            )
    if final_path and cfg.save_ckpt:
        _save_checkpoint(
            final_path,
            {
                "stage": "backbone",
                "epoch": cfg.backbone_epochs,
                "n_epochs": cfg.backbone_epochs,
                "model_state": warm_model.state_dict(),
            },
        )


def train_official_mapper(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    content_emb: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> None:
    if cfg.mapper_epochs <= 0:
        print("Official content_mapper: skipped (USIM_OFFICIAL_MLP_EPOCHS=0)")
        return

    ckpt_dir = Path(cfg.ckpt_dir) if cfg.ckpt_dir else None
    final_path = ckpt_dir / "mapper_final.pt" if ckpt_dir else None
    latest_path = ckpt_dir / "mapper_latest.pt" if ckpt_dir else None
    if final_path and cfg.auto_resume and not cfg.force_fresh and final_path.exists():
        ckpt = _torch_load(final_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Official content_mapper: loaded final checkpoint {final_path}")
        return

    train_df = _limit_df(train_df, cfg.max_mapper_interactions, cfg.seed + 17)
    item_ids = torch.as_tensor(train_df["i_idx"].to_numpy(np.int64, copy=True), dtype=torch.long, device=device)
    content_dev = content_emb.float().to(device)
    optimizer = torch.optim.Adam(model.content_mapper.parameters(), lr=cfg.mapper_lr, weight_decay=1e-4)
    n_train = int(item_ids.numel())
    start_epoch = 0
    if latest_path and cfg.auto_resume and not cfg.force_fresh and latest_path.exists():
        ckpt = _torch_load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt_state = ckpt.get("optimizer_state")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Official content_mapper: resume epoch={start_epoch} from {latest_path}")

    for epoch in range(start_epoch, cfg.mapper_epochs):
        model.content_mapper.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.mapper_batch_size):
            idx = perm[start:start + cfg.mapper_batch_size]
            batch_items = item_ids[idx]
            batch_content = content_dev[batch_items]
            target = model.warm_model.get_item_embedding(batch_items).detach()
            hidden = F.relu(model.content_mapper.fc1(batch_content))
            hidden = F.dropout(hidden, p=0.001, training=True)
            pred = model.content_mapper.fc2(hidden)
            loss = torch.norm(target - pred, p=2, dim=1, keepdim=True).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
            n_batches += 1
        print(
            f"Official content_mapper epoch [{epoch + 1}/{cfg.mapper_epochs}] "
            f"loss={loss_sum / max(1, n_batches):.4f}",
            flush=True,
        )
        if latest_path and cfg.save_ckpt:
            _save_checkpoint(
                latest_path,
                {
                    "stage": "mapper",
                    "epoch": epoch + 1,
                    "n_epochs": cfg.mapper_epochs,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if cfg.save_opt_state else None,
                },
            )
    if final_path and cfg.save_ckpt:
        _save_checkpoint(
            final_path,
            {
                "stage": "mapper",
                "epoch": cfg.mapper_epochs,
                "n_epochs": cfg.mapper_epochs,
                "model_state": model.state_dict(),
            },
        )


def _state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict_from_cpu(model: torch.nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def _official_embeddings(
    model: torch.nn.Module,
    content_emb: torch.Tensor,
    warm_item_ids: torch.Tensor,
    cold_item_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    content_dev = content_emb.float().to(device)
    warm_dev = warm_item_ids.to(device)
    cold_dev = cold_item_ids.to(device)
    user_emb = model.get_user_emb().detach()
    item_emb = model.get_item_emb(content_dev, warm_dev, cold_dev).detach()
    return user_emb, item_emb


def evaluate_official_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    content_emb: torch.Tensor,
    warm_item_ids: torch.Tensor,
    cold_item_ids: torch.Tensor,
    user_seen_items: Dict[int, set],
    eval_type: str,
    average_mode: str,
    export_item_metrics_path: Optional[str] = None,
) -> tuple[Optional[Dict[str, float]], int]:
    model.eval()
    with torch.no_grad():
        all_user_emb, all_item_emb = _official_embeddings(
            model,
            content_emb,
            warm_item_ids,
            cold_item_ids,
            device,
        )
        get_user_fn = lambda batch: all_user_emb[batch["u"]]
        return evaluate_embedding_ranker(
            loader,
            device=device,
            n_items=cfg.n_items,
            cold_threshold=cfg.cold_threshold,
            get_user_vectors_fn=get_user_fn,
            all_item_vectors=all_item_emb,
            k_list=(5, 10, 20),
            n_neg=cfg.eval_n_neg,
            eval_type=eval_type,
            full_ranking=True,
            user_seen_items=user_seen_items,
            normalize_user=False,
            average_mode=average_mode,
            export_item_metrics_path=export_item_metrics_path,
        )


def train_official_rl(
    model: torch.nn.Module,
    records: Sequence[Dict[str, object]],
    cfg: Config,
    device: torch.device,
    val_loader: DataLoader,
    content_emb: torch.Tensor,
    warm_item_ids: torch.Tensor,
    cold_item_ids: torch.Tensor,
    val_seen: Dict[int, set],
) -> tuple[int, float]:
    if not records:
        raise ValueError("No official USIM RL records were built from the training split")

    ckpt_dir = Path(cfg.ckpt_dir) if cfg.ckpt_dir else None
    latest_path = ckpt_dir / "rl_latest.pt" if ckpt_dir else None
    best_path = ckpt_dir / "rl_best.pt" if ckpt_dir else None

    best_state = _state_dict_to_cpu(model)
    best_epoch = 0
    best_val = -1.0
    start_epoch = 0

    if latest_path and cfg.auto_resume and not cfg.force_fresh and latest_path.exists():
        latest = _torch_load(latest_path, map_location=device)
        model.load_state_dict(latest["model_state"])
        actor_state = latest.get("actor_optimizer_state")
        if actor_state is not None and hasattr(model, "actor_optimizer"):
            model.actor_optimizer.load_state_dict(actor_state)
        critic_state = latest.get("critic_optimizer_state")
        if critic_state is not None and hasattr(model, "critic_optimizer"):
            model.critic_optimizer.load_state_dict(critic_state)
        start_epoch = int(latest.get("epoch", 0))
        best_epoch = int(latest.get("best_epoch", 0))
        best_val = float(latest.get("best_val", -1.0))
        best_state = _state_dict_to_cpu(model)
        print(f"Official USIM RL: resume epoch={start_epoch} from {latest_path}", flush=True)

    if best_path and cfg.auto_resume and not cfg.force_fresh and best_path.exists():
        best_ckpt = _torch_load(best_path, map_location=device)
        best_state = {k: v.detach().cpu().clone() for k, v in best_ckpt["model_state"].items()}
        best_epoch = int(best_ckpt.get("best_epoch", best_epoch))
        best_val = float(best_ckpt.get("best_val", best_val))
        print(f"Official USIM RL: loaded best checkpoint epoch={best_epoch} val={best_val:.4f}", flush=True)

    if start_epoch >= cfg.rl_epochs:
        _load_state_dict_from_cpu(model, best_state, device)
        print(f"Official USIM RL: target epochs already reached ({start_epoch}/{cfg.rl_epochs})", flush=True)
        return best_epoch, best_val

    for epoch in range(start_epoch, cfg.rl_epochs):
        model.train()
        actor_sum = 0.0
        critic_sum = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(
            iter_official_rl_batches(records, cfg.rl_batch_size, shuffle=True, seed=cfg.seed + epoch)
        ):
            if cfg.max_rl_batches > 0 and batch_idx >= cfg.max_rl_batches:
                break
            batch.to(device)
            model.update_buffer(batch, epoch)
            if len(model.buffer) > 0:
                actor_loss, critic_loss = model.optimize(device)
                actor_sum += float(actor_loss.detach().cpu().item())
                critic_sum += float(critic_loss.detach().cpu().item())
            model.buffer_clear()
            n_batches += 1

        val_metrics, n_val = evaluate_official_model(
            model,
            val_loader,
            device,
            cfg,
            content_emb,
            warm_item_ids,
            cold_item_ids,
            val_seen,
            eval_type="cold",
            average_mode=cfg.best_average_mode,
        )
        val_n10 = float((val_metrics or {}).get("N@10", 0.0))
        if val_n10 > best_val:
            best_val = val_n10
            best_epoch = epoch + 1
            best_state = _state_dict_to_cpu(model)
            if best_path and cfg.save_ckpt:
                _save_checkpoint(
                    best_path,
                    {
                        "stage": "rl",
                        "epoch": epoch + 1,
                        "n_epochs": cfg.rl_epochs,
                        "best_epoch": best_epoch,
                        "best_val": best_val,
                        "model_state": best_state,
                    },
                )

        print(
            f"Official USIM RL epoch [{epoch + 1}/{cfg.rl_epochs}] "
            f"actor={actor_sum / max(1, n_batches):.4f} "
            f"critic={critic_sum / max(1, n_batches):.4f} "
            f"val_full_cold_N@10({cfg.best_average_mode})={val_n10:.4f} "
            f"val_items={n_val}",
            flush=True,
        )
        if latest_path and cfg.save_ckpt:
            _save_checkpoint(
                latest_path,
                {
                    "stage": "rl",
                    "epoch": epoch + 1,
                    "n_epochs": cfg.rl_epochs,
                    "best_epoch": best_epoch,
                    "best_val": best_val,
                    "model_state": model.state_dict(),
                    "actor_optimizer_state": (
                        model.actor_optimizer.state_dict()
                        if cfg.save_opt_state and hasattr(model, "actor_optimizer")
                        else None
                    ),
                    "critic_optimizer_state": (
                        model.critic_optimizer.state_dict()
                        if cfg.save_opt_state and hasattr(model, "critic_optimizer")
                        else None
                    ),
                },
            )

    _load_state_dict_from_cpu(model, best_state, device)
    return best_epoch, best_val


def _print_full_only_report(
    metrics_keys: Sequence[str],
    full_cold: Dict[str, float],
    full_hot: Dict[str, float],
    count_full_cold: int,
    count_full_hot: int,
    title: str,
) -> None:
    print("\n" + "=" * 76)
    print(f"         FINAL REPORT: full ranking only ({title})")
    print("=" * 76)
    print(f"{'Metric':<10} | {'Full Cold':<12} | {'Full Hot':<12}")
    print("-" * 76)
    for metric in metrics_keys:
        print(f"{metric:<10} | {full_cold.get(metric, 0.0):<12.4f} | {full_hot.get(metric, 0.0):<12.4f}")
    print("-" * 76)
    print(f"Full samples/items: Cold={count_full_cold}, Hot={count_full_hot}")
    print("=" * 76)


def main() -> None:
    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin_clean_pop5")
    split_dir = os.environ.get("USIM_STATIC_SPLIT_DIR", "").strip()
    fallback_threshold = _env_int("USIM_COLD_THRESHOLD", 5)
    cold_threshold = load_split_cold_threshold(split_dir, fallback=fallback_threshold)

    print(f"Loading data from {data_dir} ...", flush=True)
    meta, df, content_emb = load_hin_processed(data_dir)
    cfg = Config(
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
        content_dim=int(meta.get("content_dim", int(content_emb.shape[1]))),
        cold_threshold=cold_threshold,
    )
    setup_seed(cfg.seed)

    train_df, val_df, test_df = static_split_df(
        df,
        seed=cfg.static_seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    print(
        f"Static split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} | "
        f"cold_threshold={cfg.cold_threshold}",
        flush=True,
    )

    cold_item_ids = cold_item_ids_from_splits(train_df, val_df, test_df, cfg.cold_threshold)
    cold_set = _tensor_to_int_set(cold_item_ids)
    warm_item_ids = torch.as_tensor(
        [idx for idx in range(cfg.n_items) if idx not in cold_set],
        dtype=torch.long,
    )
    rl_excluded = cold_item_ids if cfg.exclude_eval_cold_from_train else None
    rl_train_df = train_df.loc[~train_df["i_idx"].isin(cold_set)].copy() if cold_set else train_df.copy()
    records = build_official_rl_records(train_df, content_emb, excluded_item_ids=rl_excluded)
    if cfg.max_rl_items > 0 and len(records) > cfg.max_rl_items:
        records = records[:cfg.max_rl_items]
    print(
        f"Official USIM items: warm={warm_item_ids.numel()}, eval_cold={cold_item_ids.numel()}, "
        f"rl_records={len(records)}",
        flush=True,
    )

    val_loader = DataLoader(
        InteractionDataset(val_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    test_loader = DataLoader(
        InteractionDataset(test_df),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    train_seen = build_user_seen(train_df)
    val_seen = train_seen
    test_seen = clone_user_seen(train_seen)
    if cfg.test_history_policy == "train_val":
        add_user_seen_from_df(test_seen, val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BPRMF, OfficialUSIM = _import_official_classes()
    official_args = _official_args(cfg, device)
    warm_model = BPRMF(cfg.n_users, cfg.n_items, official_args).to(device)

    print(
        f"Model: official USIM adapter | repo={OFFICIAL_REPO_DIR} | device={device} | "
        f"emb_dim={cfg.emb_dim} | BPR={cfg.backbone_epochs} MLP={cfg.mapper_epochs} RL={cfg.rl_epochs}",
        flush=True,
    )

    train_official_backbone(warm_model, rl_train_df, cfg, device)
    model = OfficialUSIM(warm_model, official_args).to(device)
    train_official_mapper(model, rl_train_df, content_emb, cfg, device)

    if cfg.rl_epochs > 0:
        best_epoch, best_val = train_official_rl(
            model,
            records,
            cfg,
            device,
            val_loader,
            content_emb,
            warm_item_ids,
            cold_item_ids,
            val_seen,
        )
    else:
        best_epoch = 0
        val_metrics, _ = evaluate_official_model(
            model,
            val_loader,
            device,
            cfg,
            content_emb,
            warm_item_ids,
            cold_item_ids,
            val_seen,
            eval_type="cold",
            average_mode=cfg.best_average_mode,
        )
        best_val = float((val_metrics or {}).get("N@10", 0.0))

    sample_cold: Dict[str, float] = {}
    sample_hot: Dict[str, float] = {}
    n_sc = 0
    n_sh = 0
    if cfg.run_sampled_eval:
        with torch.no_grad():
            all_user_emb, all_item_emb = _official_embeddings(
                model, content_emb, warm_item_ids, cold_item_ids, device
            )
            get_user_fn = lambda batch: all_user_emb[batch["u"]]
            sample_cold_raw, n_sc = evaluate_embedding_ranker(
                test_loader,
                device,
                cfg.n_items,
                cfg.cold_threshold,
                get_user_fn,
                all_item_emb,
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
                cfg.n_items,
                cfg.cold_threshold,
                get_user_fn,
                all_item_emb,
                k_list=(5, 10, 20),
                n_neg=cfg.eval_n_neg,
                eval_type="hot",
                full_ranking=False,
                user_seen_items=test_seen,
                normalize_user=False,
            )
        sample_cold = sample_cold_raw or {}
        sample_hot = sample_hot_raw or {}

    full_cold, n_fc = evaluate_official_model(
        model,
        test_loader,
        device,
        cfg,
        content_emb,
        warm_item_ids,
        cold_item_ids,
        test_seen,
        eval_type="cold",
        average_mode="interaction",
    )
    full_hot, n_fh = evaluate_official_model(
        model,
        test_loader,
        device,
        cfg,
        content_emb,
        warm_item_ids,
        cold_item_ids,
        test_seen,
        eval_type="hot",
        average_mode="interaction",
    )
    full_cold_item_macro, n_fc_item_macro = evaluate_official_model(
        model,
        test_loader,
        device,
        cfg,
        content_emb,
        warm_item_ids,
        cold_item_ids,
        test_seen,
        eval_type="cold",
        average_mode="item_macro",
        export_item_metrics_path=static_result_path("per_item_full_cold_usim_official_static.csv"),
    )
    full_hot_item_macro, n_fh_item_macro = evaluate_official_model(
        model,
        test_loader,
        device,
        cfg,
        content_emb,
        warm_item_ids,
        cold_item_ids,
        test_seen,
        eval_type="hot",
        average_mode="item_macro",
        export_item_metrics_path=static_result_path("per_item_full_hot_usim_official_static.csv"),
    )

    full_cold = full_cold or {}
    full_hot = full_hot or {}
    full_cold_item_macro = full_cold_item_macro or {}
    full_hot_item_macro = full_hot_item_macro or {}
    metrics_keys = [f"{metric}@{k}" for metric in ["R", "N"] for k in (5, 10, 20)]

    if cfg.run_sampled_eval:
        print_final_report(
            eval_n_neg=cfg.eval_n_neg,
            metrics_keys=metrics_keys,
            sample_cold=sample_cold,
            sample_hot=sample_hot,
            full_cold=full_cold,
            full_hot=full_hot,
            count_sample_cold=n_sc,
            count_sample_hot=n_sh,
            count_full_cold=n_fc,
            count_full_hot=n_fh,
            title="Official USIM Static HIN",
        )
    else:
        _print_full_only_report(
            metrics_keys=metrics_keys,
            full_cold=full_cold,
            full_hot=full_hot,
            count_full_cold=n_fc,
            count_full_hot=n_fh,
            title="Official USIM Static HIN",
        )

    out = {
        "model": "USIM-official",
        "model_display": "USIM",
        "source": "official repository adapter",
        "official_code": str(OFFICIAL_REPO_DIR),
        "paper": "User-Item State Interaction for Cold-Start Recommendation",
        "paper_venue": "NeurIPS 2024",
        "protocol": "static_item_cold",
        "score_function": "raw_dot_product",
        "sample_cold": sample_cold,
        "sample_hot": sample_hot,
        "full_cold": full_cold,
        "full_hot": full_hot,
        "full_cold_item_macro": full_cold_item_macro,
        "full_hot_item_macro": full_hot_item_macro,
        "count_sample_cold": n_sc,
        "count_sample_hot": n_sh,
        "count_full_cold": n_fc,
        "count_full_hot": n_fh,
        "count_full_cold_item_macro": n_fc_item_macro,
        "count_full_hot_item_macro": n_fh_item_macro,
        "best_epoch": best_epoch,
        "best_metric": f"full_cold_N@10_{cfg.best_average_mode}",
        "best_val_full_cold_n10": best_val,
        "best_average_mode": cfg.best_average_mode,
        "eval_n_neg": cfg.eval_n_neg,
        "static_seed": cfg.static_seed,
        "seed": cfg.seed,
        "cold_threshold": cfg.cold_threshold,
        "emb_dim": cfg.emb_dim,
        "backbone_epochs": cfg.backbone_epochs,
        "mapper_epochs": cfg.mapper_epochs,
        "rl_epochs": cfg.rl_epochs,
        "rl_batch_size": cfg.rl_batch_size,
        "max_time": cfg.max_time,
        "transition_rate": cfg.transition_rate,
        "k": cfg.k,
        "reward_weight": cfg.weight,
        "reward_cost": cfg.reward_cost,
        "warm_item_count": int(warm_item_ids.numel()),
        "eval_cold_item_count": int(cold_item_ids.numel()),
        "rl_record_count": int(len(records)),
        "per_item_full_cold_path": static_result_path("per_item_full_cold_usim_official_static.csv"),
        "per_item_full_hot_path": static_result_path("per_item_full_hot_usim_official_static.csv"),
    }
    result_path = static_result_path("usim_official_static_result.json")
    pd.DataFrame([out]).to_json(result_path, orient="records", force_ascii=False)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
