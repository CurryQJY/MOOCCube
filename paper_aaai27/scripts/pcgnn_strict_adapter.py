from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_aaai27.scripts.priority_baseline_experiments import (
    PCGNN_ROOT,
    SPLIT_ROOT,
    build_train_histories,
    local_pcgnn_recbole,
    pcgnn_smoke_config_overrides,
    rows_from_frame,
    tensorize_examples,
)


OUT_DIR = ROOT / "paper_aaai27" / "baseline_sources" / "_pcgnn_strict" / "mooccube_seed2025"
DEFAULT_PCGNN_DATASET_NAME = "mooccube_strict_seed2025_full"
DEFAULT_PCGNN_CONFIG = PCGNN_ROOT / f"recbole_{DEFAULT_PCGNN_DATASET_NAME}.yaml"


def _limit_value(value: int) -> int | None:
    return None if value < 0 else value


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_torch_device(requested_device: str):
    import torch

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot access a CUDA device")
    return torch.device(requested_device)


@contextmanager
def clean_argv_for_recbole():
    old_argv = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        yield
    finally:
        sys.argv = old_argv


def metric_from_report(report: dict[str, object], metric_path: str) -> float:
    current: object = report
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return float("-inf")
        current = current[part]
    try:
        return float(current)
    except (TypeError, ValueError):
        return float("-inf")


def format_epoch_progress(
    epoch: int,
    loss: float,
    metric_name: str,
    metric_value: float,
    improved: bool,
    rs_loss: float | None = None,
    kg_loss: float | None = None,
) -> str:
    marker = " best" if improved else ""
    parts = [f"epoch={epoch}", f"loss={loss:.4f}"]
    if rs_loss is not None:
        parts.append(f"rs_loss={rs_loss:.4f}")
    if kg_loss is not None:
        parts.append(f"kg_loss={kg_loss:.4f}")
    parts.append(f"{metric_name}={metric_value:.8f}")
    return " ".join(parts) + marker


class BestValidationTracker:
    def __init__(self, metric_path: str, patience: int) -> None:
        self.metric_path = metric_path
        self.patience = int(patience)
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.best_state = None
        self.best_report: dict[str, object] | None = None
        self.bad_epochs = 0
        self.should_stop = False

    def update(self, epoch: int, validation_report: dict[str, object], state: object) -> bool:
        score = metric_from_report(validation_report, self.metric_path)
        if self.best_epoch == 0 or score > self.best_score:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.best_state = copy.deepcopy(state)
            self.best_report = copy.deepcopy(validation_report)
            self.bad_epochs = 0
            self.should_stop = False
            return True
        self.bad_epochs += 1
        self.should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return False


def build_strict_train_examples(
    train_rows: Iterable[dict[str, int]],
    token_map: dict[str, int],
    max_len: int,
    limit: int | None = None,
) -> list[dict[str, object]]:
    by_user: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in train_rows:
        if str(row["i_idx"]) in token_map:
            by_user[int(row["u_idx"])].append(row)

    examples: list[dict[str, object]] = []
    for user in sorted(by_user):
        history: list[int] = []
        for row in sorted(by_user[user], key=lambda r: int(r["timestamp"])):
            target = token_map.get(str(row["i_idx"]))
            if target is None:
                continue
            if history:
                examples.append(
                    {
                        "user": user,
                        "history": history[-max_len:],
                        "target": int(target),
                        "raw_item": int(row["i_idx"]),
                        "popularity": int(row.get("popularity", 0)),
                    }
                )
                if limit is not None and len(examples) >= limit:
                    return examples
            history.append(int(target))
    return examples


def build_strict_eval_examples(
    train_rows: Iterable[dict[str, int]],
    eval_rows: Iterable[dict[str, int]],
    token_map: dict[str, int],
    max_len: int,
    limit: int | None = None,
) -> list[dict[str, object]]:
    histories = build_train_histories(train_rows, token_map, max_len)
    examples: list[dict[str, object]] = []
    for row in sorted(eval_rows, key=lambda r: (int(r["u_idx"]), int(r["timestamp"]))):
        user = int(row["u_idx"])
        target = token_map.get(str(row["i_idx"]))
        history = histories.get(user, [])
        if target is None or not history:
            continue
        examples.append(
            {
                "user": user,
                "history": history[-max_len:],
                "target": int(target),
                "raw_item": int(row["i_idx"]),
                "popularity": int(row.get("popularity", 0)),
            }
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def build_user_seen_items(train_rows: Iterable[dict[str, int]], token_map: dict[str, int]) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for row in train_rows:
        token_id = token_map.get(str(row["i_idx"]))
        if token_id is not None:
            seen[int(row["u_idx"])].add(int(token_id))
    return seen


def build_train_item_ids(train_rows: Iterable[dict[str, int]], token_map: dict[str, int]) -> list[int]:
    item_ids = {
        int(token_id)
        for row in train_rows
        if (token_id := token_map.get(str(row["i_idx"]))) is not None and int(token_id) > 0
    }
    return sorted(item_ids)


class ItemMacroRankingAccumulator:
    def __init__(self, k_list: Iterable[int] = (5, 10, 20), cold_threshold: int = 1) -> None:
        self.k_list = tuple(int(k) for k in k_list)
        self.cold_threshold = int(cold_threshold)
        self.item_sum = {
            "cold": {f"{m}@{k}": defaultdict(float) for m in ("R", "N") for k in self.k_list},
            "hot": {f"{m}@{k}": defaultdict(float) for m in ("R", "N") for k in self.k_list},
        }
        self.item_count = {"cold": defaultdict(int), "hot": defaultdict(int)}
        self.rows = {"cold": 0, "hot": 0}

    def add_batch(
        self,
        scores: np.ndarray,
        examples: list[dict[str, object]],
        user_seen_items: dict[int, set[int]],
    ) -> None:
        if scores.shape[0] != len(examples):
            raise ValueError("scores row count must match examples")
        if not self.k_list:
            return

        masked = np.array(scores, copy=True)
        if masked.shape[1] > 0:
            masked[:, 0] = -np.inf

        target_scores: list[float] = []
        for row_idx, example in enumerate(examples):
            target = int(example["target"])
            target_scores.append(float(masked[row_idx, target]))
            for seen in user_seen_items.get(int(example["user"]), set()):
                if 0 <= int(seen) < masked.shape[1]:
                    masked[row_idx, int(seen)] = -np.inf
            if 0 <= target < masked.shape[1]:
                masked[row_idx, target] = target_scores[-1]

        max_k = min(max(self.k_list), masked.shape[1])
        order = np.argsort(-masked, axis=1)[:, :max_k]
        for row_idx, example in enumerate(examples):
            target = int(example["target"])
            raw_item = int(example["raw_item"])
            group = "cold" if int(example.get("popularity", 0)) < self.cold_threshold else "hot"
            self.item_count[group][raw_item] += 1
            self.rows[group] += 1
            for k in self.k_list:
                preds = order[row_idx, : min(k, order.shape[1])]
                hit_positions = np.where(preds == target)[0]
                hit = 1.0 if hit_positions.size > 0 else 0.0
                ndcg = 1.0 / math.log2(float(hit_positions[0]) + 2.0) if hit else 0.0
                self.item_sum[group][f"R@{k}"][raw_item] += hit
                self.item_sum[group][f"N@{k}"][raw_item] += ndcg

    def _group_result(self, group: str) -> tuple[dict[str, float], int]:
        counts = self.item_count[group]
        if not counts:
            return {}, 0
        result: dict[str, float] = {}
        for key, per_item in self.item_sum[group].items():
            values = [float(per_item[item]) / count for item, count in counts.items() if count > 0]
            result[key] = float(sum(values) / len(values)) if values else 0.0
        return result, len(counts)

    def result(self) -> dict[str, object]:
        cold, cold_count = self._group_result("cold")
        hot, hot_count = self._group_result("hot")
        return {
            "full_cold_item_macro": cold,
            "full_hot_item_macro": hot,
            "count_full_cold_item_macro": cold_count,
            "count_full_hot_item_macro": hot_count,
            "rows_full_cold": self.rows["cold"],
            "rows_full_hot": self.rows["hot"],
        }


def _iter_batches(examples: list[dict[str, object]], batch_size: int):
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def move_tensor_dict_to_device(values: dict[str, object], device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in values.items()
    }


@dataclass
class KGTrainingPool:
    head_field: str
    relation_field: str
    tail_field: str
    head_ids: np.ndarray
    relation_ids: np.ndarray
    tail_ids: np.ndarray
    entity_count: int
    used_tails_by_head: dict[int, set[int]]

    def __len__(self) -> int:
        return int(self.head_ids.shape[0])


def _kg_field_array(kg_feat: object, field: str) -> np.ndarray:
    if isinstance(kg_feat, pd.DataFrame):
        if field not in kg_feat.columns:
            raise ValueError(f"KG feature is missing required field [{field}]")
        values = kg_feat[field].to_numpy(copy=True)
    else:
        try:
            values = kg_feat[field]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"KG feature is missing required field [{field}]") from exc
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def build_kg_training_pool(
    kg_feat: object,
    head_field: str,
    relation_field: str,
    tail_field: str,
    entity_count: int,
) -> KGTrainingPool:
    head_ids = _kg_field_array(kg_feat, head_field)
    relation_ids = _kg_field_array(kg_feat, relation_field)
    tail_ids = _kg_field_array(kg_feat, tail_field)

    if not (len(head_ids) == len(relation_ids) == len(tail_ids)):
        raise ValueError("KG head/relation/tail fields must have the same length")
    if len(head_ids) == 0:
        raise ValueError("KG feature has no triples")
    if entity_count <= 1:
        raise ValueError("entity_count must include at least one non-padding entity")
    if np.any(head_ids <= 0) or np.any(tail_ids <= 0):
        raise ValueError("KG head/tail ids must be positive; id 0 is reserved for padding")
    if np.any(head_ids >= entity_count) or np.any(tail_ids >= entity_count):
        raise ValueError("KG head/tail ids must be smaller than entity_count")

    used_tails_by_head: dict[int, set[int]] = defaultdict(set)
    for head, tail in zip(head_ids.tolist(), tail_ids.tolist()):
        used_tails_by_head[int(head)].add(int(tail))

    return KGTrainingPool(
        head_field=head_field,
        relation_field=relation_field,
        tail_field=tail_field,
        head_ids=head_ids,
        relation_ids=relation_ids,
        tail_ids=tail_ids,
        entity_count=int(entity_count),
        used_tails_by_head=dict(used_tails_by_head),
    )


def _rng_integers(rng: np.random.Generator, low: int, high: int, size: int | None = None):
    return rng.integers(low, high, size=size)


def sample_kg_batch(
    pool: KGTrainingPool,
    batch_size: int,
    neg_tail_field: str,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    import torch

    if batch_size <= 0:
        raise ValueError("kg batch_size must be positive")
    if len(pool) == 0:
        raise ValueError("KG training pool is empty")
    rng = rng or np.random.default_rng()
    indices = _rng_integers(rng, 0, len(pool), size=batch_size)
    head_ids = pool.head_ids[indices]
    relation_ids = pool.relation_ids[indices]
    tail_ids = pool.tail_ids[indices]
    neg_tail_ids = np.empty(batch_size, dtype=np.int64)

    for row_idx, head in enumerate(head_ids.tolist()):
        used_tails = pool.used_tails_by_head.get(int(head), set())
        if len(used_tails) >= pool.entity_count - 1:
            raise ValueError(f"head entity [{head}] is linked to every non-padding entity")
        while True:
            candidate = int(_rng_integers(rng, 1, pool.entity_count))
            if candidate not in used_tails:
                neg_tail_ids[row_idx] = candidate
                break

    return {
        pool.head_field: torch.as_tensor(head_ids, dtype=torch.long),
        pool.relation_field: torch.as_tensor(relation_ids, dtype=torch.long),
        pool.tail_field: torch.as_tensor(tail_ids, dtype=torch.long),
        neg_tail_field: torch.as_tensor(neg_tail_ids, dtype=torch.long),
    }


def calculate_rs_loss_with_candidates(model, interaction, candidate_item_ids=None):
    import torch

    if candidate_item_ids is None:
        return model.calculate_rs_loss(interaction)
    if getattr(model, "loss_type", "CE") != "CE":
        return model.calculate_rs_loss(interaction)

    item_seq = interaction[model.ITEM_SEQ]
    item_seq_len = interaction[model.ITEM_SEQ_LEN]
    pos_items = interaction[model.ITEM_ID]
    device = item_seq.device
    candidate_indices = torch.as_tensor(candidate_item_ids, dtype=torch.long)
    candidates = candidate_indices.to(device)
    if candidates.numel() == 0:
        raise ValueError("candidate_item_ids must not be empty")

    item_to_label = torch.full((model.n_items,), -1, dtype=torch.long, device=device)
    item_to_label[candidates] = torch.arange(candidates.numel(), dtype=torch.long, device=device)
    labels = item_to_label[pos_items]
    if torch.any(labels < 0):
        missing = pos_items[labels < 0].detach().cpu().tolist()
        raise ValueError(f"positive train items missing from candidate_item_ids: {missing[:5]}")

    output, cat_prediction = model.forward(item_seq, item_seq_len)
    cat = model._get_cat_seq(candidate_indices)
    correspond_cat_emb = model.entity_embedding(cat)
    test_item_emb = torch.cat([model.entity_embedding(candidates), correspond_cat_emb], dim=1)
    logits = torch.matmul(output, test_item_emb.transpose(0, 1))
    loss = model.loss_fct(logits, labels)

    if getattr(model, "aux_weight", 0) != 0:
        pos_cats = model.dataset.dataset.item_feat["first_level_category"][pos_items.detach().cpu()].to(device)
        all_cat = [i for i in range(0, model.n_cats)]
        all_cat = model.dataset.dataset.field2id_token["first_level_category"][all_cat]
        all_cat = [model.dataset.dataset.field2token_id["entity_id"][i] for i in all_cat]
        all_cat = torch.tensor(all_cat, dtype=torch.long, device=device)
        all_cat_emb = model.entity_embedding(all_cat)
        cat_logits = torch.matmul(cat_prediction, all_cat_emb.transpose(0, 1))
        loss = loss + model.loss_fct(cat_logits, pos_cats) * model.aux_weight

    return loss


def train_one_epoch(
    model,
    interaction_cls,
    examples: list[dict[str, object]],
    max_len: int,
    batch_size: int,
    optimizer,
    kg_pool: KGTrainingPool | None = None,
    kg_batch_size: int = 0,
    kg_loss_weight: float = 1.0,
    rng: np.random.Generator | None = None,
    rs_candidate_item_ids=None,
    device=None,
) -> dict[str, float]:
    model.train()
    random.shuffle(examples)
    losses: list[float] = []
    rs_losses: list[float] = []
    kg_losses: list[float] = []
    use_kg = kg_pool is not None and kg_batch_size > 0 and kg_loss_weight != 0.0
    rng = rng or np.random.default_rng()
    for batch in _iter_batches(examples, batch_size):
        item_seq, item_len, target = tensorize_examples(batch, max_len)
        interaction_data = move_tensor_dict_to_device(
            {
            model.ITEM_SEQ: item_seq,
            model.ITEM_SEQ_LEN: item_len,
            model.ITEM_ID: target,
            },
            device,
        )
        if use_kg:
            interaction_data.update(
                move_tensor_dict_to_device(
                    sample_kg_batch(
                    kg_pool,
                    batch_size=kg_batch_size,
                    neg_tail_field=model.NEG_TAIL_ENTITY_ID,
                    rng=rng,
                    ),
                    device,
                )
            )
        interaction = interaction_cls(interaction_data)
        optimizer.zero_grad()
        rs_loss = calculate_rs_loss_with_candidates(model, interaction, rs_candidate_item_ids)
        if use_kg:
            kg_loss = model.calculate_kg_loss(interaction)
            loss = rs_loss + float(kg_loss_weight) * kg_loss
        else:
            kg_loss = None
            loss = rs_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        rs_losses.append(float(rs_loss.detach().cpu().item()))
        if kg_loss is not None:
            kg_losses.append(float(kg_loss.detach().cpu().item()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "rs_loss": float(np.mean(rs_losses)) if rs_losses else 0.0,
        "kg_loss": float(np.mean(kg_losses)) if kg_losses else 0.0,
    }


def evaluate_pcgnn_full_item_macro(
    model,
    interaction_cls,
    examples: list[dict[str, object]],
    user_seen_items: dict[int, set[int]],
    max_len: int,
    batch_size: int,
    k_list: Iterable[int] = (5, 10, 20),
    cold_threshold: int = 1,
    device=None,
) -> dict[str, object]:
    import torch

    model.eval()
    accumulator = ItemMacroRankingAccumulator(k_list=k_list, cold_threshold=cold_threshold)
    with torch.no_grad():
        for batch in _iter_batches(examples, batch_size):
            item_seq, item_len, _ = tensorize_examples(batch, max_len)
            interaction = interaction_cls(
                move_tensor_dict_to_device(
                    {model.ITEM_SEQ: item_seq, model.ITEM_SEQ_LEN: item_len},
                    device,
                )
            )
            scores = model.full_sort_predict(interaction).detach().cpu().numpy()
            accumulator.add_batch(scores, batch, user_seen_items)
    return accumulator.result()


def _read_split_rows(split_root: Path) -> tuple[list[dict[str, int]], pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_pickle(split_root / "static_train.pkl")
    val_df = pd.read_pickle(split_root / "static_val.pkl")
    test_df = pd.read_pickle(split_root / "static_test.pkl")
    return rows_from_frame(train_df.assign(popularity=train_df["popularity"])), val_df, test_df


def _frame_rows(df: pd.DataFrame) -> list[dict[str, int]]:
    cols = ["u_idx", "i_idx", "timestamp", "popularity"]
    rows: list[dict[str, int]] = []
    for row in df[cols].itertuples(index=False):
        rows.append(
            {
                "u_idx": int(row.u_idx),
                "i_idx": int(row.i_idx),
                "timestamp": int(row.timestamp),
                "popularity": int(row.popularity),
            }
        )
    return rows


def run_pcgnn_strict_adapter(args: argparse.Namespace) -> dict[str, object]:
    import torch

    args.split_root = resolve_workspace_path(args.split_root)
    args.pcgnn_root = resolve_workspace_path(args.pcgnn_root)
    args.config_file = resolve_workspace_path(args.config_file)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = resolve_workspace_path(args.checkpoint_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_rows, val_df, test_df = _read_split_rows(args.split_root)
    val_rows = _frame_rows(val_df[val_df["_split_source"].eq("strict_item_cold_val")])
    test_rows = _frame_rows(test_df[test_df["_split_source"].eq("strict_item_cold_test")])

    with local_pcgnn_recbole(args.pcgnn_root):
        from recbole.config import Config
        from recbole.data import Interaction, create_dataset, data_preparation
        from recbole.utils import get_model

        with clean_argv_for_recbole():
            config = Config(
                model="kg_model",
                dataset=args.dataset_name,
                config_file_list=[str(args.config_file)],
                config_dict=pcgnn_smoke_config_overrides(
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    device=device.type,
                ),
            )
        dataset = create_dataset(config)
        train_data, _, _ = data_preparation(config, dataset)
        token_map = {str(k): int(v) for k, v in dataset.field2token_id["item_id"].items()}
        max_len = int(config["MAX_ITEM_LIST_LENGTH"])

        train_examples = build_strict_train_examples(
            train_rows,
            token_map,
            max_len=max_len,
            limit=_limit_value(args.max_train_examples),
        )
        val_examples = build_strict_eval_examples(
            train_rows,
            val_rows,
            token_map,
            max_len=max_len,
            limit=_limit_value(args.max_val_examples),
        )
        test_examples = build_strict_eval_examples(
            train_rows,
            test_rows,
            token_map,
            max_len=max_len,
            limit=_limit_value(args.max_test_examples),
        )
        user_seen_items = build_user_seen_items(train_rows, token_map)
        train_item_ids = build_train_item_ids(train_rows, token_map)
        rs_candidate_item_ids = (
            train_item_ids
            if args.rs_candidate_mode == "warm"
            else list(range(dataset.num(config["ITEM_ID_FIELD"])))
        )
        kg_pool = (
            build_kg_training_pool(
                dataset.kg_feat,
                head_field=config["HEAD_ENTITY_ID_FIELD"],
                relation_field=config["RELATION_ID_FIELD"],
                tail_field=config["TAIL_ENTITY_ID_FIELD"],
                entity_count=dataset.num(config["ENTITY_ID_FIELD"]),
            )
            if args.kg_loss_weight != 0.0 and args.kg_batch_size > 0
            else None
        )
        kg_rng = np.random.default_rng(args.seed)

        model = get_model(config["model"])(config, train_data).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        epoch_losses = []
        epoch_loss_history: list[dict[str, float]] = []
        validation_history: list[dict[str, object]] = []
        tracker = BestValidationTracker(args.validation_metric, args.early_stop_patience)
        stopped_early = False
        best_checkpoint_path = None
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = args.checkpoint_dir / "best_model.pt"

        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(
                model,
                Interaction,
                train_examples,
                max_len=max_len,
                batch_size=args.train_batch_size,
                optimizer=optimizer,
                kg_pool=kg_pool,
                kg_batch_size=args.kg_batch_size,
                kg_loss_weight=args.kg_loss_weight,
                rng=kg_rng,
                rs_candidate_item_ids=rs_candidate_item_ids,
                device=device,
            )
            loss = train_stats["loss"]
            epoch_losses.append(loss)
            epoch_loss_history.append(train_stats)
            validation_report = evaluate_pcgnn_full_item_macro(
                model,
                Interaction,
                val_examples,
                user_seen_items,
                max_len=max_len,
                batch_size=args.eval_batch_size,
                k_list=args.k_list,
                cold_threshold=args.cold_threshold,
                device=device,
            )
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            improved = tracker.update(epoch, validation_report, state)
            validation_score = metric_from_report(validation_report, args.validation_metric)
            validation_history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "rs_loss": train_stats["rs_loss"],
                    "kg_loss": train_stats["kg_loss"],
                    "validation_metric": args.validation_metric,
                    "validation_score": validation_score,
                    "improved": improved,
                    "validation": validation_report,
                }
            )
            print(
                format_epoch_progress(
                    epoch=epoch,
                    loss=loss,
                    metric_name=args.validation_metric,
                    metric_value=validation_score,
                    improved=improved,
                    rs_loss=train_stats["rs_loss"],
                    kg_loss=train_stats["kg_loss"] if kg_pool is not None else None,
                ),
                flush=True,
            )
            if improved and best_checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "validation_metric": args.validation_metric,
                        "validation_score": validation_score,
                        "model_state_dict": state,
                    },
                    best_checkpoint_path,
                )
            if tracker.should_stop:
                stopped_early = True
                break

        if tracker.best_state is not None:
            model.load_state_dict(tracker.best_state)

        validation = tracker.best_report or {}
        test = evaluate_pcgnn_full_item_macro(
            model,
            Interaction,
            test_examples,
            user_seen_items,
            max_len=max_len,
            batch_size=args.eval_batch_size,
            k_list=args.k_list,
            cold_threshold=args.cold_threshold,
            device=device,
        )

    return {
        "model": "PCGNN",
        "protocol": "strict_item_cold_full_catalog_item_macro",
        "seed": args.seed,
        "session_graph_backend": "torch_batch_scatter",
        "requested_device": args.device,
        "device": str(device),
        "dataset_name": args.dataset_name,
        "split_root": str(args.split_root),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "kg_batch_size": args.kg_batch_size,
        "kg_loss_weight": args.kg_loss_weight,
        "kg_triples": len(kg_pool) if kg_pool is not None else 0,
        "rs_candidate_mode": args.rs_candidate_mode,
        "rs_candidate_items": len(train_item_ids) if args.rs_candidate_mode == "warm" else int(model.n_items),
        "max_train_examples": args.max_train_examples,
        "max_val_examples": args.max_val_examples,
        "max_test_examples": args.max_test_examples,
        "train_sequence_examples": len(train_examples),
        "validation_sequence_examples": len(val_examples),
        "test_sequence_examples": len(test_examples),
        "epoch_losses": epoch_losses,
        "epoch_loss_history": epoch_loss_history,
        "validation_metric": args.validation_metric,
        "best_epoch": tracker.best_epoch,
        "best_validation_score": tracker.best_score,
        "early_stop_patience": args.early_stop_patience,
        "stopped_early": stopped_early,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "validation_history": validation_history,
        "validation": validation,
        "test": test,
        "notes": [
            "This adapter bypasses PCGNN's stock sequential dataloader because it drops strict item-cold validation/test batches.",
            "Training jointly optimizes PCGNN recommendation loss and KG margin-ranking loss; the external evaluator still uses strict full-catalog item-macro metrics.",
            "The default RS candidate mode computes cross-entropy only over train-split items, so strict item-cold validation/test courses are not treated as negative classes during RS training.",
            "Scores are produced by PCGNN full_sort_predict, then train-history items and padding token 0 are masked while the target score is restored.",
            "Reported metrics are full-catalog item-macro Recall/NDCG under the existing strict split. Runs with capped examples are adaptation/smoke results, not final paper numbers.",
        ],
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# PCGNN Strict Adapter Report",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Train sequence examples | {report['train_sequence_examples']} |",
        f"| Validation sequence examples | {report['validation_sequence_examples']} |",
        f"| Test sequence examples | {report['test_sequence_examples']} |",
        f"| Epochs | {report['epochs']} |",
        f"| Last loss | {report['epoch_losses'][-1] if report['epoch_losses'] else 0.0:.4f} |",
        f"| Last RS loss | {report['epoch_loss_history'][-1]['rs_loss'] if report.get('epoch_loss_history') else 0.0:.4f} |",
        f"| Last KG loss | {report['epoch_loss_history'][-1]['kg_loss'] if report.get('epoch_loss_history') else 0.0:.4f} |",
        f"| KG triples | {report.get('kg_triples', 0)} |",
        f"| KG loss weight | {report.get('kg_loss_weight', 0.0)} |",
        f"| RS candidate mode | {report.get('rs_candidate_mode', '')} |",
        f"| RS candidate items | {report.get('rs_candidate_items', 0)} |",
        f"| Validation metric | {report.get('validation_metric', '')} |",
        f"| Best epoch | {report.get('best_epoch', 0)} |",
        f"| Best validation score | {float(report.get('best_validation_score', 0.0)):.4f} |",
        f"| Stopped early | {report.get('stopped_early', False)} |",
        f"| Best checkpoint | {report.get('best_checkpoint_path') or ''} |",
        "",
        "## Validation",
        "",
        "```json",
        json.dumps(report["validation"], indent=2),
        "```",
        "",
        "## Test",
        "",
        "```json",
        json.dumps(report["test"], indent=2),
        "```",
        "",
        "## Notes",
        "",
    ]
    for note in report["notes"]:
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--pcgnn-root", type=Path, default=PCGNN_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_PCGNN_DATASET_NAME)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_PCGNN_CONFIG,
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--kg-batch-size", type=int, default=256)
    parser.add_argument("--kg-loss-weight", type=float, default=1.0)
    parser.add_argument("--rs-candidate-mode", choices=["warm", "full"], default="warm")
    parser.add_argument("--max-train-examples", type=int, default=-1)
    parser.add_argument("--max-val-examples", type=int, default=-1)
    parser.add_argument("--max-test-examples", type=int, default=-1)
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--k-list", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--validation-metric", default="full_cold_item_macro.N@10")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=Path, default=OUT_DIR / "checkpoints")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.out_dir = resolve_workspace_path(args.out_dir)
    report = run_pcgnn_strict_adapter(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "pcgnn_strict_adapter_report.json"
    md_path = args.out_dir / "pcgnn_strict_adapter_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
