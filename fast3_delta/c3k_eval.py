"""Full-catalog evaluation for Cold-Consistent Knowledge Calibration."""

from __future__ import annotations

import csv
import dataclasses
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fast3_delta.eval import compute_ranking_metric_values


@dataclasses.dataclass(frozen=True)
class C3KItemBank:
    """Candidate-invariant course tensors permitted to be cached at inference."""

    item_vectors: torch.Tensor
    strict_cold_mask: torch.Tensor
    all_cold_item_vectors: torch.Tensor
    item_bank_seconds: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_c3k_item_bank(
    model,
    device: torch.device,
    *,
    item_batch: int = 128,
) -> C3KItemBank:
    """Build the mixed strict-cold/warm candidate bank without user evidence."""
    device = torch.device(device)
    popularity = getattr(model, "item_popularity", None)
    if popularity is None:
        raise RuntimeError("C3K item bank requires train-derived item popularity")
    strict_cold_mask = popularity.to(device=device).float().view(-1) < float(
        model.cfg.cold_threshold
    )
    if int(strict_cold_mask.numel()) != int(model.cfg.n_items):
        raise RuntimeError("item popularity and candidate catalog have different sizes")

    was_training = model.training
    model.eval()
    item_vectors = []
    all_cold_vectors = []
    block_size = max(1, int(item_batch))
    _sync(device)
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for begin in range(0, int(model.cfg.n_items), block_size):
                end = min(int(model.cfg.n_items), begin + block_size)
                item_ids = torch.arange(begin, end, dtype=torch.long, device=device)
                strict_chunk = strict_cold_mask[begin:end]
                llm = model._empty_llm(item_ids)
                item_vectors.append(model.item_view(item_ids, llm, strict_chunk))
                all_cold_vectors.append(
                    model.item_view(item_ids, llm, torch.ones_like(strict_chunk))
                )
        _sync(device)
        item_bank_seconds = time.perf_counter() - start
    finally:
        model.train(was_training)
    return C3KItemBank(
        item_vectors=torch.cat(item_vectors, dim=0),
        strict_cold_mask=strict_cold_mask,
        all_cold_item_vectors=torch.cat(all_cold_vectors, dim=0),
        item_bank_seconds=float(item_bank_seconds),
    )


def _seen_mask(model, user_ids: torch.Tensor, user_seen_items: Mapping[int, set[int]] | None) -> torch.Tensor:
    """Build a full-catalog Boolean mask from the declared history policy."""
    device = user_ids.device
    cached = getattr(model, "user_seen_index", None)
    if cached is not None:
        return cached.index_select(0, user_ids).to(device=device, dtype=torch.bool)
    mask = torch.zeros(
        (int(user_ids.numel()), int(model.cfg.n_items)), dtype=torch.bool, device=device
    )
    if not user_seen_items:
        return mask
    for row, user_id in enumerate(user_ids.detach().cpu().tolist()):
        seen = user_seen_items.get(int(user_id), ())
        valid = [int(item_id) for item_id in seen if 0 <= int(item_id) < model.cfg.n_items]
        if valid:
            mask[row, torch.as_tensor(valid, dtype=torch.long, device=device)] = True
    return mask


def _selected_rows(popularity: torch.Tensor, eval_type: str, cold_threshold: float) -> torch.Tensor:
    eval_type = str(eval_type).strip().lower()
    if eval_type == "cold":
        return popularity < cold_threshold
    if eval_type == "hot":
        return popularity >= cold_threshold
    if eval_type == "all":
        return torch.ones_like(popularity, dtype=torch.bool)
    raise ValueError("eval_type must be one of: cold, hot, all")


def _write_item_metrics(path: str, item_counts: Mapping[int, int], item_sums: Mapping[str, Mapping[int, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(item_sums)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "count"] + keys)
        writer.writeheader()
        for item_id in sorted(item_counts):
            count = max(1, int(item_counts[item_id]))
            row: dict[str, Any] = {"item_id": int(item_id), "count": count}
            row.update({key: float(item_sums[key][item_id] / count) for key in keys})
            writer.writerow(row)


def evaluate_c3k(
    model,
    loader,
    device: torch.device,
    *,
    item_bank: C3KItemBank | None = None,
    k_list: Sequence[int] = (5, 10, 20),
    eval_type: str = "cold",
    user_seen_items: Mapping[int, set[int]] | None = None,
    average_mode: str = "item_macro",
    item_block: int = 128,
    query_block: int = 32,
    calibration: bool = True,
    export_item_metrics_path: str | None = None,
) -> tuple[dict[str, float] | None, int, dict[str, float | int]]:
    """Evaluate full ranking using C3K's shared score for every candidate."""
    device = torch.device(device)
    average_mode = str(average_mode).strip().lower()
    if average_mode not in {"item_macro", "interaction"}:
        raise ValueError("average_mode must be 'item_macro' or 'interaction'")
    if min(k_list, default=1) < 1:
        raise ValueError("all ranking cutoffs must be positive")
    query_block = max(1, int(query_block))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    total_start = time.perf_counter()
    built_here = item_bank is None
    if item_bank is None:
        item_bank = build_c3k_item_bank(model, device, item_batch=item_block)

    metric_sums: dict[str, float] = defaultdict(float)
    item_sums: dict[str, dict[int, float]] = {
        f"{metric}@{k}": defaultdict(float) for metric in ("R", "N") for k in k_list
    }
    item_counts: dict[int, int] = defaultdict(int)
    query_count = 0
    score_seconds = 0.0
    ranking_seconds = 0.0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch, pop, _llm in loader:
                pop = pop.to(device=device, dtype=torch.float32).view(-1)
                selected = _selected_rows(pop, eval_type, float(model.cfg.cold_threshold))
                if not bool(selected.any()):
                    continue
                batch_users = batch["u"].to(device=device, dtype=torch.long).view(-1)
                batch_items = batch["i"].to(device=device, dtype=torch.long).view(-1)
                user_ids = batch_users[selected]
                target_ids = batch_items[selected]
                for begin in range(0, int(user_ids.numel()), query_block):
                    end = min(int(user_ids.numel()), begin + query_block)
                    users = user_ids[begin:end]
                    targets = target_ids[begin:end]
                    _sync(device)
                    score_start = time.perf_counter()
                    scores = model.score_catalog(
                        users,
                        item_bank.item_vectors,
                        user_seen_items,
                        item_block=item_block,
                        calibration=calibration,
                    )
                    _sync(device)
                    score_seconds += time.perf_counter() - score_start

                    ranking_start = time.perf_counter()
                    rows = torch.arange(int(users.numel()), device=device)
                    target_scores = scores[rows, targets].clone()
                    scores = scores.masked_fill(_seen_mask(model, users, user_seen_items), -1e9)
                    scores[rows, targets] = target_scores
                    values = compute_ranking_metric_values(
                        scores, target_indices=targets, k_list=k_list
                    )
                    _sync(device)
                    ranking_seconds += time.perf_counter() - ranking_start
                    target_list = [int(item_id) for item_id in targets.detach().cpu().tolist()]
                    if average_mode == "interaction":
                        for key, metric_values in values.items():
                            metric_sums[key] += float(metric_values.sum().item())
                    else:
                        for row, item_id in enumerate(target_list):
                            item_counts[item_id] += 1
                            for key, metric_values in values.items():
                                item_sums[key][item_id] += float(metric_values[row].item())
                    query_count += int(users.numel())
    finally:
        model.train(was_training)

    _sync(device)
    total_seconds = time.perf_counter() - total_start
    if query_count < 1:
        metrics = None
        count = 0
    elif average_mode == "interaction":
        metrics = {key: value / query_count for key, value in metric_sums.items()}
        count = query_count
    else:
        metrics = {}
        for key, per_item in item_sums.items():
            values = [per_item[item_id] / item_counts[item_id] for item_id in item_counts]
            metrics[key] = sum(values) / max(1, len(values))
        count = len(item_counts)
        if export_item_metrics_path:
            _write_item_metrics(export_item_metrics_path, item_counts, item_sums)

    timing: dict[str, float | int] = {
        "item_bank_seconds": float(item_bank.item_bank_seconds if built_here else 0.0),
        "score_seconds": float(score_seconds),
        "ranking_seconds": float(ranking_seconds),
        "total_inference_seconds": float(total_seconds),
        "query_count": int(query_count),
        "candidate_count": int(model.cfg.n_items),
        "peak_memory_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    return metrics, count, timing


__all__ = ["C3KItemBank", "build_c3k_item_bank", "evaluate_c3k"]
