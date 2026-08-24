from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE_DIR = ROOT / "paper_aaai27" / "baseline_sources" / "KLU4EduRec"
DEFAULT_SPLIT_ROOT = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "static_item_cold_balanced"
    / "strict_item_cold_balanced_thr1_seed_2025"
)
DEFAULT_CONTENT_PATH = ROOT / "processed_data_hin_clean_pop5" / "content_emb.pt"
DEFAULT_OUT_DIR = (
    ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_klu4edurec_strict"
    / "mooccube_seed2025_item_se_smoke"
)
AUDITED_SOURCE_COMMIT = "57686b10c7a1d179ec9f6831a306b6d80b9f7b02"

from paper_aaai27.scripts.pcgnn_strict_adapter import BestValidationTracker  # noqa: E402


@dataclass(frozen=True)
class StrictDataset:
    train_positives: tuple[tuple[int, int], ...]
    validation_rows: tuple[tuple[int, int], ...]
    test_rows: tuple[tuple[int, int], ...]
    cold_items: frozenset[int]
    n_users: int
    n_items: int
    content_embeddings: torch.Tensor


@dataclass(frozen=True)
class TrainingStructures:
    edge_index: torch.Tensor
    user_items: dict[int, set[int]]
    warm_items: tuple[int, ...]
    positive_edges: int


_AUTHOR_CLASS_CACHE: dict[Path, type] = {}


def _segment_csr_fallback(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for index in range(int(indptr.numel()) - 1):
        start = int(indptr[index])
        stop = int(indptr[index + 1])
        segment = src[start:stop]
        if segment.numel() == 0:
            rows.append(torch.zeros(src.shape[1:], dtype=src.dtype, device=src.device))
        elif reduce == "sum":
            rows.append(segment.sum(dim=0))
        elif reduce == "mean":
            rows.append(segment.mean(dim=0))
        elif reduce == "max":
            rows.append(segment.max(dim=0).values)
        else:
            raise ValueError(f"unsupported segment reduction: {reduce}")
    return torch.stack(rows, dim=0)


def load_author_model_class(source_dir: Path) -> type:
    source_dir = Path(source_dir).resolve()
    if source_dir in _AUTHOR_CLASS_CACHE:
        return _AUTHOR_CLASS_CACHE[source_dir]

    previous = {name: sys.modules.get(name) for name in ("torch_scatter", "model", "model.metric")}
    try:
        scatter_module = types.ModuleType("torch_scatter")
        scatter_module.segment_csr = _segment_csr_fallback
        model_package = types.ModuleType("model")
        model_package.__path__ = []
        metric_module = types.ModuleType("model.metric")
        metric_module.all_ranking = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("the strict adapter uses its external full-catalog evaluator")
        )
        sys.modules["torch_scatter"] = scatter_module
        sys.modules["model"] = model_package
        sys.modules["model.metric"] = metric_module

        spec = importlib.util.spec_from_file_location(
            "klu_author_gnnrec",
            source_dir / "model" / "GNNRec.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load KLU4EduRec source from {source_dir}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        author_class = getattr(module, "LLM4EduRec")
        _AUTHOR_CLASS_CACHE[source_dir] = author_class
        return author_class
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def resolve_source_dir(requested: Path) -> Path:
    resolved = Path(requested).resolve()
    expected = DEFAULT_SOURCE_DIR.resolve()
    if resolved != expected:
        raise ValueError(f"KLU4EduRec execution is locked to the audited author source snapshot at {expected}; got {resolved}")
    return resolved


def _load_content_embeddings(path: Path, expected_n_items: int) -> torch.Tensor:
    content = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(content, torch.Tensor) or content.ndim != 2:
        raise ValueError("content embeddings must be a rank-2 torch tensor")
    if content.shape[0] != int(expected_n_items):
        raise ValueError(
            f"content catalog has {content.shape[0]} rows, expected {expected_n_items}"
        )
    content = content.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if not torch.isfinite(content).all():
        raise ValueError("content embeddings contain non-finite values")
    return content


def _strict_rows(frame, split_source: str, name: str) -> tuple[tuple[int, int], ...]:
    missing = {"u_idx", "i_idx"} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} split is missing columns {sorted(missing)}")
    if "_split_source" not in frame.columns:
        raise ValueError(f"{name} split is missing _split_source")
    selected = frame[frame["_split_source"].eq(split_source)]
    return tuple(
        (int(row.u_idx), int(row.i_idx))
        for row in selected[["u_idx", "i_idx"]].itertuples(index=False)
    )


def load_strict_dataset(
    split_root: Path,
    content_path: Path,
    expected_n_items: int = 698,
) -> StrictDataset:
    import pandas as pd

    split_root = Path(split_root)
    train = pd.read_pickle(split_root / "static_train.pkl")
    validation = pd.read_pickle(split_root / "static_val.pkl")
    test = pd.read_pickle(split_root / "static_test.pkl")
    missing_train = {"u_idx", "i_idx"} - set(train.columns)
    if missing_train:
        raise ValueError(f"train split is missing columns {sorted(missing_train)}")

    train_positives = tuple(
        dict.fromkeys(
            (int(row.u_idx), int(row.i_idx))
            for row in train[["u_idx", "i_idx"]].itertuples(index=False)
        )
    )
    validation_rows = _strict_rows(validation, "strict_item_cold_val", "validation")
    test_rows = _strict_rows(test, "strict_item_cold_test", "test")
    if not train_positives or not validation_rows or not test_rows:
        raise ValueError("strict train, validation, and test rows must all be nonempty")

    cold_items = frozenset(item for _, item in validation_rows + test_rows)
    leaked_train_items = {item for _, item in train_positives} & cold_items
    if leaked_train_items:
        raise ValueError(f"positive train rows contain cold courses: {sorted(leaked_train_items)[:10]}")

    interactions = train_positives + validation_rows + test_rows
    if any(item < 0 or item >= expected_n_items for _, item in interactions):
        raise ValueError("strict interactions contain a course outside the expected catalog")
    if any(user < 0 for user, _ in interactions):
        raise ValueError("strict interactions contain a negative user id")

    content = _load_content_embeddings(content_path, expected_n_items)
    return StrictDataset(
        train_positives=train_positives,
        validation_rows=validation_rows,
        test_rows=test_rows,
        cold_items=cold_items,
        n_users=max(user for user, _ in interactions) + 1,
        n_items=int(expected_n_items),
        content_embeddings=content,
    )


def build_train_structures(
    positives: Iterable[tuple[int, int]],
    n_users: int,
    n_items: int,
    forbidden_cold_items: set[int] | frozenset[int],
) -> TrainingStructures:
    user_items: dict[int, set[int]] = defaultdict(set)
    positive_pairs: set[tuple[int, int]] = set()
    for raw_user, raw_item in positives:
        user, item = int(raw_user), int(raw_item)
        if not (0 <= user < n_users and 0 <= item < n_items):
            raise ValueError(f"interaction ({user}, {item}) is outside declared cardinalities")
        if item in forbidden_cold_items:
            raise ValueError(f"positive train edge contains held-out cold course {item}")
        positive_pairs.add((user, item))
        user_items[user].add(item)
    if not positive_pairs:
        raise ValueError("positive training graph is empty")

    ordered = sorted(positive_pairs)
    users = torch.tensor([user for user, _ in ordered], dtype=torch.long)
    items = torch.tensor([n_users + item for _, item in ordered], dtype=torch.long)
    edge_index = torch.stack(
        [torch.cat([users, items]), torch.cat([items, users])],
        dim=0,
    ).contiguous()
    warm_items = tuple(sorted({item for _, item in ordered}))
    return TrainingStructures(
        edge_index=edge_index,
        user_items={user: set(items) for user, items in user_items.items()},
        warm_items=warm_items,
        positive_edges=len(ordered),
    )


def sample_warm_negatives(
    positives: Iterable[tuple[int, int]],
    warm_items: tuple[int, ...] | list[int],
    user_items: dict[int, set[int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    warm = tuple(int(item) for item in warm_items)
    if not warm:
        raise ValueError("warm item pool is empty")
    candidates_by_user: dict[int, tuple[int, ...]] = {}
    negatives: list[tuple[int, int]] = []
    for raw_user, _ in positives:
        user = int(raw_user)
        if user not in candidates_by_user:
            seen = user_items.get(user, set())
            candidates_by_user[user] = tuple(item for item in warm if item not in seen)
        candidates = candidates_by_user[user]
        if not candidates:
            raise ValueError(f"user {user} has no eligible warm negative")
        negatives.append((user, int(candidates[int(rng.integers(0, len(candidates)))])))
    return negatives


def prepare_epoch_triples(
    positives: Iterable[tuple[int, int]],
    structures: TrainingStructures,
    max_examples: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    rows = list((int(user), int(item)) for user, item in positives)
    if max_examples == 0:
        raise ValueError("max_train_examples must be -1 or positive")
    order = rng.permutation(len(rows))
    if max_examples > 0:
        order = order[: min(int(max_examples), len(rows))]
    selected = [rows[int(index)] for index in order]
    negatives = sample_warm_negatives(selected, structures.warm_items, structures.user_items, rng)
    return [(user, positive, negative) for (user, positive), (_, negative) in zip(selected, negatives)]


class KLU4EduRecStrictModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        edge_index: torch.Tensor,
        content_embeddings: torch.Tensor,
        embed_dim: int,
        n_layers: int,
        edge_drop: float,
        item_temperature: float,
        item_loss_reg: float,
        weight_decay: float,
        device: torch.device,
        source_dir: Path = DEFAULT_SOURCE_DIR,
    ) -> None:
        super().__init__()
        author_class = load_author_model_class(source_dir)
        content_on_device = content_embeddings.to(device=device, dtype=torch.float32)
        config = SimpleNamespace(
            n_users=int(n_users),
            m_items=int(n_items),
            embed_size=int(embed_dim),
            n_layers=int(n_layers),
            node_drop=0.0,
            edge_drop=float(edge_drop),
            mode="item_se",
            add_self_loops=True,
            item_fusion_func="gating",
            item_temp=float(item_temperature),
            item_loss_reg=float(item_loss_reg),
            user_segments_type="allseq",
            user_fusion_func="gating",
            user_temp=0.1,
            user_loss_reg=0.0,
            wd=float(weight_decay),
            pretrained_item_embeddings=content_on_device,
            item_LLM_emb_dim=int(content_on_device.shape[1]),
        )
        self.author_model = author_class(config, edge_index.to(device), device).to(device)
        # The released constructor reinitializes every Embedding, including the frozen pretrained table.
        with torch.no_grad():
            self.author_model.item_LLM_embedding.weight.copy_(content_on_device)
        self.author_model.item_LLM_embedding.weight.requires_grad_(False)
        self._cached_users: torch.Tensor | None = None
        self._cached_items: torch.Tensor | None = None

    def calculate_loss(self, users: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        return self.author_model.calculate_loss(users, positives, negatives)

    def prepare_catalog_scoring(self) -> None:
        users, items, *_ = self.author_model.forward()
        self._cached_users = users.detach()
        self._cached_items = items.detach()

    def clear_catalog_scoring(self) -> None:
        self._cached_users = None
        self._cached_items = None

    def score_catalog(self, user: int, item_ids: torch.Tensor) -> torch.Tensor:
        if self._cached_users is None or self._cached_items is None:
            raise RuntimeError("prepare_catalog_scoring must be called before scoring")
        user_vector = self._cached_users[int(user)]
        return torch.sum(self._cached_items[item_ids] * user_vector.unsqueeze(0), dim=1)

    def score_user_batch(self, users: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        if self._cached_users is None or self._cached_items is None:
            raise RuntimeError("prepare_catalog_scoring must be called before scoring")
        return self._cached_users[users] @ self._cached_items[item_ids].t()


class ItemMacroRankingAccumulator:
    def __init__(self, k_list: Iterable[int] = (5, 10, 20), cold_threshold: int = 1) -> None:
        self.k_list = tuple(int(k) for k in k_list)
        self.cold_threshold = int(cold_threshold)
        self.item_sum = {
            "cold": {f"{metric}@{k}": defaultdict(float) for metric in ("R", "N") for k in self.k_list},
            "hot": {f"{metric}@{k}": defaultdict(float) for metric in ("R", "N") for k in self.k_list},
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
        masked = np.asarray(scores, dtype=np.float32).copy()
        for row_index, example in enumerate(examples):
            target = int(example["target"])
            target_score = float(masked[row_index, target])
            seen = [item for item in user_seen_items.get(int(example["user"]), set()) if 0 <= item < masked.shape[1]]
            if seen:
                masked[row_index, seen] = -np.inf
            masked[row_index, target] = target_score

        max_k = min(max(self.k_list), masked.shape[1])
        order = np.argsort(-masked, axis=1)[:, :max_k]
        for row_index, example in enumerate(examples):
            target = int(example["target"])
            raw_item = int(example["raw_item"])
            group = "cold" if int(example.get("popularity", 0)) < self.cold_threshold else "hot"
            self.item_count[group][raw_item] += 1
            self.rows[group] += 1
            for k in self.k_list:
                positions = np.where(order[row_index, : min(k, order.shape[1])] == target)[0]
                hit = 1.0 if positions.size else 0.0
                ndcg = 1.0 / math.log2(float(positions[0]) + 2.0) if hit else 0.0
                self.item_sum[group][f"R@{k}"][raw_item] += hit
                self.item_sum[group][f"N@{k}"][raw_item] += ndcg

    def _group_result(self, group: str) -> tuple[dict[str, float], int]:
        counts = self.item_count[group]
        if not counts:
            return {}, 0
        result: dict[str, float] = {}
        for key, per_item in self.item_sum[group].items():
            values = [float(per_item[item]) / count for item, count in counts.items() if count]
            result[key] = float(np.mean(values)) if values else 0.0
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


def evaluate_split(
    model: KLU4EduRecStrictModel,
    rows: Iterable[tuple[int, int]],
    structures: TrainingStructures,
    n_items: int,
    max_users: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    targets_by_user: dict[int, list[int]] = defaultdict(list)
    for user, item in rows:
        targets_by_user[int(user)].append(int(item))
    users = sorted(targets_by_user)
    if max_users >= 0:
        users = users[:max_users]
    if not users:
        raise ValueError("evaluation has no users after applying max_users")

    model.eval()
    item_ids = torch.arange(n_items, dtype=torch.long, device=device)
    accumulator = ItemMacroRankingAccumulator(k_list=(5, 10, 20), cold_threshold=1)
    score_chunks: list[np.ndarray] = []
    model.prepare_catalog_scoring()
    try:
        with torch.no_grad():
            for start in range(0, len(users), batch_size):
                batch_users = users[start : start + batch_size]
                user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
                batch_scores = model.score_user_batch(user_tensor, item_ids).cpu().numpy().astype(np.float32, copy=False)
                score_chunks.append(batch_scores)
                for row_index, user in enumerate(batch_users):
                    targets = targets_by_user[user]
                    examples = [
                        {"user": user, "target": target, "raw_item": target, "popularity": 0}
                        for target in targets
                    ]
                    repeated = np.repeat(batch_scores[row_index : row_index + 1], len(targets), axis=0)
                    accumulator.add_batch(repeated, examples, structures.user_items)
    finally:
        model.clear_catalog_scoring()

    report = accumulator.result()
    report.update(
        {
            "evaluated_users": len(users),
            "candidate_courses": int(n_items),
            "score_std": float(np.std(np.concatenate(score_chunks, axis=0))),
        }
    )
    return report


def resolve_torch_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return torch.device(requested)


def configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_one_epoch(
    model: KLU4EduRecStrictModel,
    triples: list[tuple[int, int, int]],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float | int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.train()
    losses: list[float] = []
    for start in range(0, len(triples), batch_size):
        batch = triples[start : start + batch_size]
        users = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
        positives = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        negatives = torch.tensor([row[2] for row in batch], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.calculate_loss(users, positives, negatives)
        if not torch.isfinite(loss):
            raise FloatingPointError("KLU4EduRec produced a non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)), "batches": len(losses), "examples": len(triples)}


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _source_hashes(source_dir: Path) -> dict[str, str]:
    relative_paths = ("main.py", "model/GNNRec.py", "model/dataset.py", "model/metric.py", "readme.md")
    return {
        relative: hashlib.sha256((source_dir / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _write_report(report: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "klu4edurec_strict_adapter_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_klu4edurec_strict_adapter(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    configure_reproducibility(args.seed)
    device = resolve_torch_device(args.device)
    source_dir = resolve_source_dir(args.source_dir)
    dataset = load_strict_dataset(args.split_root, args.content_path, args.expected_courses)
    structures = build_train_structures(
        dataset.train_positives,
        dataset.n_users,
        dataset.n_items,
        dataset.cold_items,
    )
    model = KLU4EduRecStrictModel(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        edge_index=structures.edge_index,
        content_embeddings=dataset.content_embeddings,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        edge_drop=args.edge_drop,
        item_temperature=args.item_temperature,
        item_loss_reg=args.item_loss_reg,
        weight_decay=args.weight_decay,
        device=device,
        source_dir=source_dir,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    tracker = BestValidationTracker(metric_path="full_cold_item_macro.N@10", patience=args.patience)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, object]] = []
    all_negatives: list[tuple[int, int]] = []

    for epoch in range(1, args.epochs + 1):
        triples = prepare_epoch_triples(dataset.train_positives, structures, args.max_train_examples, rng)
        all_negatives.extend((user, negative) for user, _, negative in triples)
        train_stats = train_one_epoch(model, triples, args.batch_size, optimizer, device)
        validation = evaluate_split(
            model,
            dataset.validation_rows,
            structures,
            dataset.n_items,
            args.max_eval_users,
            args.eval_batch_size,
            device,
        )
        improved = tracker.update(epoch, validation, _cpu_state_dict(model))
        history.append({"epoch": epoch, "train": train_stats, "validation": validation, "improved": improved})
        print(
            f"epoch={epoch} loss={train_stats['loss']:.6f} "
            f"val_N@10={validation['full_cold_item_macro'].get('N@10', 0.0):.8f}"
            f"{' best' if improved else ''}",
            flush=True,
        )
        if tracker.should_stop:
            break

    if tracker.best_state is None or tracker.best_report is None:
        raise RuntimeError("validation did not produce a checkpoint")
    model.load_state_dict(tracker.best_state)
    validation = tracker.best_report
    test = evaluate_split(
        model,
        dataset.test_rows,
        structures,
        dataset.n_items,
        args.max_eval_users,
        args.eval_batch_size,
        device,
    )

    losses = [float(row["train"]["loss"]) for row in history]
    cold_content = dataset.content_embeddings[list(sorted(dataset.cold_items))]
    gates = {
        "device_request_satisfied": args.device != "cuda" or device.type == "cuda",
        "author_source_class_loaded": model.author_model.__class__.__module__.startswith("klu_author"),
        "source_mode_item_se": model.author_model.mode == "item_se",
        "content_catalog_aligned": dataset.content_embeddings.shape[0] == dataset.n_items,
        "cold_semantics_nonconstant": cold_content.shape[0] > 1 and float(cold_content.std()) > 1e-8,
        "nonempty_train_graph": structures.positive_edges > 0 and structures.edge_index.numel() > 0,
        "warm_only_negatives": all(
            item in structures.warm_items and item not in structures.user_items.get(user, set())
            for user, item in all_negatives
        ),
        "finite_loss": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "nonconstant_validation_scores": float(validation["score_std"]) > 1e-8,
        "nonconstant_test_scores": float(test["score_std"]) > 1e-8,
        "validation_has_cold_courses": int(validation["count_full_cold_item_macro"]) > 0,
        "test_has_cold_courses": int(test["count_full_cold_item_macro"]) > 0,
        "no_positive_cold_train_edges": not ({item for _, item in dataset.train_positives} & dataset.cold_items),
        "full_catalog_ranking": int(validation["candidate_courses"]) == dataset.n_items,
        "train_history_masked": True,
        "item_macro_metrics": True,
    }
    report: dict[str, object] = {
        "model": "KLU4EduRec-item_se (author source, strict adapter)",
        "status": "smoke_passed" if all(gates.values()) else "smoke_failed",
        "seed": int(args.seed),
        "device": str(device),
        "source_dir": str(source_dir),
        "source_commit": AUDITED_SOURCE_COMMIT,
        "source_hashes": _source_hashes(source_dir),
        "split_root": str(Path(args.split_root).resolve()),
        "content_path": str(Path(args.content_path).resolve()),
        "best_epoch": int(tracker.best_epoch),
        "checkpoint_metric": "validation.full_cold_item_macro.N@10",
        "protocol": {
            "source_mode": "item_se",
            "full_model_claimed": False,
            "pretrained_semantics_restored_after_author_init": True,
            "adaptation_scope": (
                "Official item-semantic mode only; author precomputed user summaries cannot be mapped to the strict users."
            ),
            "positive_train_edges": structures.positive_edges,
            "warm_courses": len(structures.warm_items),
            "cold_courses": len(dataset.cold_items),
            "candidate_courses": dataset.n_items,
            "content_shape": list(dataset.content_embeddings.shape),
            "full_catalog_ranking": True,
            "train_history_masked": True,
            "item_macro": True,
            "warm_only_training_negatives": True,
            "validation_checkpointing": True,
            "expected_course_catalog": args.expected_courses,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "bitwise_cuda_reproducibility_verified": False,
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
        },
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "embed_dim": args.embed_dim,
            "n_layers": args.n_layers,
            "edge_drop": args.edge_drop,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "item_temperature": args.item_temperature,
            "item_loss_reg": args.item_loss_reg,
            "max_train_examples": args.max_train_examples,
            "max_eval_users": args.max_eval_users,
        },
        "history": history,
        "validation": validation,
        "test": test,
        "gates": gates,
        "elapsed_seconds": time.time() - started,
    }
    _write_report(report, Path(args.out_dir))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict item-cold adapter for author KLU4EduRec item-semantic mode")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--content-path", type=Path, default=DEFAULT_CONTENT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--expected-courses", type=int, default=698)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--edge-drop", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--item-temperature", type=float, default=0.1)
    parser.add_argument("--item-loss-reg", type=float, default=1e-4)
    parser.add_argument("--max-train-examples", type=int, default=100000)
    parser.add_argument("--max-eval-users", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.expected_courses <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("course count, epochs, and batch sizes must be positive")
    if args.patience < 0:
        parser.error("patience must be nonnegative")
    if args.max_train_examples == 0 or args.max_train_examples < -1 or args.max_eval_users < -1:
        parser.error("example caps must be -1 or positive")
    return args


def main() -> None:
    report = run_klu4edurec_strict_adapter(parse_args())
    summary = {
        "model": report["model"],
        "status": report["status"],
        "best_epoch": report["best_epoch"],
        "device": report["device"],
        "elapsed_seconds": report["elapsed_seconds"],
        "validation": report["validation"]["full_cold_item_macro"],
        "test": report["test"]["full_cold_item_macro"],
        "gates": report["gates"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
