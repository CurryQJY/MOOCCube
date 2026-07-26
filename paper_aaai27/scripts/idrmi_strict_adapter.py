from __future__ import annotations

import argparse
import heapq
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
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA_DIR = (
    ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_prepared"
    / "mooccube_seed2025"
    / "idrmi"
    / "Data"
    / "moocCube"
)
DEFAULT_SOURCE_DIR = ROOT / "paper_aaai27" / "baseline_sources" / "IDRMI"
DEFAULT_SPLIT_ROOT = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "static_item_cold_balanced"
    / "strict_item_cold_balanced_thr1_seed_2025"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "paper_aaai27"
    / "baseline_sources"
    / "_idrmi_strict"
    / "mooccube_seed2025_smoke"
)

from paper_aaai27.scripts.pcgnn_strict_adapter import (  # noqa: E402
    BestValidationTracker,
)


_SOURCE_CLASS_CACHE: dict[Path, tuple[type, type]] = {}


def _load_class_from_file(module_name: str, path: Path, class_name: str, dummy_modules: tuple[str, ...] = ()):
    previous = {name: sys.modules.get(name) for name in dummy_modules}
    try:
        for name in dummy_modules:
            sys.modules[name] = types.ModuleType(name)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load IDRMI source module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def load_idrmi_source_classes(source_dir: Path) -> tuple[type, type]:
    source_dir = Path(source_dir).resolve()
    if source_dir not in _SOURCE_CLASS_CACHE:
        ngcf_class = _load_class_from_file(
            "idrmi_author_ngcf",
            source_dir / "NGCF" / "NGCF.py",
            "NGCF",
        )
        kgcn_class = _load_class_from_file(
            "idrmi_author_kgcn",
            source_dir / "KGCN.py",
            "KGCN",
            dummy_modules=("dataloader4kg",),
        )
        _SOURCE_CLASS_CACHE[source_dir] = (ngcf_class, kgcn_class)
    return _SOURCE_CLASS_CACHE[source_dir]


SourceNGCF, SourceKGCN = load_idrmi_source_classes(DEFAULT_SOURCE_DIR)


class DeviceKGCN(SourceKGCN):
    def forward(self, users, items, is_evaluate: bool = False):
        import torch
        import torch.nn.functional as functional

        device = self.user_embedding.weight.device
        users = users.to(device=device, dtype=torch.long)
        items = items.to(device=device, dtype=torch.long)
        item_numpy = items.detach().cpu().numpy()
        neighbor_ids = torch.as_tensor(np.asarray(self.adj_entity)[item_numpy], dtype=torch.long, device=device)
        relation_ids = torch.as_tensor(np.asarray(self.adj_relation)[item_numpy], dtype=torch.long, device=device)

        user_embeddings = self.user_embedding(users)
        item_embeddings = self.entity_embedding(items)
        neighbor_entities = self.entity_embedding(neighbor_ids)
        neighbor_relations = self.relation_embedding(relation_ids)
        repeated_users = user_embeddings.unsqueeze(1).expand(-1, self.n_neighbors, -1)
        relation_scores = torch.sum(repeated_users * neighbor_relations, dim=2)
        relation_weights = functional.softmax(relation_scores, dim=-1).unsqueeze(2)
        neighbor_vectors = torch.sum(relation_weights * neighbor_entities, dim=1)
        return self.aggregator(item_embeddings, neighbor_vectors, is_evaluate)


class StrictNGCF(SourceNGCF):
    def _convert_sp_mat_to_sp_tensor(self, matrix):
        coo = matrix.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64, copy=False))
        values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
        return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()

    def sparse_dropout(self, values, rate, noise_shape):
        random_tensor = 1.0 - float(rate) + torch.rand(noise_shape, device=values.device)
        keep = torch.floor(random_tensor).to(torch.bool)
        output = torch.sparse_coo_tensor(
            values.indices()[:, keep],
            values.values()[keep],
            size=values.shape,
            device=values.device,
        ).coalesce()
        return output * (1.0 / (1.0 - float(rate)))

    def propagate_all(self, drop_flag: bool) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = (
            self.sparse_dropout(self.sparse_norm_adj, self.node_dropout, self.sparse_norm_adj._nnz())
            if drop_flag
            else self.sparse_norm_adj
        )
        embeddings = torch.cat([self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], dim=0)
        layers = [embeddings]
        for layer_idx in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(adjacency, embeddings)
            embeddings = (
                torch.matmul(side_embeddings, self.weight_dict[f"W_gc_{layer_idx}"])
                + self.weight_dict[f"b_gc_{layer_idx}"]
            )
            embeddings = functional.leaky_relu(embeddings, negative_slope=0.2)
            embeddings = functional.dropout(
                embeddings,
                p=float(self.mess_dropout[layer_idx]),
                training=self.training,
            )
            layers.append(functional.normalize(embeddings, p=2, dim=1))

        combined = torch.stack(layers, dim=0).sum(dim=0)
        return combined[: self.n_user], combined[self.n_user :]

    def forward(self, users, drop_flag: bool = True):
        all_users, _ = self.propagate_all(drop_flag=drop_flag)
        return all_users[users]


def build_kg_neighbors(
    triples: Iterable[tuple[int, int, int]],
    n_entities: int,
    n_neighbors: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if n_entities <= 0 or n_neighbors <= 0:
        raise ValueError("n_entities and n_neighbors must be positive")
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for head, relation, tail in triples:
        head = int(head)
        relation = int(relation)
        tail = int(tail)
        if not (0 <= head < n_entities and 0 <= tail < n_entities):
            raise ValueError("KG entity id is outside n_entities")
        graph[head].append((tail, relation))
        graph[tail].append((head, relation))

    adjacent_entities = np.zeros((n_entities, n_neighbors), dtype=np.int64)
    adjacent_relations = np.zeros((n_entities, n_neighbors), dtype=np.int64)
    for entity in range(n_entities):
        neighbors = graph.get(entity)
        if not neighbors:
            neighbors = [(entity, 0)]
        indices = rng.choice(len(neighbors), size=n_neighbors, replace=len(neighbors) < n_neighbors)
        adjacent_entities[entity] = [neighbors[int(index)][0] for index in indices]
        adjacent_relations[entity] = [neighbors[int(index)][1] for index in indices]
    return adjacent_entities, adjacent_relations


class IDRMIStrictModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_entities: int,
        n_relations: int,
        norm_adj: sp.csr_matrix,
        adj_entity: np.ndarray,
        adj_relation: np.ndarray,
        user_items: dict[int, set[int]],
        item_users: dict[int, set[int]],
        embed_dim: int,
        n_neighbors: int,
        batch_size: int,
        device: torch.device,
        ngcf_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        ngcf_device = device if ngcf_device is None else ngcf_device
        source_args = SimpleNamespace(
            device=ngcf_device,
            embed_size=int(embed_dim),
            batch_size=int(batch_size),
            node_dropout=[0.1],
            mess_dropout=[0.1, 0.1, 0.1],
            layer_size=str([int(embed_dim)] * 3),
            regs="[1e-5]",
        )
        self.ngcf = StrictNGCF(n_users, n_items, norm_adj, source_args)
        self.kgcn = DeviceKGCN(
            n_users,
            n_entities,
            n_relations,
            adj_entity,
            adj_relation,
            n_neighbors=n_neighbors,
            e_dim=embed_dim,
            aggregator_method="sum",
            act_method=functional.relu,
            drop_rate=0.0,
        )
        self.user_items = user_items
        self.item_users = item_users
        self.n_items = int(n_items)
        self.primary_device = device
        self.ngcf_device = ngcf_device
        self.register_buffer(
            "history_matrix",
            build_history_tensor(user_items, n_users, n_items, device),
            persistent=False,
        )
        self.register_buffer(
            "course_match_table",
            build_course_match_table(item_users, n_users, n_items, device),
            persistent=False,
        )
        self._cached_user_vectors: torch.Tensor | None = None
        self.ngcf.to(ngcf_device)
        self.kgcn.to(device)

    def forward(self, users, items, training_factors: bool = True):
        primary_users = users.to(self.primary_device, dtype=torch.long)
        items = items.to(self.primary_device, dtype=torch.long)
        ngcf_users = users.to(self.ngcf_device, dtype=torch.long)
        user_vectors = self.ngcf(ngcf_users, drop_flag=bool(self.training)).to(self.primary_device)
        item_vectors = self.kgcn(primary_users, items, is_evaluate=not self.training)
        factors = gpu_interest_factors(
            primary_users,
            items,
            user_vectors,
            item_vectors,
            self.history_matrix,
            self.course_match_table,
        )
        return source_fusion_score(user_vectors, item_vectors, factors)

    def score_catalog(self, user: int, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.to(device=self.primary_device, dtype=torch.long)
        if self._cached_user_vectors is None:
            all_users, _ = self.ngcf.propagate_all(drop_flag=False)
        else:
            all_users = self._cached_user_vectors
        user_vector = all_users[int(user)].to(self.primary_device).unsqueeze(0).expand(item_ids.shape[0], -1)
        repeated_users = torch.full_like(item_ids, int(user))
        item_vectors = self.kgcn(repeated_users, item_ids, is_evaluate=True)
        factors = gpu_interest_factors(
            repeated_users,
            item_ids,
            user_vector,
            item_vectors,
            self.history_matrix,
            self.course_match_table,
        )
        return source_fusion_score(user_vector, item_vectors, factors)

    def prepare_catalog_scoring(self) -> None:
        self.eval()
        with torch.no_grad():
            self._cached_user_vectors, _ = self.ngcf.propagate_all(drop_flag=False)

    def clear_catalog_scoring(self) -> None:
        self._cached_user_vectors = None


@dataclass(frozen=True)
class TrainingStructures:
    norm_adj: sp.csr_matrix
    user_items: dict[int, set[int]]
    item_users: dict[int, set[int]]
    warm_items: tuple[int, ...]
    positive_edges: int


@dataclass(frozen=True)
class StrictDataset:
    train_positives: tuple[tuple[int, int], ...]
    validation_rows: tuple[tuple[int, int], ...]
    test_rows: tuple[tuple[int, int], ...]
    kg_triples: tuple[tuple[int, int, int], ...]
    cold_items: frozenset[int]
    n_users: int
    n_items: int
    n_entities: int
    n_relations: int


def _read_kg_triples(path: Path) -> tuple[tuple[int, int, int], ...]:
    triples: list[tuple[int, int, int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"invalid KG row at {path}:{line_number}")
            triples.append(tuple(int(value) for value in parts))
    if not triples:
        raise ValueError("KG file is empty")
    return tuple(triples)


def load_strict_dataset(
    split_root: Path,
    kg_path: Path,
    expected_n_items: int | None = None,
) -> StrictDataset:
    import pandas as pd

    split_root = Path(split_root)
    train = pd.read_pickle(split_root / "static_train.pkl")
    validation = pd.read_pickle(split_root / "static_val.pkl")
    test = pd.read_pickle(split_root / "static_test.pkl")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        missing = {"u_idx", "i_idx"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split is missing columns {sorted(missing)}")

    validation_cold = validation[validation["_split_source"].eq("strict_item_cold_val")]
    test_cold = test[test["_split_source"].eq("strict_item_cold_test")]
    train_positives = tuple(
        (int(row.u_idx), int(row.i_idx)) for row in train[["u_idx", "i_idx"]].itertuples(index=False)
    )
    validation_rows = tuple(
        (int(row.u_idx), int(row.i_idx))
        for row in validation_cold[["u_idx", "i_idx"]].itertuples(index=False)
    )
    test_rows = tuple(
        (int(row.u_idx), int(row.i_idx)) for row in test_cold[["u_idx", "i_idx"]].itertuples(index=False)
    )
    if not train_positives or not validation_rows or not test_rows:
        raise ValueError("strict train, validation, and test rows must all be nonempty")

    cold_items = frozenset(item for _, item in validation_rows + test_rows)
    leaked_train_items = {item for _, item in train_positives} & cold_items
    if leaked_train_items:
        raise ValueError(f"positive train rows contain cold courses: {sorted(leaked_train_items)[:10]}")

    interactions = train_positives + validation_rows + test_rows
    n_users = max(user for user, _ in interactions) + 1
    observed_items = {item for _, item in interactions}
    if expected_n_items is None:
        n_items = max(observed_items) + 1
    else:
        if expected_n_items <= 0:
            raise ValueError("expected_n_items must be positive")
        expected_items = set(range(int(expected_n_items)))
        missing_items = expected_items - observed_items
        outside_items = observed_items - expected_items
        if missing_items or outside_items:
            raise ValueError(
                "expected course catalog does not match strict interactions: "
                f"missing={sorted(missing_items)[:10]} outside={sorted(outside_items)[:10]}"
            )
        n_items = int(expected_n_items)
    kg_triples = _read_kg_triples(Path(kg_path))
    n_entities = max(max(head, tail) for head, _, tail in kg_triples) + 1
    n_relations = max(relation for _, relation, _ in kg_triples) + 1
    if n_entities < n_items:
        raise ValueError("KG entity space does not cover the full course catalog")

    return StrictDataset(
        train_positives=train_positives,
        validation_rows=validation_rows,
        test_rows=test_rows,
        kg_triples=kg_triples,
        cold_items=cold_items,
        n_users=n_users,
        n_items=n_items,
        n_entities=n_entities,
        n_relations=n_relations,
    )


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
        if not self.k_list:
            return

        masked = np.asarray(scores, dtype=np.float32).copy()
        target_scores: list[float] = []
        for row_idx, example in enumerate(examples):
            target = int(example["target"])
            target_scores.append(float(masked[row_idx, target]))
            for seen in user_seen_items.get(int(example["user"]), set()):
                if 0 <= int(seen) < masked.shape[1]:
                    masked[row_idx, int(seen)] = -np.inf
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
                predictions = order[row_idx, : min(k, order.shape[1])]
                positions = np.where(predictions == target)[0]
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


def build_train_structures(
    rows: Iterable[tuple[int, int, int]],
    n_users: int,
    n_items: int,
    forbidden_cold_items: set[int] | frozenset[int],
) -> TrainingStructures:
    user_items: dict[int, set[int]] = defaultdict(set)
    item_users: dict[int, set[int]] = defaultdict(set)
    positive_pairs: set[tuple[int, int]] = set()

    for user, item, label in rows:
        user = int(user)
        item = int(item)
        if not (0 <= user < n_users and 0 <= item < n_items):
            raise ValueError(f"interaction ({user}, {item}) is outside declared cardinalities")
        if int(label) != 1:
            continue
        if item in forbidden_cold_items:
            raise ValueError(f"positive train edge contains held-out cold course {item}")
        positive_pairs.add((user, item))
        user_items[user].add(item)
        item_users[item].add(user)

    if not positive_pairs:
        raise ValueError("positive training graph is empty")

    interaction = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for user, item in positive_pairs:
        interaction[user, item] = 1.0

    adjacency = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32).tolil()
    interaction_lil = interaction.tolil()
    adjacency[:n_users, n_users:] = interaction_lil
    adjacency[n_users:, :n_users] = interaction_lil.T
    adjacency = adjacency.tocsr()

    source_with_self_loops = adjacency + sp.eye(adjacency.shape[0], dtype=np.float32, format="csr")
    row_sum = np.asarray(source_with_self_loops.sum(axis=1)).reshape(-1)
    inv_degree = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inv_degree[nonzero] = 1.0 / row_sum[nonzero]
    norm_adj = sp.diags(inv_degree).dot(source_with_self_loops).tocsr()

    return TrainingStructures(
        norm_adj=norm_adj,
        user_items={user: set(items) for user, items in user_items.items()},
        item_users={item: set(users) for item, users in item_users.items()},
        warm_items=tuple(sorted(item_users)),
        positive_edges=len(positive_pairs),
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
    for user, _ in positives:
        user = int(user)
        if user not in candidates_by_user:
            seen = user_items.get(user, set())
            candidates_by_user[user] = tuple(item for item in warm if item not in seen)
        candidates = candidates_by_user[user]
        if not candidates:
            raise ValueError(f"user {user} has no eligible warm negative")
        negatives.append((user, int(candidates[int(rng.integers(0, len(candidates)))])))
    return negatives


def prepare_labeled_training_rows(
    positives: Iterable[tuple[int, int]],
    structures: TrainingStructures,
    max_examples: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    positive_rows = list((int(user), int(item)) for user, item in positives)
    if not positive_rows:
        raise ValueError("training positives are empty")
    if max_examples == 0 or max_examples == 1:
        raise ValueError("max_examples must be negative or at least 2")
    positive_limit = len(positive_rows) if max_examples < 0 else min(len(positive_rows), max_examples // 2)
    if positive_limit < len(positive_rows):
        selected = rng.choice(len(positive_rows), size=positive_limit, replace=False)
        chosen_positives = [positive_rows[int(index)] for index in selected]
    else:
        chosen_positives = positive_rows
    negatives = sample_warm_negatives(
        chosen_positives,
        structures.warm_items,
        structures.user_items,
        rng,
    )
    rows = [(user, item, 1) for user, item in chosen_positives]
    rows.extend((user, item, 0) for user, item in negatives)
    rng.shuffle(rows)
    return rows


def source_fusion_score(user_vectors, item_vectors, interest_factors):
    import torch

    if user_vectors.shape != item_vectors.shape:
        raise ValueError("user and item vectors must have identical shapes")
    if interest_factors.ndim != 2 or interest_factors.shape != (user_vectors.shape[0], 3):
        raise ValueError("interest_factors must have shape [batch, 3]")

    learned_score = torch.sigmoid(torch.sum(user_vectors * item_vectors, dim=-1))
    raw_interest = torch.mean(interest_factors.to(learned_score), dim=1)
    minimum = torch.min(raw_interest)
    maximum = torch.max(raw_interest)
    interest_weight = 0.5 + (raw_interest - minimum) / (maximum - minimum + 1e-5)
    return torch.tanh(learned_score * interest_weight)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def source_interest_factors(
    user_ids,
    item_ids,
    user_vectors,
    item_vectors,
    user_items: dict[int, set[int]],
    item_users: dict[int, set[int]],
):
    import torch

    users = [int(value) for value in user_ids.detach().cpu().tolist()]
    items = [int(value) for value in item_ids.detach().cpu().tolist()]
    user_numpy = user_vectors.detach().cpu().numpy()
    item_numpy = item_vectors.detach().cpu().numpy()
    batch_size = len(users)

    course_match: list[float] = []
    for user, candidate in zip(users, items):
        history = set(user_items.get(user, set()))
        history.discard(candidate)
        values: list[float] = []
        candidate_users = item_users.get(candidate, set())
        for historical_item in history:
            historical_users = item_users.get(historical_item, set())
            overlap = len(historical_users & candidate_users)
            q_history = overlap / (len(historical_users) + 1e-4)
            q_candidate = overlap / (len(candidate_users) + 1e-4)
            values.append((q_history + q_candidate) / 2.0)
        course_match.append(float(np.mean(values)) if values else 0.0)

    user_choice: list[float] = []
    for row in range(batch_size):
        similarities = [_cosine_similarity(user_numpy[row], user_numpy[other]) for other in range(batch_size)]
        nearest = heapq.nlargest(6, range(batch_size), key=similarities.__getitem__)
        support = 0.0
        for other in nearest:
            if items[row] in user_items.get(users[other], set()):
                support += similarities[other]
        user_choice.append(support / 6.0)

    course_preference: list[float] = []
    for row in range(batch_size):
        similarities = []
        for other in range(batch_size):
            similarity = 1.0 / (float(np.linalg.norm(item_numpy[row] - item_numpy[other])) + 1.0)
            similarities.append(0.0 if similarity == 1.0 else similarity)
        nearest = heapq.nlargest(3, range(batch_size), key=similarities.__getitem__)
        preference = 0.0
        for other in nearest:
            if users[row] in item_users.get(items[other], set()):
                preference += 1.0 / (float(np.linalg.norm(item_numpy[row] - item_numpy[other])) + 1.0)
        course_preference.append(preference / 3.0)

    factors = np.column_stack((course_match, user_choice, course_preference)).astype(np.float32, copy=False)
    return torch.as_tensor(factors, device=user_vectors.device, dtype=user_vectors.dtype)


def build_history_tensor(
    user_items: dict[int, set[int]],
    n_users: int,
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    history = torch.zeros((n_users, n_items), dtype=torch.bool, device=device)
    users: list[int] = []
    items: list[int] = []
    for user, selected_items in user_items.items():
        for item in selected_items:
            users.append(int(user))
            items.append(int(item))
    if users:
        history[
            torch.tensor(users, dtype=torch.long, device=device),
            torch.tensor(items, dtype=torch.long, device=device),
        ] = True
    return history


def build_course_match_table(
    item_users: dict[int, set[int]],
    n_users: int,
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    item_indices: list[int] = []
    user_indices: list[int] = []
    for item, selected_users in item_users.items():
        for user in selected_users:
            item_indices.append(int(item))
            user_indices.append(int(user))
    values = np.ones(len(item_indices), dtype=np.float32)
    incidence = sp.csr_matrix(
        (values, (item_indices, user_indices)),
        shape=(n_items, n_users),
        dtype=np.float32,
    )
    overlap = (incidence @ incidence.T).toarray().astype(np.float32, copy=False)
    counts = np.asarray(incidence.sum(axis=1), dtype=np.float32).reshape(-1)
    inverse = 1.0 / (counts + 1e-4)
    table = overlap * (inverse[:, None] + inverse[None, :]) * 0.5
    return torch.as_tensor(table, dtype=torch.float32, device=device)


def gpu_interest_factors(
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    history_matrix: torch.Tensor,
    course_match_table: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        batch_size = int(user_ids.shape[0])
        if batch_size == 0:
            return torch.empty((0, 3), dtype=user_vectors.dtype, device=user_vectors.device)

        histories = history_matrix[user_ids].clone()
        histories[torch.arange(batch_size, device=user_ids.device), item_ids] = False
        history_float = histories.to(course_match_table.dtype)
        history_counts = history_float.sum(dim=1)
        match_sums = (course_match_table[item_ids] * history_float).sum(dim=1)
        course_match = torch.where(
            history_counts > 0,
            match_sums / history_counts.clamp_min(1.0),
            torch.zeros_like(match_sums),
        )

        normalized_users = functional.normalize(user_vectors.detach(), p=2, dim=1)
        user_similarity = normalized_users @ normalized_users.T
        user_neighbor_count = min(6, batch_size)
        user_neighbors = torch.topk(user_similarity, k=user_neighbor_count, dim=1).indices
        neighbor_users = user_ids[user_neighbors]
        candidate_items = item_ids.unsqueeze(1).expand(-1, user_neighbor_count)
        selected_by_neighbors = history_matrix[neighbor_users, candidate_items]
        neighbor_similarity = torch.gather(user_similarity, 1, user_neighbors)
        user_choice = (
            neighbor_similarity * selected_by_neighbors.to(neighbor_similarity.dtype)
        ).sum(dim=1) / 6.0

        item_distance = torch.cdist(item_vectors.detach(), item_vectors.detach(), p=2)
        item_similarity = 1.0 / (item_distance + 1.0)
        item_ranking_scores = item_similarity.clone()
        item_ranking_scores[item_ranking_scores == 1.0] = 0.0
        item_neighbor_count = min(3, batch_size)
        item_neighbors = torch.topk(item_ranking_scores, k=item_neighbor_count, dim=1).indices
        neighbor_items = item_ids[item_neighbors]
        selected_by_user = history_matrix[user_ids.unsqueeze(1), neighbor_items]
        neighbor_item_similarity = torch.gather(item_similarity, 1, item_neighbors)
        course_preference = (
            neighbor_item_similarity * selected_by_user.to(neighbor_item_similarity.dtype)
        ).sum(dim=1) / 3.0

        return torch.stack((course_match, user_choice, course_preference), dim=1).to(user_vectors)


def catalog_interest_factors(
    user: int,
    item_ids: torch.Tensor,
    item_vectors: torch.Tensor,
    user_items: dict[int, set[int]],
    item_users: dict[int, set[int]],
) -> torch.Tensor:
    items = [int(item) for item in item_ids.detach().cpu().tolist()]
    history = set(user_items.get(int(user), set()))

    course_match: list[float] = []
    for candidate in items:
        candidate_history = history - {candidate}
        candidate_users = item_users.get(candidate, set())
        values: list[float] = []
        for historical_item in candidate_history:
            historical_users = item_users.get(historical_item, set())
            overlap = len(historical_users & candidate_users)
            values.append(
                (
                    overlap / (len(historical_users) + 1e-4)
                    + overlap / (len(candidate_users) + 1e-4)
                )
                / 2.0
            )
        course_match.append(float(np.mean(values)) if values else 0.0)

    support_scale = min(6, len(items)) / 6.0
    user_choice = [support_scale if candidate in history else 0.0 for candidate in items]

    with torch.no_grad():
        distances = torch.cdist(item_vectors.detach(), item_vectors.detach(), p=2)
        similarities = 1.0 / (distances + 1.0)
        ranking_scores = similarities.clone()
        ranking_scores[ranking_scores == 1.0] = 0.0
        neighbor_count = min(3, len(items))
        nearest = torch.topk(ranking_scores, k=neighbor_count, dim=1).indices
        item_tensor = item_ids.to(nearest.device)
        nearest_items = item_tensor[nearest]
        history_mask = torch.zeros_like(nearest_items, dtype=similarities.dtype)
        for historical_item in history:
            history_mask = torch.maximum(history_mask, (nearest_items == historical_item).to(similarities.dtype))
        preference = torch.gather(similarities, 1, nearest).mul(history_mask).sum(dim=1) / 3.0

    first = torch.as_tensor(course_match, device=item_vectors.device, dtype=item_vectors.dtype)
    second = torch.as_tensor(user_choice, device=item_vectors.device, dtype=item_vectors.dtype)
    return torch.stack((first, second, preference.to(item_vectors)), dim=1)


def evaluate_split(
    model,
    rows: Iterable[tuple[int, int]],
    structures: TrainingStructures,
    n_items: int,
    max_users: int,
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
    score_values: list[np.ndarray] = []
    if hasattr(model, "prepare_catalog_scoring"):
        model.prepare_catalog_scoring()
    try:
        with torch.no_grad():
            for user in users:
                scores = model.score_catalog(user, item_ids).detach().cpu().numpy().astype(np.float32, copy=False)
                if scores.shape != (n_items,):
                    raise ValueError("catalog scorer must return one score per course")
                score_values.append(scores)
                targets = targets_by_user[user]
                examples = [
                    {"user": user, "target": target, "raw_item": target, "popularity": 0}
                    for target in targets
                ]
                repeated_scores = np.repeat(scores.reshape(1, -1), len(targets), axis=0)
                accumulator.add_batch(repeated_scores, examples, structures.user_items)
    finally:
        if hasattr(model, "clear_catalog_scoring"):
            model.clear_catalog_scoring()

    report = accumulator.result()
    report.update(
        {
            "evaluated_users": len(users),
            "candidate_courses": int(n_items),
            "score_std": float(np.std(np.concatenate(score_values))),
        }
    )
    return report


def resolve_torch_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return torch.device(requested)


def resolve_source_dir(requested: Path) -> Path:
    resolved = Path(requested).resolve()
    expected = DEFAULT_SOURCE_DIR.resolve()
    if resolved != expected:
        raise ValueError(
            f"IDRMI execution is locked to the audited author source snapshot at {expected}; got {resolved}"
        )
    return resolved


def configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def train_one_epoch(
    model: IDRMIStrictModel,
    rows: list[tuple[int, int, int]],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float | int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.train()
    losses: list[float] = []
    score_chunks: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        users = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
        items = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        labels = torch.tensor([row[2] for row in batch], dtype=torch.float32, device=device)
        scores = model(users, items)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("IDRMI produced non-finite training scores")
        loss = functional.binary_cross_entropy(scores.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("IDRMI produced a non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        score_chunks.append(scores.detach().cpu().numpy())
    return {
        "loss": float(np.mean(losses)),
        "batches": len(losses),
        "train_score_std": float(np.std(np.concatenate(score_chunks))),
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _source_hashes(source_dir: Path) -> dict[str, str]:
    relative_files = (
        Path("NGCF") / "NGCF.py",
        Path("KGCN.py"),
        Path("course_match.py"),
        Path("user_choice.py"),
        Path("course_preference.py"),
    )
    hashes: dict[str, str] = {}
    for relative in relative_files:
        path = source_dir / relative
        hashes[str(relative).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _write_report(report: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "idrmi_strict_adapter_report.json"
    md_path = out_dir / "idrmi_strict_adapter_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    validation = report["validation"]
    test = report["test"]
    lines = [
        "# IDRMI Strict Adapter Report",
        "",
        f"- status: `{report['status']}`",
        f"- seed: `{report['seed']}`",
        f"- device: `{report['device']}`",
        f"- best epoch: `{report['best_epoch']}`",
        f"- positive train edges: `{report['protocol']['positive_train_edges']}`",
        f"- candidate courses: `{report['protocol']['candidate_courses']}`",
        "",
        "## Metrics",
        "",
        f"- validation: `{json.dumps(validation['full_cold_item_macro'], sort_keys=True)}`",
        f"- test: `{json.dumps(test['full_cold_item_macro'], sort_keys=True)}`",
        f"- validation score std: `{validation['score_std']:.8f}`",
        f"- test score std: `{test['score_std']:.8f}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in report["gates"].items())
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_idrmi_strict_adapter(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    configure_reproducibility(args.seed)
    device = resolve_torch_device(args.device)
    ngcf_device = device if args.ngcf_device == "same" else torch.device(args.ngcf_device)
    source_dir = resolve_source_dir(args.source_dir)

    dataset = load_strict_dataset(
        args.split_root,
        args.kg_path,
        expected_n_items=args.expected_courses,
    )
    structures = build_train_structures(
        ((user, item, 1) for user, item in dataset.train_positives),
        dataset.n_users,
        dataset.n_items,
        set(dataset.cold_items),
    )
    rng = np.random.default_rng(args.seed)
    training_rows = prepare_labeled_training_rows(
        dataset.train_positives,
        structures,
        args.max_train_examples,
        rng,
    )
    adjacent_entities, adjacent_relations = build_kg_neighbors(
        dataset.kg_triples,
        dataset.n_entities,
        args.n_neighbors,
        rng,
    )
    model = IDRMIStrictModel(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        norm_adj=structures.norm_adj,
        adj_entity=adjacent_entities,
        adj_relation=adjacent_relations,
        user_items=structures.user_items,
        item_users=structures.item_users,
        embed_dim=args.embed_dim,
        n_neighbors=args.n_neighbors,
        batch_size=args.batch_size,
        device=device,
        ngcf_device=ngcf_device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tracker = BestValidationTracker(metric_path="full_cold_item_macro.N@10", patience=args.patience)
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, training_rows, args.batch_size, optimizer, device)
        validation = evaluate_split(
            model,
            dataset.validation_rows,
            structures,
            dataset.n_items,
            args.max_eval_users,
            device,
        )
        improved = tracker.update(epoch, validation, _cpu_state_dict(model))
        history.append({"epoch": epoch, "train": train_stats, "validation": validation, "improved": improved})
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
        device,
    )

    negatives = [(user, item) for user, item, label in training_rows if label == 0]
    losses = [float(row["train"]["loss"]) for row in history]
    gates = {
        "cuda_selected": device.type == "cuda",
        "nonempty_adjacency": structures.positive_edges > 0 and structures.norm_adj.nnz > 0,
        "warm_only_negatives": all(
            item in structures.warm_items and item not in structures.user_items.get(user, set())
            for user, item in negatives
        ),
        "finite_loss": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "nonconstant_validation_scores": float(validation["score_std"]) > 1e-8,
        "nonconstant_test_scores": float(test["score_std"]) > 1e-8,
        "validation_has_cold_courses": int(validation["count_full_cold_item_macro"]) > 0,
        "test_has_cold_courses": int(test["count_full_cold_item_macro"]) > 0,
        "no_positive_cold_train_edges": not ({item for _, item in dataset.train_positives} & dataset.cold_items),
    }
    required_gates = [value for name, value in gates.items() if name != "cuda_selected" or args.device == "cuda"]
    status = "smoke_passed" if all(required_gates) else "smoke_failed"
    report: dict[str, object] = {
        "model": "IDRMI (author source, strict adapter)",
        "status": status,
        "seed": int(args.seed),
        "device": str(device),
        "ngcf_device": str(ngcf_device),
        "source_dir": str(source_dir),
        "source_hashes": _source_hashes(source_dir),
        "split_root": str(Path(args.split_root).resolve()),
        "kg_path": str(Path(args.kg_path).resolve()),
        "best_epoch": int(tracker.best_epoch),
        "checkpoint_metric": "validation.full_cold_item_macro.N@10",
        "protocol": {
            "positive_train_edges": structures.positive_edges,
            "training_rows_used": len(training_rows),
            "warm_courses": len(structures.warm_items),
            "cold_courses": len(dataset.cold_items),
            "candidate_courses": dataset.n_items,
            "kg_entities": dataset.n_entities,
            "kg_relations": dataset.n_relations,
            "train_history_masked": True,
            "full_catalog_ranking": True,
            "item_macro": True,
            "expected_course_catalog": args.expected_courses,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "bitwise_cuda_reproducibility_verified": False,
            "determinism_note": (
                "PyTorch deterministic algorithms are enabled, but repeated CUDA audits changed scores and some "
                "Top-K outcomes; treat single-seed CUDA values as non-bitwise-reproducible."
            ),
            "ngcf_sparse_propagation_device": str(ngcf_device),
            "interest_factor_backend": "torch_vectorized",
            "interest_factor_device": str(device),
            "history_tensor_mib": (
                model.history_matrix.numel() * model.history_matrix.element_size() / (1024.0 * 1024.0)
            ),
        },
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "embed_dim": args.embed_dim,
            "n_neighbors": args.n_neighbors,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
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
    torch.save(tracker.best_state, Path(args.out_dir) / "best_model.pt")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-faithful IDRMI on a strict course-cold split")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--kg-path", type=Path, default=DEFAULT_DATA_DIR / "kg_index.tsv")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--ngcf-device", choices=("cpu", "same"), default="same")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-examples", type=int, default=2048)
    parser.add_argument("--max-eval-users", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--expected-courses", type=int, default=698)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = run_idrmi_strict_adapter(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
