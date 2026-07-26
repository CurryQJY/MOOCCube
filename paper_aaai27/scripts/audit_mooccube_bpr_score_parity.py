"""Audit raw-dot versus cosine score parity for frozen MOOCCube BPR embeddings.

The audit deliberately does not consume or overwrite a main-table artifact.  It
uses only the immutable shared split, masks train history, and evaluates every
test target against the complete catalog.  The official ALDI BPR teacher stores
user and item vectors concatenated in one ``.npy`` file; the local matched BPR
trainer below stores the two matrices separately in ``.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
DEFAULT_SPLIT_DIR = Path(
    "outputs/content_delta_pop5/static_item_cold_balanced/"
    "strict_item_cold_balanced_thr1_seed_2025"
)
DEFAULT_OUTPUT_DIR = Path("outputs/score_parity/mooccube_seed2025")


@dataclass(frozen=True)
class FrozenSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    split_dir: Path
    split_sha256: Dict[str, str]


@dataclass(frozen=True)
class TeacherSamplingTables:
    warm_users: np.ndarray
    positive_items: np.ndarray
    positive_starts: np.ndarray
    positive_lengths: np.ndarray
    warm_items: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--meta-path", type=Path, default=Path("processed_data_hin_clean_pop5/meta.json"))
    parser.add_argument("--official-embedding", type=Path, required=True)
    parser.add_argument("--local-embedding", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cold-threshold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--train-local-bpr", action="store_true")
    parser.add_argument("--local-epochs", type=int, default=200)
    parser.add_argument("--local-emb-dim", type=int, default=200)
    parser.add_argument("--local-lr", type=float, default=1e-3)
    parser.add_argument("--local-reg-rate", type=float, default=1e-3)
    parser.add_argument("--local-seed", type=int, default=2025)
    parser.add_argument("--local-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--eval-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, label: str) -> None:
    missing = {"u_idx", "i_idx", "popularity"} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} split is missing columns: {sorted(missing)}")


def load_frozen_split(split_dir: Path, meta_path: Path) -> FrozenSplit:
    split_dir = split_dir.resolve()
    train_path = split_dir / "static_train.pkl"
    test_path = split_dir / "static_test.pkl"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen split artifact: {path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    train = pd.read_pickle(train_path).copy()
    test = pd.read_pickle(test_path).copy()
    _require_columns(train, "train")
    _require_columns(test, "test")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_users = int(meta["n_users"])
    n_items = int(meta["n_items"])

    for name, frame in (("train", train), ("test", test)):
        if frame.empty:
            raise ValueError(f"{name} split is empty")
        if frame["u_idx"].min() < 0 or frame["u_idx"].max() >= n_users:
            raise ValueError(f"{name} user IDs are outside metadata range")
        if frame["i_idx"].min() < 0 or frame["i_idx"].max() >= n_items:
            raise ValueError(f"{name} item IDs are outside metadata range")

    train_pairs = set(zip(train["u_idx"].astype(int), train["i_idx"].astype(int)))
    test_pairs = set(zip(test["u_idx"].astype(int), test["i_idx"].astype(int)))
    overlap = train_pairs & test_pairs
    if overlap:
        raise ValueError(
            "Frozen split has train/test user-item overlap; refusing parity audit. "
            f"Example={next(iter(overlap))}"
        )

    return FrozenSplit(
        train=train,
        test=test,
        n_users=n_users,
        n_items=n_items,
        split_dir=split_dir,
        split_sha256={
            "static_train.pkl": _sha256(train_path),
            "static_test.pkl": _sha256(test_path),
        },
    )


def build_train_seen(train_df: pd.DataFrame) -> Dict[int, set[int]]:
    seen: Dict[int, set[int]] = {}
    for user_id, item_id in zip(train_df["u_idx"].to_numpy(), train_df["i_idx"].to_numpy()):
        seen.setdefault(int(user_id), set()).add(int(item_id))
    return seen


def build_teacher_sampling_tables(train_df: pd.DataFrame) -> TeacherSamplingTables:
    """Build compact train-only lookup tables for official-style BPR sampling."""
    warm_users = []
    positive_parts = []
    positive_starts = []
    positive_lengths = []
    offset = 0
    for user_id, group in train_df.groupby("u_idx", sort=True):
        positive = group["i_idx"].to_numpy(dtype=np.int64, copy=True)
        if positive.size == 0:
            continue
        warm_users.append(int(user_id))
        positive_parts.append(positive)
        positive_starts.append(offset)
        positive_lengths.append(len(positive))
        offset += len(positive)
    if not positive_parts:
        raise ValueError("Cannot sample BPR triplets from an empty train split")
    return TeacherSamplingTables(
        warm_users=np.asarray(warm_users, dtype=np.int64),
        positive_items=np.concatenate(positive_parts),
        positive_starts=np.asarray(positive_starts, dtype=np.int64),
        positive_lengths=np.asarray(positive_lengths, dtype=np.int64),
        warm_items=np.unique(train_df["i_idx"].to_numpy(dtype=np.int64, copy=False)),
    )


def sample_teacher_triplets(
    *,
    tables: TeacherSamplingTables,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample BPR triplets with ALDI's train-only user/positive/item policy."""
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    if tables.warm_users.size == 0 or tables.warm_items.size == 0:
        raise ValueError("Cannot sample BPR triplets without train-visible users and items")
    user_rows = rng.integers(0, tables.warm_users.size, size=n_pairs)
    users = tables.warm_users[user_rows]
    positive_lengths = tables.positive_lengths[user_rows]
    positive_offsets = (rng.random(n_pairs) * positive_lengths).astype(np.int64)
    positives = tables.positive_items[tables.positive_starts[user_rows] + positive_offsets]
    negatives = rng.choice(tables.warm_items, size=n_pairs, replace=True)
    return users, positives, negatives


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def score_matrix(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    score_mode: str,
) -> np.ndarray:
    """Return a score matrix using the requested rank-equivalent score function."""
    if score_mode == "raw_dot":
        return np.asarray(user_embeddings, dtype=np.float32) @ np.asarray(item_embeddings, dtype=np.float32).T
    if score_mode == "cosine":
        return _l2_normalize(user_embeddings) @ _l2_normalize(item_embeddings).T
    raise ValueError(f"Unsupported score mode: {score_mode}")


def _metric_values(top_indices: np.ndarray, targets: np.ndarray, k_list: Sequence[int]) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for k in k_list:
        top_k = top_indices[:, :k]
        hit_positions = top_k == targets[:, None]
        hit = hit_positions.any(axis=1)
        rank = hit_positions.argmax(axis=1)
        result[f"R@{k}"] = hit.astype(np.float64)
        ndcg = np.zeros(targets.shape[0], dtype=np.float64)
        ndcg[hit] = 1.0 / np.log2(rank[hit].astype(np.float64) + 2.0)
        result[f"N@{k}"] = ndcg
    return result


def _metric_names(k_list: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"{prefix}@{int(k)}" for prefix in ("R", "N") for k in k_list)


def _top_indices(scores: np.ndarray, max_k: int) -> np.ndarray:
    if max_k < 1:
        raise ValueError("max_k must be positive")
    if max_k > scores.shape[1]:
        raise ValueError("max_k exceeds catalog size")
    partition = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
    partition_scores = np.take_along_axis(scores, partition, axis=1)
    order = np.argsort(-partition_scores, axis=1, kind="stable")
    return np.take_along_axis(partition, order, axis=1)


def _torch_top_indices(
    *,
    user_batch: np.ndarray,
    item_embeddings: np.ndarray,
    score_mode: str,
    targets: np.ndarray,
    user_ids: np.ndarray,
    train_seen: Mapping[int, set[int]],
    max_k: int,
    device_name: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--eval-device=cuda was requested but CUDA is unavailable")
    device = torch.device(device_name)
    users_t = torch.as_tensor(np.asarray(user_batch, dtype=np.float32), device=device)
    items_t = torch.as_tensor(np.asarray(item_embeddings, dtype=np.float32), device=device)
    if score_mode == "cosine":
        eps = float(np.finfo(np.float32).eps)
        users_t = F.normalize(users_t, p=2, dim=1, eps=eps)
        items_t = F.normalize(items_t, p=2, dim=1, eps=eps)
    elif score_mode != "raw_dot":
        raise ValueError(f"Unsupported score mode: {score_mode}")

    scores = users_t @ items_t.t()
    row_indices = torch.arange(len(targets), device=device)
    targets_t = torch.as_tensor(targets, dtype=torch.long, device=device)
    target_scores = scores[row_indices, targets_t].clone()
    n_items = scores.shape[1]
    for row, user_id in enumerate(user_ids.tolist()):
        seen_items = train_seen.get(int(user_id), set())
        if seen_items:
            seen_idx = np.fromiter(seen_items, dtype=np.int64, count=len(seen_items))
            seen_idx = seen_idx[(seen_idx >= 0) & (seen_idx < n_items)]
            if seen_idx.size:
                seen_idx_t = torch.as_tensor(seen_idx, dtype=torch.long, device=device)
                scores[row, seen_idx_t] = -torch.inf
    scores[row_indices, targets_t] = target_scores
    return torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True).indices.cpu().numpy()


def _select_group(test_df: pd.DataFrame, cold_threshold: int, eval_group: str) -> pd.DataFrame:
    if eval_group == "cold":
        return test_df.loc[test_df["popularity"] < cold_threshold].copy()
    if eval_group == "hot":
        return test_df.loc[test_df["popularity"] >= cold_threshold].copy()
    raise ValueError(f"Unsupported evaluation group: {eval_group}")


def evaluate_item_macro(
    *,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    train_seen: Mapping[int, set[int]],
    cold_threshold: int,
    score_mode: str,
    eval_group: str,
    k_list: Sequence[int] = (5, 10, 20),
    batch_size: int = 4096,
    eval_device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate full-catalog item-macro metrics with a train-only seen mask."""
    if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise ValueError("Embedding matrices must be two-dimensional")
    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError("User and item embedding dimensions differ")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if eval_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported eval device: {eval_device}")

    selected = _select_group(test_df, cold_threshold, eval_group)
    selected = selected.reset_index(drop=True)
    requested_metric_names = _metric_names(k_list)
    if selected.empty:
        return {
            "eval_group": eval_group,
            "score_mode": score_mode,
            "target_rows": 0,
            "item_count": 0,
            "metrics": {},
            "per_item": pd.DataFrame(columns=["item_id", "count", *requested_metric_names]),
        }

    n_items = int(item_embeddings.shape[0])
    max_k = min(max(k_list), n_items)
    effective_k_list = tuple(min(int(k), n_items) for k in k_list)
    if len(set(effective_k_list)) != len(effective_k_list):
        raise ValueError("Catalog is smaller than requested K values")

    item_metric_sums: Dict[int, Dict[str, float]] = {}
    item_counts: Dict[int, int] = {}
    selected_users = selected["u_idx"].to_numpy(dtype=np.int64, copy=False)
    selected_items = selected["i_idx"].to_numpy(dtype=np.int64, copy=False)
    if selected_users.min() < 0 or selected_users.max() >= user_embeddings.shape[0]:
        raise ValueError("Evaluation user IDs exceed user embedding matrix")
    if selected_items.min() < 0 or selected_items.max() >= n_items:
        raise ValueError("Evaluation item IDs exceed item embedding matrix")

    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        user_ids = selected_users[start:stop]
        targets = selected_items[start:stop]
        user_batch = user_embeddings[user_ids]
        if eval_device == "cpu":
            scores = score_matrix(user_batch, item_embeddings, score_mode)
            target_scores = scores[np.arange(stop - start), targets].copy()
            for row, user_id in enumerate(user_ids.tolist()):
                seen_items = train_seen.get(int(user_id), set())
                if seen_items:
                    seen_idx = np.fromiter(seen_items, dtype=np.int64, count=len(seen_items))
                    seen_idx = seen_idx[(seen_idx >= 0) & (seen_idx < n_items)]
                    scores[row, seen_idx] = -np.inf
            scores[np.arange(stop - start), targets] = target_scores
            top_indices = _top_indices(scores, max_k=max_k)
        else:
            top_indices = _torch_top_indices(
                user_batch=user_batch,
                item_embeddings=item_embeddings,
                score_mode=score_mode,
                targets=targets,
                user_ids=user_ids,
                train_seen=train_seen,
                max_k=max_k,
                device_name=eval_device,
            )
        values = _metric_values(top_indices, targets, effective_k_list)
        for row, item_id in enumerate(targets.tolist()):
            item_id = int(item_id)
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
            accum = item_metric_sums.setdefault(item_id, {metric: 0.0 for metric in requested_metric_names})
            for requested_k, effective_k in zip(k_list, effective_k_list):
                accum[f"R@{requested_k}"] += float(values[f"R@{effective_k}"][row])
                accum[f"N@{requested_k}"] += float(values[f"N@{effective_k}"][row])

    per_item_rows = []
    for item_id in sorted(item_counts):
        count = item_counts[item_id]
        row = {"item_id": item_id, "count": count}
        for metric in requested_metric_names:
            row[metric] = item_metric_sums[item_id][metric] / count
        per_item_rows.append(row)
    per_item = pd.DataFrame(per_item_rows, columns=["item_id", "count", *requested_metric_names])
    metrics = {metric: float(per_item[metric].mean()) for metric in requested_metric_names}
    return {
        "eval_group": eval_group,
        "score_mode": score_mode,
        "target_rows": int(len(selected)),
        "item_count": int(len(item_counts)),
        "metrics": metrics,
        "per_item": per_item,
    }


def _embedding_norm_summary(user_embeddings: np.ndarray, item_embeddings: np.ndarray) -> Dict[str, float]:
    user_norms = np.linalg.norm(user_embeddings, axis=1)
    item_norms = np.linalg.norm(item_embeddings, axis=1)
    return {
        "user_norm_mean": float(user_norms.mean()),
        "user_norm_std": float(user_norms.std()),
        "item_norm_mean": float(item_norms.mean()),
        "item_norm_std": float(item_norms.std()),
        "item_norm_min": float(item_norms.min()),
        "item_norm_max": float(item_norms.max()),
    }


def load_official_teacher_embeddings(path: Path, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(path)
    expected_shape = (n_users + n_items,)
    if values.ndim != 2 or values.shape[0] != expected_shape[0]:
        raise ValueError(
            f"Official teacher embedding must have {expected_shape[0]} rows, got {values.shape}"
        )
    return values[:n_users].astype(np.float32, copy=False), values[n_users:].astype(np.float32, copy=False)


def load_local_embeddings(path: Path, n_users: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        if "user_embeddings" not in values or "item_embeddings" not in values:
            raise ValueError("Local embedding archive must contain user_embeddings and item_embeddings")
        users = values["user_embeddings"].astype(np.float32, copy=False)
        items = values["item_embeddings"].astype(np.float32, copy=False)
    if users.shape[0] != n_users or items.shape[0] != n_items:
        raise ValueError(
            "Local embedding row counts do not match frozen metadata: "
            f"users={users.shape}, items={items.shape}, expected=({n_users}, {n_items})"
        )
    if users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]:
        raise ValueError("Local user/item embeddings have incompatible shapes")
    return users, items


def _seed_local_bpr(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _local_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--local-device=cuda was requested but CUDA is unavailable")
    return torch.device(requested)


def train_local_matched_bpr(
    *,
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    output_path: Path,
    seed: int,
    emb_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    reg_rate: float,
    device_name: str,
) -> Dict[str, Any]:
    """Train a local BPR-MF teacher with ALDI's train-only sampling policy.

    The local implementation matches the official teacher's observed policy:
    sample train users with replacement, draw one observed positive for each,
    draw negatives from train-visible items without rejection, use Adam, and
    use its batch-level L2 regularizer.  It intentionally saves the final
    epoch rather than selecting with validation data, keeping this audit's
    embeddings independent of the test set.
    """
    import torch
    import torch.nn.functional as F

    if epochs < 1 or emb_dim < 1 or batch_size < 2:
        raise ValueError("local BPR requires positive epochs/dimension and batch_size >= 2")
    _seed_local_bpr(seed)
    device = _local_device(device_name)
    users = torch.empty((n_users, emb_dim), device=device, requires_grad=True)
    items = torch.empty((n_items, emb_dim), device=device, requires_grad=True)
    # TensorFlow's truncated_normal_initializer(stddev=0.01) cuts at two stddev.
    torch.nn.init.trunc_normal_(users, mean=0.0, std=0.01, a=-0.02, b=0.02)
    torch.nn.init.trunc_normal_(items, mean=0.0, std=0.01, a=-0.02, b=0.02)
    optimizer = torch.optim.Adam((users, items), lr=lr, eps=1e-8)

    sampling_tables = build_teacher_sampling_tables(train_df)
    n_pairs = int(len(train_df))
    if sampling_tables.warm_users.size == 0 or sampling_tables.warm_items.size == 0 or n_pairs <= batch_size:
        raise ValueError("Training data is too small for the official-size BPR batching policy")

    rng = np.random.default_rng(seed)
    batches_per_epoch = len(range(0, n_pairs - batch_size, batch_size))
    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        sampled_users, sampled_pos, sampled_neg = sample_teacher_triplets(
            tables=sampling_tables,
            n_pairs=n_pairs,
            rng=rng,
        )

        sampled_users_t = torch.as_tensor(sampled_users, dtype=torch.long, device=device)
        sampled_pos_t = torch.as_tensor(sampled_pos, dtype=torch.long, device=device)
        sampled_neg_t = torch.as_tensor(sampled_neg, dtype=torch.long, device=device)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        for start in range(0, n_pairs - batch_size, batch_size):
            stop = start + batch_size
            user_idx = sampled_users_t[start:stop]
            pos_idx = sampled_pos_t[start:stop]
            neg_idx = sampled_neg_t[start:stop]
            z_u = users[user_idx]
            z_p = items[pos_idx]
            z_n = items[neg_idx]
            loss = F.softplus(-((z_u * z_p).sum(dim=1) - (z_u * z_n).sum(dim=1))).mean()
            loss = loss + reg_rate * 0.5 * (z_u.norm() + z_p.norm() + z_n.norm()) / batch_size
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        final_loss = float((total_loss / max(1, batches_per_epoch)).detach().cpu())
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"local_bpr epoch={epoch}/{epochs} loss={final_loss:.6f}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        user_embeddings=users.detach().cpu().numpy().astype(np.float32),
        item_embeddings=items.detach().cpu().numpy().astype(np.float32),
    )
    metadata = {
        "implementation": "local_matched_bpr",
        "selection": "last_epoch_without_validation",
        "seed": int(seed),
        "epochs": int(epochs),
        "emb_dim": int(emb_dim),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "reg_rate": float(reg_rate),
        "device": str(device),
        "batches_per_epoch": int(batches_per_epoch),
        "final_loss": float(final_loss),
        "output": str(output_path.resolve()),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _result_row(source: str, score_mode: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "embedding_source": source,
        "score_mode": score_mode,
        "eval_group": result["eval_group"],
        "target_rows": result["target_rows"],
        "item_count": result["item_count"],
    }
    row.update(result["metrics"])
    return row


def run_audit(
    *,
    frozen_split: FrozenSplit,
    embedding_sources: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    cold_threshold: int,
    batch_size: int,
    eval_device: str,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    output_dir = output_dir.resolve()
    summary_path = output_dir / "score_parity_summary.csv"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing audit output: {summary_path}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seen = build_train_seen(frozen_split.train)
    detail_rows = []
    summary_rows = []
    source_norms: Dict[str, Dict[str, float]] = {}

    for source, (user_embeddings, item_embeddings) in embedding_sources.items():
        if user_embeddings.shape[0] != frozen_split.n_users or item_embeddings.shape[0] != frozen_split.n_items:
            raise ValueError(f"{source} embedding row counts do not match frozen split metadata")
        source_norms[source] = _embedding_norm_summary(user_embeddings, item_embeddings)
        grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
        for score_mode in ("raw_dot", "cosine"):
            for eval_group in ("cold", "hot"):
                result = evaluate_item_macro(
                    user_embeddings=user_embeddings,
                    item_embeddings=item_embeddings,
                    test_df=frozen_split.test,
                    train_seen=train_seen,
                    cold_threshold=cold_threshold,
                    score_mode=score_mode,
                    eval_group=eval_group,
                    batch_size=batch_size,
                    eval_device=eval_device,
                )
                result["per_item"].to_csv(
                    output_dir / f"per_item_{source}_{score_mode}_{eval_group}.csv",
                    index=False,
                )
                grouped[(score_mode, eval_group)] = result
                detail_rows.append(_result_row(source, score_mode, result))

        for score_mode in ("raw_dot", "cosine"):
            row: Dict[str, Any] = {"embedding_source": source, "score_mode": score_mode}
            for eval_group in ("cold", "hot"):
                result = grouped[(score_mode, eval_group)]
                row[f"{eval_group}_target_rows"] = result["target_rows"]
                row[f"{eval_group}_item_count"] = result["item_count"]
                for metric in METRICS:
                    row[f"{eval_group}_{metric}"] = result["metrics"][metric]
            summary_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(output_dir / "score_parity_detail.csv", index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "purpose": "MOOCCube seed-2025 BPR score-function parity audit",
        "split_dir": str(frozen_split.split_dir),
        "split_sha256": frozen_split.split_sha256,
        "n_users": frozen_split.n_users,
        "n_items": frozen_split.n_items,
        "train_rows": int(len(frozen_split.train)),
        "test_rows": int(len(frozen_split.test)),
        "cold_threshold": int(cold_threshold),
        "candidate_set": "full catalog",
        "averaging": "item_macro",
        "eval_device": eval_device,
        "history_mask": "train_only; target score is restored after masking",
        "raw_dot_definition": "dot product; official teacher sigmoid(dot) is rank-equivalent",
        "cosine_definition": "L2-normalize user and item vectors before dot product",
        "embedding_norms": source_norms,
        "outputs": {
            "summary": str(summary_path),
            "detail": str(output_dir / "score_parity_detail.csv"),
        },
    }
    (output_dir / "score_parity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return summary, detail, manifest


def main() -> None:
    args = parse_args()
    frozen_split = load_frozen_split(args.split_dir, args.meta_path)
    output_dir = args.output_dir.resolve()
    local_embedding = args.local_embedding or output_dir / "local_matched_bpr_embeddings.npz"
    if args.train_local_bpr:
        train_metadata = train_local_matched_bpr(
            train_df=frozen_split.train,
            n_users=frozen_split.n_users,
            n_items=frozen_split.n_items,
            output_path=local_embedding,
            seed=args.local_seed,
            emb_dim=args.local_emb_dim,
            epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.local_lr,
            reg_rate=args.local_reg_rate,
            device_name=args.local_device,
        )
        print(json.dumps(train_metadata, indent=2))
    if not local_embedding.is_file():
        raise FileNotFoundError(
            f"Missing local BPR embeddings: {local_embedding}. Use --train-local-bpr or provide --local-embedding."
        )
    official = load_official_teacher_embeddings(
        args.official_embedding, frozen_split.n_users, frozen_split.n_items
    )
    local = load_local_embeddings(local_embedding, frozen_split.n_users, frozen_split.n_items)
    summary, _, _ = run_audit(
        frozen_split=frozen_split,
        embedding_sources={"official_aldi_bpr_teacher": official, "local_matched_bpr": local},
        output_dir=output_dir,
        cold_threshold=args.cold_threshold,
        batch_size=args.batch_size,
        eval_device=args.eval_device,
        overwrite=args.overwrite,
    )
    print(summary.to_string(index=False))
    print(f"Saved score-parity audit to {output_dir}")


if __name__ == "__main__":
    main()
