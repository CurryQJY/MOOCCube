from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
KGREC_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "KGRec"
DEFAULT_ATOMIC_DIR = ROOT / "paper_aaai27" / "baseline_sources" / "_kgrec_strict" / "mooccube_seed2025_atomic"
DEFAULT_OUTPUT_DIR = ROOT / "paper_aaai27" / "baseline_sources" / "_kgrec_strict" / "mooccube_seed2025_single"


def read_grouped_pairs(path: str | Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            user = int(tokens[0])
            for item in tokens[1:]:
                pairs.append((user, int(item)))
    return pairs


def build_seen_by_user(pairs: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for user, item in pairs:
        seen[int(user)].add(int(item))
    return dict(seen)


def sample_warm_negatives(
    train_pairs: np.ndarray,
    user_seen_by_user: Mapping[int, set[int]],
    warm_item_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    triples = np.empty((train_pairs.shape[0], 3), dtype=np.int64)
    triples[:, :2] = train_pairs[:, :2]
    warm = np.asarray(warm_item_ids, dtype=np.int64)
    if warm.size == 0:
        raise ValueError("warm_item_ids must be nonempty")

    fallback_cache: dict[int, np.ndarray] = {}
    for row_idx, (user, _pos_item) in enumerate(train_pairs[:, :2]):
        user = int(user)
        seen = user_seen_by_user.get(user, set())
        neg = int(warm[int(rng.integers(0, warm.size))])
        tries = 0
        while neg in seen and tries < 32:
            neg = int(warm[int(rng.integers(0, warm.size))])
            tries += 1
        if neg in seen:
            candidates = fallback_cache.get(user)
            if candidates is None:
                candidates = np.asarray([item for item in warm if int(item) not in seen], dtype=np.int64)
                fallback_cache[user] = candidates
            if candidates.size == 0:
                raise ValueError(f"user {user} has no warm negative candidates")
            neg = int(candidates[int(rng.integers(0, candidates.size))])
        triples[row_idx, 2] = neg
    return triples


def _empty_metric_dict(k_list: Sequence[int]) -> dict[str, float]:
    return {f"{name}@{k}": 0.0 for name in ("R", "N") for k in k_list}


def _finalize_item_macro(
    item_metric_sums: Mapping[str, Mapping[str, Mapping[int, float]]],
    item_counts: Mapping[str, Mapping[int, int]],
    k_list: Sequence[int],
) -> dict[str, object]:
    output: dict[str, object] = {}
    counts: dict[str, int] = {}
    for group in ("cold", "hot", "all"):
        metrics = _empty_metric_dict(k_list)
        item_ids = sorted(item_counts[group])
        counts[f"{group}_items"] = len(item_ids)
        counts[f"{group}_rows"] = int(sum(item_counts[group].values()))
        if item_ids:
            for metric_name in metrics:
                item_means = [
                    item_metric_sums[group][metric_name][item] / item_counts[group][item]
                    for item in item_ids
                ]
                metrics[metric_name] = float(np.mean(item_means))
        output[f"full_{group}_item_macro"] = metrics
    output["counts"] = counts
    return output


def _accumulate_pair_metrics(
    *,
    scores: np.ndarray,
    user: int,
    target: int,
    train_seen_by_user: Mapping[int, set[int]],
    cold_item_ids: set[int],
    k_list: Sequence[int],
    item_metric_sums: dict[str, dict[str, defaultdict[int, float]]],
    item_counts: dict[str, defaultdict[int, int]],
) -> None:
    row = np.array(scores, copy=True)
    if target < 0 or target >= row.shape[0]:
        return
    target_score = float(row[target])
    for seen_item in train_seen_by_user.get(int(user), set()):
        if seen_item != target and 0 <= seen_item < row.shape[0]:
            row[seen_item] = -np.inf
    rank = 1 + int(np.sum(row > target_score))
    groups = ["all", "cold" if target in cold_item_ids else "hot"]
    for group in groups:
        item_counts[group][target] += 1
        for k in k_list:
            hit = 1.0 if rank <= k else 0.0
            ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
            item_metric_sums[group][f"R@{k}"][target] += hit
            item_metric_sums[group][f"N@{k}"][target] += ndcg


def evaluate_item_macro_from_scores(
    *,
    scores: np.ndarray,
    eval_pairs: Sequence[tuple[int, int]],
    train_seen_by_user: Mapping[int, set[int]],
    cold_item_ids: set[int],
    k_list: Sequence[int] = (5, 10, 20),
) -> dict[str, object]:
    item_metric_sums: dict[str, dict[str, defaultdict[int, float]]] = {
        group: {metric: defaultdict(float) for metric in _empty_metric_dict(k_list)}
        for group in ("cold", "hot", "all")
    }
    item_counts: dict[str, defaultdict[int, int]] = {group: defaultdict(int) for group in ("cold", "hot", "all")}
    for user, target in eval_pairs:
        if 0 <= int(user) < scores.shape[0]:
            _accumulate_pair_metrics(
                scores=scores[int(user)],
                user=int(user),
                target=int(target),
                train_seen_by_user=train_seen_by_user,
                cold_item_ids=cold_item_ids,
                k_list=k_list,
                item_metric_sums=item_metric_sums,
                item_counts=item_counts,
            )
    return _finalize_item_macro(item_metric_sums, item_counts, k_list)


def evaluate_model_item_macro(
    model,
    *,
    eval_pairs: Sequence[tuple[int, int]],
    train_seen_by_user: Mapping[int, set[int]],
    cold_item_ids: set[int],
    n_items: int,
    device,
    batch_size: int,
    k_list: Sequence[int] = (5, 10, 20),
) -> dict[str, object]:
    import torch

    pairs_by_user: dict[int, list[int]] = defaultdict(list)
    for user, target in eval_pairs:
        pairs_by_user[int(user)].append(int(target))
    users = sorted(pairs_by_user)

    item_metric_sums: dict[str, dict[str, defaultdict[int, float]]] = {
        group: {metric: defaultdict(float) for metric in _empty_metric_dict(k_list)}
        for group in ("cold", "hot", "all")
    }
    item_counts: dict[str, defaultdict[int, int]] = {group: defaultdict(int) for group in ("cold", "hot", "all")}

    model.eval()
    with torch.no_grad():
        entity_gcn_emb, user_gcn_emb = model.generate()
        item_gcn_emb = entity_gcn_emb[:n_items]
        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            user_tensor = torch.as_tensor(batch_users, dtype=torch.long, device=device)
            batch_scores = model.rating(user_gcn_emb[user_tensor], item_gcn_emb).detach().cpu().numpy()
            for row_idx, user in enumerate(batch_users):
                for target in pairs_by_user[user]:
                    _accumulate_pair_metrics(
                        scores=batch_scores[row_idx],
                        user=user,
                        target=target,
                        train_seen_by_user=train_seen_by_user,
                        cold_item_ids=cold_item_ids,
                        k_list=k_list,
                        item_metric_sums=item_metric_sums,
                        item_counts=item_counts,
                    )
    return _finalize_item_macro(item_metric_sums, item_counts, k_list)


def prepare_loader_dataset(atomic_dir: Path, output_dir: Path, dataset_name: str) -> Path:
    loader_dir = output_dir / "_loader_data" / dataset_name
    loader_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(atomic_dir / "train.txt", loader_dir / "train.txt")
    shutil.copyfile(atomic_dir / "kg_final.txt", loader_dir / "kg_final.txt")
    validation = (atomic_dir / "validation.txt").read_text(encoding="utf-8")
    test = (atomic_dir / "test.txt").read_text(encoding="utf-8")
    (loader_dir / "test.txt").write_text(validation + test, encoding="utf-8")
    return loader_dir


@dataclass
class RunConfig:
    seed: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    eval_every: int
    patience: int
    dim: int
    lr: float
    l2: float
    context_hops: int
    node_dropout_rate: float
    mess_dropout_rate: float
    max_train_batches: int
    requested_device: str


def run_single_seed(args: argparse.Namespace) -> dict[str, object]:
    if str(KGREC_ROOT) not in sys.path:
        sys.path.insert(0, str(KGREC_ROOT))

    import torch
    from modules.KGRec import KGRec
    import utils.data_loader as data_loader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device("cuda:0" if requested_device == "cuda" else "cpu")

    atomic_dir = Path(args.atomic_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((atomic_dir / "strict_manifest.json").read_text(encoding="utf-8"))
    dataset_name = f"kgrec_strict_seed{args.seed}_loader"
    loader_dir = prepare_loader_dataset(atomic_dir, output_dir, dataset_name)

    data_loader.n_users = 0
    data_loader.n_items = 0
    data_loader.n_entities = 0
    data_loader.n_relations = 0
    data_loader.n_nodes = 0
    data_loader.train_user_set = defaultdict(list)
    data_loader.test_user_set = defaultdict(list)

    kg_args = SimpleNamespace(
        dataset=dataset_name,
        data_path=str(loader_dir.parent) + os.sep,
        model="KGSR",
        cuda=1 if requested_device == "cuda" else 0,
        gpu_id=0,
        dim=args.dim,
        channel=args.dim,
        l2=args.l2,
        lr=args.lr,
        inverse_r=True,
        node_dropout=1,
        node_dropout_rate=args.node_dropout_rate,
        mess_dropout=1,
        mess_dropout_rate=args.mess_dropout_rate,
        context_hops=args.context_hops,
        mae_coef=args.mae_coef,
        mae_msize=args.mae_msize,
        cl_coef=args.cl_coef,
        cl_tau=args.cl_tau,
        cl_drop_ratio=args.cl_drop_ratio,
        ab=None,
        save=False,
        out_dir=str(output_dir),
    )

    train_cf, _test_cf, _user_dict, n_params, graph, mat_list = data_loader.load_data(kg_args)
    if int(n_params["n_items"]) != int(manifest["n_items"]):
        raise RuntimeError(f"KGRec loader n_items={n_params['n_items']} != manifest n_items={manifest['n_items']}")

    train_pairs = read_grouped_pairs(atomic_dir / "train.txt")
    validation_pairs = read_grouped_pairs(atomic_dir / "validation.txt")
    test_pairs = read_grouped_pairs(atomic_dir / "test.txt")
    train_seen_by_user = build_seen_by_user(train_pairs)
    warm_item_ids = np.asarray(manifest["warm_item_ids"], dtype=np.int64)
    cold_item_ids = {int(item) for item in manifest["cold_item_ids"]}

    model = KGRec(n_params, kg_args, graph, mat_list[2][0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    progress_path = output_dir / "training_progress.jsonl"
    checkpoint_path = output_dir / "best_model.pt"
    report_path = output_dir / "kgrec_strict_adapter_report.json"
    best_score = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    best_validation: dict[str, object] | None = None
    start_time = time.time()

    config = RunConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_every=args.eval_every,
        patience=args.patience,
        dim=args.dim,
        lr=args.lr,
        l2=args.l2,
        context_hops=args.context_hops,
        node_dropout_rate=args.node_dropout_rate,
        mess_dropout_rate=args.mess_dropout_rate,
        max_train_batches=args.max_train_batches,
        requested_device=args.device,
    )

    with progress_path.open("a", encoding="utf-8") as progress:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_cf_with_neg = sample_warm_negatives(
                train_cf.astype(np.int64),
                train_seen_by_user,
                warm_item_ids,
                seed=args.seed + epoch,
            )
            order = np.arange(train_cf_with_neg.shape[0])
            np.random.default_rng(args.seed + epoch * 997).shuffle(order)
            train_cf_with_neg = train_cf_with_neg[order]

            model.train()
            losses: list[float] = []
            batch_count = 0
            for start in range(0, train_cf_with_neg.shape[0], args.batch_size):
                end = min(start + args.batch_size, train_cf_with_neg.shape[0])
                if end <= start:
                    continue
                batch_np = train_cf_with_neg[start:end]
                batch_tensor = torch.from_numpy(batch_np).to(device).long()
                batch = {
                    "users": batch_tensor[:, 0],
                    "pos_items": batch_tensor[:, 1],
                    "neg_items": batch_tensor[:, 2],
                    "batch_start": start,
                }
                loss, _loss_dict = model(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                batch_count += 1
                if args.max_train_batches > 0 and batch_count >= args.max_train_batches:
                    break

            row: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "train_batches": batch_count,
                "epoch_seconds": time.time() - epoch_start,
            }

            if epoch % args.eval_every == 0:
                validation = evaluate_model_item_macro(
                    model,
                    eval_pairs=validation_pairs,
                    train_seen_by_user=train_seen_by_user,
                    cold_item_ids=cold_item_ids,
                    n_items=int(manifest["n_items"]),
                    device=device,
                    batch_size=args.eval_batch_size,
                )
                score = float(validation["full_cold_item_macro"]["N@10"])
                row["validation_full_cold_item_macro"] = validation["full_cold_item_macro"]
                row["validation_score"] = score
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_validation = validation
                    bad_epochs = 0
                    torch.save(model.state_dict(), checkpoint_path)
                    row["best"] = True
                else:
                    bad_epochs += 1
                    row["best"] = False
                row["bad_epochs"] = bad_epochs
                progress.write(json.dumps(row, sort_keys=True) + "\n")
                progress.flush()
                if args.patience > 0 and bad_epochs >= args.patience:
                    break
            else:
                progress.write(json.dumps(row, sort_keys=True) + "\n")
                progress.flush()

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate_model_item_macro(
        model,
        eval_pairs=test_pairs,
        train_seen_by_user=train_seen_by_user,
        cold_item_ids=cold_item_ids,
        n_items=int(manifest["n_items"]),
        device=device,
        batch_size=args.eval_batch_size,
    )

    report = {
        "model": "KGRec (adapted)",
        "status": "complete",
        "seed": args.seed,
        "atomic_dir": str(atomic_dir),
        "output_dir": str(output_dir),
        "requested_device": args.device,
        "device": requested_device,
        "config": asdict(config),
        "strict_protocol": {
            "cf_train_source": "static_train.pkl via train.txt",
            "negative_sampling": "warm_items_only",
            "candidate_mode": "full_catalog",
            "train_history_masking": True,
            "item_macro_metrics": True,
            "user_video_edges_excluded": True,
        },
        "data": {
            "n_users": manifest["n_users"],
            "n_items": manifest["n_items"],
            "n_entities": manifest["n_entities"],
            "n_relations": manifest["n_relations"],
            "n_train_pairs": manifest["n_train_pairs"],
            "n_validation_pairs": manifest["n_validation_pairs"],
            "n_test_pairs": manifest["n_test_pairs"],
            "n_kg_triples": manifest["n_kg_triples"],
        },
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation": best_validation,
        "test": test_metrics,
        "runtime_seconds": time.time() - start_time,
        "progress_path": str(progress_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adapted KGRec under strict item-cold protocol.")
    parser.add_argument("--atomic-dir", type=Path, default=DEFAULT_ATOMIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--context-hops", type=int, default=2)
    parser.add_argument("--node-dropout-rate", type=float, default=0.5)
    parser.add_argument("--mess-dropout-rate", type=float, default=0.1)
    parser.add_argument("--mae-coef", type=float, default=0.1)
    parser.add_argument("--mae-msize", type=int, default=256)
    parser.add_argument("--cl-coef", type=float, default=0.01)
    parser.add_argument("--cl-tau", type=float, default=1.0)
    parser.add_argument("--cl-drop-ratio", type=float, default=0.5)
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    report = run_single_seed(parse_args())
    print(json.dumps({"status": report["status"], "best_epoch": report["best_epoch"], "output_dir": report["output_dir"]}))


if __name__ == "__main__":
    main()
