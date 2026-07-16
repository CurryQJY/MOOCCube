from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from paper_aaai27.scripts.pcgnn_strict_adapter import (
    DEFAULT_PCGNN_CONFIG,
    PCGNN_ROOT,
    ItemMacroRankingAccumulator,
    _frame_rows,
    _iter_batches,
    _read_split_rows,
    build_strict_eval_examples,
    build_user_seen_items,
    clean_argv_for_recbole,
    evaluate_pcgnn_full_item_macro,
    move_tensor_dict_to_device,
    resolve_torch_device,
)
from paper_aaai27.scripts.priority_baseline_experiments import (
    local_pcgnn_recbole,
    pcgnn_smoke_config_overrides,
    tensorize_examples,
)
from ranking_topk_export import MASKED_SCORE_THRESHOLD, TopKJsonlExporter


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = ROOT / "paper_aaai27" / "baseline_sources" / "_pcgnn_strict"
SPLIT_FILENAMES = (
    "static_protocol_manifest.json",
    "static_split_assignments.csv",
    "static_split_counts.csv",
    "static_split_sources.csv",
    "static_split_summary.json",
    "static_train.pkl",
    "static_val.pkl",
    "static_test.pkl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def mask_pcgnn_scores(
    scores: torch.Tensor,
    examples: list[dict[str, object]],
    user_seen_items: Mapping[int, set[int]],
) -> torch.Tensor:
    if scores.ndim != 2 or int(scores.shape[0]) != len(examples):
        raise ValueError("scores must have shape [len(examples), items]")

    masked = scores.detach().clone()
    if masked.shape[1] > 0:
        masked[:, 0] = -torch.inf
    for row_idx, example in enumerate(examples):
        target = int(example["target"])
        if not 0 <= target < int(masked.shape[1]):
            raise ValueError(f"target item id is outside the score catalog: {target}")
        target_score = scores[row_idx, target].detach().clone()
        for seen in user_seen_items.get(int(example["user"]), set()):
            if 0 <= int(seen) < int(masked.shape[1]):
                masked[row_idx, int(seen)] = -torch.inf
        masked[row_idx, target] = target_score
    return masked


def _rank_masked_scores(masked_scores: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    scores = masked_scores.detach().cpu().numpy()
    keep_k = min(int(top_k), int(scores.shape[1]))
    order = np.argsort(-scores, axis=1)[:, :keep_k]
    top_scores = np.take_along_axis(scores, order, axis=1)
    return torch.as_tensor(order, dtype=torch.long), torch.as_tensor(top_scores)


def replay_and_export_pcgnn(
    *,
    model,
    interaction_cls,
    examples: list[dict[str, object]],
    user_seen_items: Mapping[int, set[int]],
    internal_to_raw: Mapping[int, int],
    max_len: int,
    batch_size: int,
    top_k: int,
    output_path: Path,
    metadata: Mapping[str, object],
    device,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    metric_ks = tuple(sorted({5, 10, 20, int(top_k)}))
    accumulator = ItemMacroRankingAccumulator(k_list=metric_ks, cold_threshold=1)
    model.eval()
    record_count = 0

    with TopKJsonlExporter(output_path, top_k=top_k, metadata=metadata) as exporter:
        with torch.no_grad():
            for batch in _iter_batches(examples, batch_size):
                item_seq, item_len, _ = tensorize_examples(batch, max_len)
                interaction = interaction_cls(
                    move_tensor_dict_to_device(
                        {model.ITEM_SEQ: item_seq, model.ITEM_SEQ_LEN: item_len},
                        device,
                    )
                )
                raw_scores = model.full_sort_predict(interaction).detach().cpu()
                accumulator.add_batch(raw_scores.numpy(), batch, dict(user_seen_items))
                masked_scores = mask_pcgnn_scores(raw_scores, batch, user_seen_items)
                top_internal, top_scores = _rank_masked_scores(masked_scores, top_k)

                if top_internal.shape[1] != top_k:
                    raise RuntimeError(
                        f"catalog has only {top_internal.shape[1]} candidates; cannot export Top-{top_k}"
                    )
                if not torch.isfinite(top_scores).all() or torch.any(top_scores <= MASKED_SCORE_THRESHOLD):
                    raise RuntimeError("fewer than top_k valid unmasked PCGNN candidates")

                raw_rows: list[list[int]] = []
                for row in top_internal.tolist():
                    try:
                        raw_rows.append([int(internal_to_raw[int(item)]) for item in row])
                    except KeyError as exc:
                        raise RuntimeError(f"missing raw item token for internal id {exc.args[0]}") from exc
                top_raw = torch.as_tensor(raw_rows, dtype=torch.long)
                exporter.write_precomputed_batch(
                    top_raw,
                    top_scores,
                    user_ids=[int(example["user"]) for example in batch],
                    target_item_ids=[int(example["raw_item"]) for example in batch],
                    target_popularity=[int(example.get("popularity", 0)) for example in batch],
                )
                record_count += len(batch)

    return {
        "record_count": record_count,
        "metrics": accumulator.result(),
    }


def compare_replay_to_report(
    replay: Mapping[str, object],
    native_report: Mapping[str, object],
    *,
    tolerance: float,
    raise_on_metric_drift: bool = True,
) -> dict[str, object]:
    native_test = dict(native_report.get("test", {}))
    replay_metrics = dict(replay.get("metrics", {}))
    count_checks = {
        "test_sequence_examples": (
            int(replay.get("record_count", -1)),
            int(native_report.get("test_sequence_examples", -2)),
        ),
        "rows_full_cold": (
            int(replay_metrics.get("rows_full_cold", -1)),
            int(native_test.get("rows_full_cold", -2)),
        ),
        "count_full_cold_item_macro": (
            int(replay_metrics.get("count_full_cold_item_macro", -1)),
            int(native_test.get("count_full_cold_item_macro", -2)),
        ),
    }
    mismatched_counts = {
        name: {"replay": replay_value, "report": report_value}
        for name, (replay_value, report_value) in count_checks.items()
        if replay_value != report_value
    }
    if mismatched_counts:
        raise RuntimeError(f"replay count mismatch: {mismatched_counts}")

    native_ranking = dict(native_test.get("full_cold_item_macro", {}))
    replay_ranking = dict(replay_metrics.get("full_cold_item_macro", {}))
    drift: dict[str, float] = {}
    for metric, expected in native_ranking.items():
        if metric not in replay_ranking:
            raise RuntimeError(f"replay metric is missing: {metric}")
        drift[metric] = abs(float(replay_ranking[metric]) - float(expected))
    max_drift = max(drift.values(), default=0.0)
    passed = max_drift <= float(tolerance)
    if not passed and raise_on_metric_drift:
        raise RuntimeError(
            f"PCGNN replay metric drift {max_drift:.17g} exceeds tolerance {tolerance:.17g}: {drift}"
        )
    return {
        "passed": passed,
        "tolerance": float(tolerance),
        "count_checks": {
            name: {"replay": replay_value, "report": report_value}
            for name, (replay_value, report_value) in count_checks.items()
        },
        "metric_abs_drift": drift,
        "max_abs_metric_drift": max_drift,
    }


def compare_checkpoint_validation_to_report(
    replay_validation: Mapping[str, object],
    native_report: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    tolerance: float,
) -> dict[str, object]:
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    report_epoch = int(native_report.get("best_epoch", -2))
    if checkpoint_epoch != report_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {checkpoint_epoch} != {report_epoch}")
    checkpoint_metric = str(checkpoint.get("validation_metric", ""))
    report_metric = str(native_report.get("validation_metric", ""))
    if checkpoint_metric != report_metric:
        raise RuntimeError(f"checkpoint validation metric mismatch: {checkpoint_metric} != {report_metric}")
    checkpoint_score = float(checkpoint.get("validation_score", float("nan")))
    report_score = float(native_report.get("best_validation_score", float("nan")))
    score_drift = abs(checkpoint_score - report_score)
    if not math.isfinite(score_drift) or score_drift > tolerance:
        raise RuntimeError(f"checkpoint validation score mismatch: {checkpoint_score} != {report_score}")

    native_validation = dict(native_report.get("validation", {}))
    count_checks = {
        "rows_full_cold": (
            int(replay_validation.get("rows_full_cold", -1)),
            int(native_validation.get("rows_full_cold", -2)),
        ),
        "count_full_cold_item_macro": (
            int(replay_validation.get("count_full_cold_item_macro", -1)),
            int(native_validation.get("count_full_cold_item_macro", -2)),
        ),
    }
    mismatched_counts = {
        name: {"replay": replay_value, "report": report_value}
        for name, (replay_value, report_value) in count_checks.items()
        if replay_value != report_value
    }
    if mismatched_counts:
        raise RuntimeError(f"checkpoint validation count mismatch: {mismatched_counts}")

    native_ranking = dict(native_validation.get("full_cold_item_macro", {}))
    replay_ranking = dict(replay_validation.get("full_cold_item_macro", {}))
    drift: dict[str, float] = {}
    for metric, expected in native_ranking.items():
        if metric not in replay_ranking:
            raise RuntimeError(f"checkpoint validation replay metric is missing: {metric}")
        drift[metric] = abs(float(replay_ranking[metric]) - float(expected))
    max_drift = max(drift.values(), default=0.0)
    if max_drift > tolerance:
        raise RuntimeError(f"checkpoint validation metric drift: {drift}")
    return {
        "passed": True,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_validation_metric": checkpoint_metric,
        "checkpoint_validation_score": checkpoint_score,
        "checkpoint_score_abs_drift": score_drift,
        "count_checks": {
            name: {"replay": replay_value, "report": report_value}
            for name, (replay_value, report_value) in count_checks.items()
        },
        "metric_abs_drift": drift,
        "max_abs_metric_drift": max_drift,
    }


def _metrics_from_export(rows: Iterable[dict[str, object]], k_list: Iterable[int]) -> dict[str, object]:
    k_values = tuple(sorted({int(k) for k in k_list}))
    sums = {
        "cold": {f"{metric}@{k}": defaultdict(float) for metric in ("R", "N") for k in k_values},
        "hot": {f"{metric}@{k}": defaultdict(float) for metric in ("R", "N") for k in k_values},
    }
    counts = {"cold": defaultdict(int), "hot": defaultdict(int)}
    row_counts = {"cold": 0, "hot": 0}
    for row in rows:
        target = int(row["target_item_id"])
        group = "cold" if int(row.get("target_popularity", 0)) < 1 else "hot"
        recommendations = [int(item) for item in row["recommended_item_ids"]]
        counts[group][target] += 1
        row_counts[group] += 1
        for k in k_values:
            prefix = recommendations[:k]
            try:
                position = prefix.index(target)
            except ValueError:
                hit = 0.0
                ndcg = 0.0
            else:
                hit = 1.0
                ndcg = 1.0 / math.log2(position + 2.0)
            sums[group][f"R@{k}"][target] += hit
            sums[group][f"N@{k}"][target] += ndcg

    result: dict[str, object] = {}
    for group in ("cold", "hot"):
        ranking = {}
        for metric, per_item in sums[group].items():
            values = [float(per_item[item]) / count for item, count in counts[group].items()]
            ranking[metric] = float(sum(values) / len(values)) if values else 0.0
        result[f"full_{group}_item_macro"] = ranking if counts[group] else {}
        result[f"count_full_{group}_item_macro"] = len(counts[group])
        result[f"rows_full_{group}"] = row_counts[group]
    return result


def validate_pcgnn_topk_export(
    *,
    output_path: Path,
    examples: list[dict[str, object]],
    user_seen_items: Mapping[int, set[int]],
    internal_to_raw: Mapping[int, int],
    top_k: int,
    expected_metadata: Mapping[str, object],
    replay_metrics: Mapping[str, object],
    tolerance: float,
) -> dict[str, object]:
    rows = [json.loads(line) for line in Path(output_path).open("r", encoding="utf-8")]
    if len(rows) != len(examples):
        raise RuntimeError(f"Top-K record count mismatch: {len(rows)} != {len(examples)}")

    raw_seen = {
        int(user): {int(internal_to_raw[item]) for item in items if item in internal_to_raw}
        for user, items in user_seen_items.items()
    }
    for index, (row, example) in enumerate(zip(rows, examples)):
        if int(row.get("sample_index", -1)) != index:
            raise RuntimeError(f"non-sequential sample_index at row {index}")
        for key, expected in expected_metadata.items():
            if row.get(key) != expected:
                raise RuntimeError(f"metadata mismatch at row {index}: {key}")
        if int(row["user_id"]) != int(example["user"]):
            raise RuntimeError(f"user sequence mismatch at row {index}")
        if int(row["target_item_id"]) != int(example["raw_item"]):
            raise RuntimeError(f"target sequence mismatch at row {index}")
        items = [int(item) for item in row["recommended_item_ids"]]
        scores = [float(score) for score in row["recommended_scores"]]
        if len(items) != top_k or len(scores) != top_k:
            raise RuntimeError(f"row {index} is not exactly Top-{top_k}")
        if len(set(items)) != top_k:
            raise RuntimeError(f"duplicate recommendation at row {index}")
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError(f"non-finite recommendation score at row {index}")
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise RuntimeError(f"recommendation scores are not sorted at row {index}")
        leaked = set(items) & raw_seen.get(int(example["user"]), set())
        leaked.discard(int(example["raw_item"]))
        if leaked:
            raise RuntimeError(f"seen-item leakage at row {index}: {sorted(leaked)[:5]}")

    export_metrics = _metrics_from_export(rows, k_list=(5, 10, 20, top_k))
    replay_ranking = dict(replay_metrics.get("full_cold_item_macro", {}))
    export_ranking = dict(export_metrics.get("full_cold_item_macro", {}))
    drift = {
        metric: abs(float(export_ranking[metric]) - float(value))
        for metric, value in replay_ranking.items()
        if metric in export_ranking
    }
    max_drift = max(drift.values(), default=0.0)
    if max_drift > tolerance:
        raise RuntimeError(f"exported ranking metric drift: {drift}")
    return {
        "passed": True,
        "record_count": len(rows),
        "exact_top_k": int(top_k),
        "seen_item_leakage_count": 0,
        "export_metrics": export_metrics,
        "metric_abs_drift_from_replay": drift,
        "max_abs_metric_drift_from_replay": max_drift,
    }


def build_pcgnn_export_manifest(
    *,
    seed: int,
    top_k: int,
    checkpoint_path: Path,
    checkpoint_sha256_before: str,
    checkpoint_sha256_after: str,
    report_path: Path,
    config_path: Path,
    split_paths: Iterable[Path],
    script_paths: Iterable[Path],
    topk_output: Path,
    replay_result: Path,
    record_count: int,
) -> dict[str, object]:
    if checkpoint_sha256_before != checkpoint_sha256_after:
        raise RuntimeError("checkpoint changed during export")
    checkpoint_path = Path(checkpoint_path).resolve()
    current = _sha256(checkpoint_path)
    if current != checkpoint_sha256_before:
        raise RuntimeError("checkpoint hash does not bind current file")
    actual_count = sum(1 for _ in Path(topk_output).open("r", encoding="utf-8"))
    if actual_count != int(record_count):
        raise RuntimeError(f"Top-K record count changed while building manifest: {actual_count} != {record_count}")
    replay_payload = json.loads(Path(replay_result).read_text(encoding="utf-8"))
    if int(replay_payload.get("record_count", -1)) != actual_count:
        raise RuntimeError("replay result coverage does not match Top-K export")

    return {
        "schema_version": 1,
        "model": "pcgnn",
        "seed": int(seed),
        "top_k": int(top_k),
        "record_count": actual_count,
        "status": replay_payload.get("status", "checkpoint_replay_valid"),
        "native_report_test_reproduced": bool(
            replay_payload.get("native_report_comparison", {}).get("passed", False)
        ),
        "restored_state": "best_model.pt:model_state_dict",
        "checkpoint": {
            "path": str(checkpoint_path),
            "size": int(checkpoint_path.stat().st_size),
            "sha256_before": checkpoint_sha256_before,
            "sha256_after": checkpoint_sha256_after,
        },
        "report": _file_binding(Path(report_path)),
        "config": _file_binding(Path(config_path)),
        "split_files": [_file_binding(Path(path)) for path in split_paths],
        "script_files": [_file_binding(Path(path)) for path in script_paths],
        "topk_output": _file_binding(Path(topk_output)),
        "replay_result": _file_binding(Path(replay_result)),
    }


def _internal_to_raw_item_map(dataset, item_field: str) -> dict[int, int]:
    raw_tokens = dataset.field2id_token[item_field]
    mapping: dict[int, int] = {}
    for internal_id, token in enumerate(raw_tokens):
        if internal_id == 0:
            continue
        try:
            mapping[int(internal_id)] = int(str(token))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"PCGNN item token is not a MOOCCube integer id: {token!r}") from exc
    return mapping


def _validate_seed_identity(seed: int, report: Mapping[str, object], split_root: Path) -> None:
    report_seed = report.get("seed")
    if report_seed is not None and int(report_seed) != int(seed):
        raise RuntimeError(f"report seed mismatch: {report_seed} != {seed}")
    if not split_root.name.endswith(f"seed_{int(seed)}"):
        raise RuntimeError(f"split directory is not bound to seed {seed}: {split_root}")


def export_seed(args: argparse.Namespace) -> dict[str, object]:
    seed = int(args.seed)
    run_dir = (args.run_dir or DEFAULT_RUN_ROOT / f"mooccube_seed{seed}_full_formal_kg_warm").resolve()
    report_path = (args.report or run_dir / "pcgnn_strict_adapter_report.json").resolve()
    checkpoint_path = (args.checkpoint or run_dir / "checkpoints" / "best_model.pt").resolve()
    config_path = Path(args.config_file).resolve()
    pcgnn_root = Path(args.pcgnn_root).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split_root = (args.split_root or Path(str(report["split_root"]))).resolve()
    output_dir = (args.output_dir or run_dir / "p1_top20_export").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_output = output_dir / "pcgnn_top20.jsonl"
    replay_output = output_dir / "pcgnn_replay_result.json"
    manifest_output = output_dir / "export_manifest.json"

    _validate_seed_identity(seed, report, split_root)
    if str(report.get("dataset_name")) != str(args.dataset_name or report.get("dataset_name")):
        raise RuntimeError("dataset name override does not match the native report")
    dataset_name = str(args.dataset_name or report["dataset_name"])
    reported_checkpoint = report.get("best_checkpoint_path")
    if reported_checkpoint and Path(str(reported_checkpoint)).resolve() != checkpoint_path:
        raise RuntimeError("selected checkpoint does not match best_checkpoint_path in report")

    for required in (checkpoint_path, report_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    split_paths = [split_root / filename for filename in SPLIT_FILENAMES]
    for split_path in split_paths:
        if not split_path.is_file():
            raise FileNotFoundError(split_path)

    before_hash = _sha256(checkpoint_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_torch_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_rows, val_df, test_df = _read_split_rows(split_root)
    val_rows = _frame_rows(val_df[val_df["_split_source"].eq("strict_item_cold_val")])
    test_rows = _frame_rows(test_df[test_df["_split_source"].eq("strict_item_cold_test")])
    eval_batch_size = int(args.eval_batch_size or report["eval_batch_size"])
    train_batch_size = int(report["train_batch_size"])

    with local_pcgnn_recbole(pcgnn_root):
        from recbole.config import Config
        from recbole.data import Interaction, create_dataset, data_preparation
        from recbole.utils import get_model

        with clean_argv_for_recbole():
            config = Config(
                model="kg_model",
                dataset=dataset_name,
                config_file_list=[str(config_path)],
                config_dict=pcgnn_smoke_config_overrides(
                    train_batch_size=train_batch_size,
                    eval_batch_size=eval_batch_size,
                    device=device.type,
                ),
            )
        dataset = create_dataset(config)
        train_data, _, _ = data_preparation(config, dataset)
        token_map = {str(k): int(v) for k, v in dataset.field2token_id[config["ITEM_ID_FIELD"]].items()}
        internal_to_raw = _internal_to_raw_item_map(dataset, config["ITEM_ID_FIELD"])
        max_len = int(config["MAX_ITEM_LIST_LENGTH"])
        validation_examples = build_strict_eval_examples(train_rows, val_rows, token_map, max_len=max_len)
        examples = build_strict_eval_examples(train_rows, test_rows, token_map, max_len=max_len)
        user_seen_items = build_user_seen_items(train_rows, token_map)
        model = get_model(config["model"])(config, train_data).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("PCGNN checkpoint is missing model_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        # Reproduce the adapter's validation-before-test inference state before exporting test rankings.
        validation_replay = evaluate_pcgnn_full_item_macro(
            model,
            Interaction,
            validation_examples,
            user_seen_items,
            max_len=max_len,
            batch_size=eval_batch_size,
            k_list=(5, 10, 20),
            cold_threshold=1,
            device=device,
        )
        validation_comparison = compare_checkpoint_validation_to_report(
            validation_replay,
            report,
            checkpoint,
            tolerance=float(args.tolerance),
        )

        metadata = {
            "model": "pcgnn",
            "seed": seed,
            "protocol": str(report["protocol"]),
            "split_id": split_root.name,
            "dataset_name": dataset_name,
        }
        replay = replay_and_export_pcgnn(
            model=model,
            interaction_cls=Interaction,
            examples=examples,
            user_seen_items=user_seen_items,
            internal_to_raw=internal_to_raw,
            max_len=max_len,
            batch_size=eval_batch_size,
            top_k=int(args.top_k),
            output_path=topk_output,
            metadata=metadata,
            device=device,
        )

    comparison = compare_replay_to_report(
        replay,
        report,
        tolerance=float(args.tolerance),
    )
    validation = validate_pcgnn_topk_export(
        output_path=topk_output,
        examples=examples,
        user_seen_items=user_seen_items,
        internal_to_raw=internal_to_raw,
        top_k=int(args.top_k),
        expected_metadata=metadata,
        replay_metrics=dict(replay["metrics"]),
        tolerance=float(args.tolerance),
    )
    replay.update(
        {
            "status": (
                "checkpoint_replay_valid"
                if comparison["passed"]
                else "checkpoint_replay_valid_legacy_report_test_unreproducible"
            ),
            "model": "pcgnn",
            "seed": seed,
            "protocol": str(report["protocol"]),
            "split_root": str(split_root),
            "dataset_name": dataset_name,
            "device": str(device),
            "top_k": int(args.top_k),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_validation_metric": checkpoint.get("validation_metric"),
            "checkpoint_validation_score": float(checkpoint.get("validation_score", float("nan"))),
            "checkpoint_validation_replay": validation_replay,
            "checkpoint_validation_comparison": validation_comparison,
            "native_report_comparison": comparison,
            "export_validation": validation,
        }
    )
    _write_json_atomic(replay_output, replay)

    after_hash = _sha256(checkpoint_path)
    script_paths = [
        Path(__file__),
        ROOT / "ranking_topk_export.py",
        ROOT / "paper_aaai27" / "scripts" / "pcgnn_strict_adapter.py",
        ROOT / "paper_aaai27" / "scripts" / "priority_baseline_experiments.py",
        pcgnn_root / "recbole" / "model" / "sequential_recommender" / "kg_model.py",
    ]
    manifest = build_pcgnn_export_manifest(
        seed=seed,
        top_k=int(args.top_k),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256_before=before_hash,
        checkpoint_sha256_after=after_hash,
        report_path=report_path,
        config_path=config_path,
        split_paths=split_paths,
        script_paths=script_paths,
        topk_output=topk_output,
        replay_result=replay_output,
        record_count=int(replay["record_count"]),
    )
    _write_json_atomic(manifest_output, manifest)
    print(
        f"[P1-PCGNN] seed={seed} rows={replay['record_count']} "
        f"max_metric_drift={comparison['max_abs_metric_drift']:.3g} wrote={output_dir}",
        flush=True,
    )
    return replay


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only PCGNN checkpoint replay and Top-K export")
    parser.add_argument("--seed", type=int, required=True, choices=(2025, 2026, 2027))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pcgnn-root", type=Path, default=PCGNN_ROOT)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_PCGNN_CONFIG)
    parser.add_argument("--dataset-name")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args(argv)


def main() -> None:
    export_seed(parse_args())


if __name__ == "__main__":
    main()
