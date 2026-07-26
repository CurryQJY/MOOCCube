"""Isolated static-training entry point for Cold-Consistent Knowledge Calibration.

It deliberately imports legacy code only for stable data splitting and course
metadata extraction.  Model training, checkpoint selection, full ranking, and
all run artifacts are owned by this C3K entry point.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader

import usim_feedback_fast3_content_delta as legacy
from fast3_delta.c3k_eval import build_c3k_item_bank, evaluate_c3k
from fast3_delta.c3k_model import C3KFeedbackUSIM
from fast3_delta.config import Fast3Config
from fast3_delta.pseudocold import build_pseudocold_plan, mask_user_item_history


_METHOD = "C3K"
_SCHEMA_VERSION = "c3k-static-v1"


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = int(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = float(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


@dataclasses.dataclass(frozen=True)
class C3KRunConfig:
    seed: int
    output_dir: str
    checkpoint_dir: str
    n_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    pseudo_cold_ratio: float
    pseudo_cold_min_popularity: float
    hot_tolerance: float
    min_delta: float
    item_block: int
    query_block: int
    test_history_policy: str
    validation_only: bool
    validation_max_rows: int
    gate_max: float
    consistency_weight: float
    gate_weight: float
    train_negatives: int
    warm_seen: float
    redundancy_threshold: float

    @classmethod
    def from_environment(cls) -> "C3KRunConfig":
        seed = _env_int("C3K_SEED", _env_int("USIM_STATIC_SEED", 2025), minimum=0)
        policy = os.environ.get("C3K_TEST_HISTORY", "train_only").strip().lower()
        if policy not in {"train_only", "train_val"}:
            raise ValueError("C3K_TEST_HISTORY must be train_only or train_val")
        return cls(
            seed=seed,
            output_dir=os.environ.get("C3K_OUTPUT_DIR", "outputs/c3k/manual").strip(),
            checkpoint_dir=os.environ.get("C3K_CHECKPOINT_DIR", "checkpoints/c3k/manual").strip(),
            n_epochs=_env_int("C3K_EPOCHS", 40, minimum=1),
            patience=_env_int("C3K_PATIENCE", 6, minimum=1),
            batch_size=_env_int("C3K_BATCH_SIZE", 512, minimum=2),
            learning_rate=_env_float("C3K_LR", 5e-4, minimum=0.0),
            pseudo_cold_ratio=_env_float("C3K_PSEUDO_COLD_RATIO", 0.10, minimum=0.0, maximum=1.0),
            pseudo_cold_min_popularity=_env_float("C3K_PSEUDO_COLD_MIN_POP", 1.0, minimum=0.0),
            hot_tolerance=_env_float("C3K_HOT_TOLERANCE", 0.003, minimum=0.0),
            min_delta=_env_float("C3K_MIN_DELTA", 1e-4, minimum=0.0),
            item_block=_env_int("C3K_ITEM_BLOCK", 128, minimum=1),
            query_block=_env_int("C3K_QUERY_BLOCK", 128, minimum=1),
            test_history_policy=policy,
            validation_only=os.environ.get("C3K_VALIDATION_ONLY", "0") == "1",
            validation_max_rows=_env_int("C3K_VALIDATION_MAX_ROWS", 0, minimum=0),
            gate_max=_env_float("C3K_GATE_MAX", 0.20, minimum=1e-8),
            consistency_weight=_env_float("C3K_CONSISTENCY_WEIGHT", 0.10, minimum=0.0),
            gate_weight=_env_float("C3K_GATE_WEIGHT", 0.001, minimum=0.0),
            train_negatives=_env_int("C3K_TRAIN_NEGATIVES", 16, minimum=1),
            warm_seen=_env_float("C3K_WARM_SEEN", 5.0, minimum=1.0),
            redundancy_threshold=_env_float(
                "C3K_REDUNDANCY_THRESHOLD", 0.70, minimum=0.0, maximum=0.99
            ),
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    temp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def c3k_source_hash() -> str:
    root = Path(__file__).resolve().parent
    source_files = [
        root / "c3k_static.py",
        root / "fast3_delta" / "c3k_model.py",
        root / "fast3_delta" / "c3k_eval.py",
        root / "run_c3k_3seed.ps1",
    ]
    payload = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in source_files
        if path.exists()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_c3k_manifest_payload(
    *,
    seed: int,
    source_hash: str,
    pseudo_cold: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Small public manifest constructor used by runner tests and audits."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "method": _METHOD,
        "seed": int(seed),
        "source_hash": str(source_hash),
        "id_dropout": "disabled",
        "training_inference_score": "shared_c3k_pair_score",
        "pseudo_cold": dict(pseudo_cold),
        "run_config": dict(run_config),
    }


def summarize_stable_epoch_timing(
    epoch_records: list[Mapping[str, Any]],
    *,
    seed: int,
    source_hash: str,
    selected_epoch: int | None,
) -> dict[str, Any]:
    """Summarize training throughput after excluding the first warm-up epoch."""
    stable = list(epoch_records[1:]) if len(epoch_records) > 1 else []
    durations = [float(record["seconds"]) for record in stable]
    sample_counts = [int(record["samples"]) for record in stable]
    batch_counts = [int(record["batches"]) for record in stable]
    mean_seconds = statistics.mean(durations) if durations else None
    mean_samples = statistics.mean(sample_counts) if sample_counts else None
    return {
        "schema_version": _SCHEMA_VERSION,
        "seed": int(seed),
        "source_hash": str(source_hash),
        "selected_epoch": None if selected_epoch is None else int(selected_epoch),
        "excluded_warmup_epoch": 1 if epoch_records else None,
        "stable_epoch_count": len(stable),
        "mean_seconds": mean_seconds,
        "std_seconds": statistics.pstdev(durations) if len(durations) > 1 else 0.0 if durations else None,
        "min_seconds": min(durations) if durations else None,
        "max_seconds": max(durations) if durations else None,
        "mean_batches": statistics.mean(batch_counts) if batch_counts else None,
        "mean_samples": mean_samples,
        "samples_per_second": (mean_samples / mean_seconds)
        if mean_seconds is not None and mean_seconds > 0.0 and mean_samples is not None
        else None,
        "epoch_records": [dict(record) for record in epoch_records],
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metric(metrics: Mapping[str, float] | None, key: str) -> float:
    return float((metrics or {}).get(key, 0.0))


def _combine_item_macro(
    cold: Mapping[str, float] | None,
    cold_count: int,
    hot: Mapping[str, float] | None,
    hot_count: int,
) -> dict[str, float]:
    keys = set((cold or {}).keys()) | set((hot or {}).keys())
    denominator = max(1, int(cold_count) + int(hot_count))
    return {
        key: (_metric(cold, key) * int(cold_count) + _metric(hot, key) * int(hot_count))
        / denominator
        for key in keys
    }


def select_validation_rows(frame: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    """Return a deterministic validation-only preflight subset when requested."""
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.sample(n=int(max_rows), random_state=int(seed)).sort_index()


def _passes_selector(
    calibrated: Mapping[str, Mapping[str, float]],
    uncalibrated: Mapping[str, Mapping[str, float]],
    run_config: C3KRunConfig,
) -> bool:
    return (
        _metric(calibrated.get("cold"), "N@10")
        > _metric(uncalibrated.get("cold"), "N@10") + run_config.min_delta
        and _metric(calibrated.get("hot"), "N@10")
        >= _metric(uncalibrated.get("hot"), "N@10") - run_config.hot_tolerance
    )


def _is_better_selector_candidate(
    candidate: Mapping[str, Mapping[str, float]],
    previous: Mapping[str, Mapping[str, float]] | None,
    min_delta: float,
) -> bool:
    if previous is None:
        return True
    candidate_cold_n = _metric(candidate.get("cold"), "N@10")
    previous_cold_n = _metric(previous.get("cold"), "N@10")
    if candidate_cold_n > previous_cold_n + min_delta:
        return True
    if abs(candidate_cold_n - previous_cold_n) <= min_delta:
        candidate_cold_r = _metric(candidate.get("cold"), "R@10")
        previous_cold_r = _metric(previous.get("cold"), "R@10")
        if candidate_cold_r > previous_cold_r + 1e-12:
            return True
        if abs(candidate_cold_r - previous_cold_r) <= 1e-12:
            return _metric(candidate.get("overall"), "N@10") > _metric(
                previous.get("overall"), "N@10"
            )
    return False


def _set_c3k_config(cfg: Fast3Config, run_config: C3KRunConfig) -> None:
    """Freeze C3K controls while leaving the legacy implementation untouched."""
    cfg.batch_size = int(run_config.batch_size)
    cfg.n_epochs = int(run_config.n_epochs)
    cfg.lr = float(run_config.learning_rate)
    cfg.dropout_prob = 0.0
    cfg.use_mixed_hard_neg = False
    cfg.mask_known_pos_neg = True
    cfg.mask_same_item_neg = True
    cfg.use_content_delta = False
    cfg.content_delta_train_on_id_dropout = False
    cfg.use_pseudo_cold_train = False
    cfg.train_force_cold = False
    cfg.ckg_rl_v1_enabled = False
    cfg.use_usim_refined_eval = False
    cfg.use_course_rerank = False
    cfg.use_course_reward = False
    cfg.use_prereq_aux_loss = False
    cfg.use_sage_lite = False
    cfg.use_sage_aux_loss = False
    cfg.use_cgrc_recon = False


def _c3k_trainable_parameters(model: C3KFeedbackUSIM) -> list[torch.nn.Parameter]:
    forbidden_prefixes = (
        "agent.",
        "cgrc_recon_mlp.",
        "sage_gate_",
        "sage_score_gate_",
        "content_delta",
        "llm_proj.",
    )
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(forbidden_prefixes)
    ]


def _build_course_artifacts(df: pd.DataFrame, train_df: pd.DataFrame, cfg: Fast3Config):
    source = os.environ.get("C3K_ARTIFACT_SOURCE", "all_metadata").strip().lower()
    if source not in {"all_metadata", "train"}:
        raise ValueError("C3K_ARTIFACT_SOURCE must be all_metadata or train")
    artifact_df = df if source == "all_metadata" else train_df
    if not cfg.feedback_load_course_artifacts:
        return None, legacy._empty_course_stats(cfg.n_items), source
    artifacts, stats = legacy.build_course_artifacts(
        artifact_df,
        cfg.n_items,
        relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
        prereq_min_support=cfg.prereq_min_support,
        prereq_max_per_item=cfg.prereq_max_per_item,
        prereq_min_items=cfg.prereq_min_items,
        prereq_max_forward=cfg.prereq_max_forward,
    )
    return artifacts, stats, source


def _evaluate_partitioned_validation(
    model: C3KFeedbackUSIM,
    loader: DataLoader,
    device: torch.device,
    history: Mapping[int, set[int]],
    run_config: C3KRunConfig,
    *,
    calibration: bool,
) -> tuple[dict[str, Mapping[str, float]], dict[str, Any]]:
    bank = build_c3k_item_bank(model, device, item_batch=run_config.item_block)
    cold, cold_count, cold_timing = evaluate_c3k(
        model,
        loader,
        device,
        item_bank=bank,
        eval_type="cold",
        user_seen_items=history,
        item_block=run_config.item_block,
        query_block=run_config.query_block,
        calibration=calibration,
    )
    hot, hot_count, hot_timing = evaluate_c3k(
        model,
        loader,
        device,
        item_bank=bank,
        eval_type="hot",
        user_seen_items=history,
        item_block=run_config.item_block,
        query_block=run_config.query_block,
        calibration=calibration,
    )
    overall = _combine_item_macro(cold, cold_count, hot, hot_count)
    return (
        {"cold": cold or {}, "hot": hot or {}, "overall": overall},
        {
            "item_bank_seconds": bank.item_bank_seconds,
            "cold": cold_timing,
            "hot": hot_timing,
            "cold_item_count": cold_count,
            "hot_item_count": hot_count,
        },
    )


def _test_metrics(
    model: C3KFeedbackUSIM,
    loader: DataLoader,
    device: torch.device,
    history: Mapping[int, set[int]],
    run_config: C3KRunConfig,
    output_dir: Path,
) -> tuple[dict[str, Mapping[str, float]], dict[str, Any]]:
    bank = build_c3k_item_bank(model, device, item_batch=run_config.item_block)
    cold, cold_count, cold_timing = evaluate_c3k(
        model,
        loader,
        device,
        item_bank=bank,
        eval_type="cold",
        user_seen_items=history,
        item_block=run_config.item_block,
        query_block=run_config.query_block,
        export_item_metrics_path=str(output_dir / "per_item_cold.csv"),
    )
    hot, hot_count, hot_timing = evaluate_c3k(
        model,
        loader,
        device,
        item_bank=bank,
        eval_type="hot",
        user_seen_items=history,
        item_block=run_config.item_block,
        query_block=run_config.query_block,
        export_item_metrics_path=str(output_dir / "per_item_hot.csv"),
    )
    overall, overall_count, overall_timing = evaluate_c3k(
        model,
        loader,
        device,
        item_bank=bank,
        eval_type="all",
        user_seen_items=history,
        item_block=run_config.item_block,
        query_block=run_config.query_block,
        export_item_metrics_path=str(output_dir / "per_item_overall.csv"),
    )
    return (
        {"cold": cold or {}, "hot": hot or {}, "overall": overall or {}},
        {
            "item_bank_seconds": bank.item_bank_seconds,
            "production_full_catalog": overall_timing,
            "cold_metric_replay": cold_timing,
            "hot_metric_replay": hot_timing,
            "cold_item_count": cold_count,
            "hot_item_count": hot_count,
            "overall_item_count": overall_count,
        },
    )


def run_c3k_static_experiment(
    df: pd.DataFrame,
    cfg: Fast3Config,
    device: torch.device,
    content_emb: torch.Tensor,
) -> dict[str, Any]:
    """Run a complete C3K static protocol without using the legacy trainer."""
    run_config = C3KRunConfig.from_environment()
    output_dir = Path(run_config.output_dir).resolve()
    checkpoint_dir = Path(run_config.checkpoint_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    _set_c3k_config(cfg, run_config)
    legacy.setup_seed(run_config.seed)

    train_df, val_df, test_df, split_info = legacy._static_split_df(df)
    train_df, val_df, test_df, item_train_popularity = legacy._apply_train_popularity(
        train_df, val_df, test_df, cfg
    )
    pseudo_plan = build_pseudocold_plan(
        item_train_popularity,
        ratio=run_config.pseudo_cold_ratio,
        min_popularity=run_config.pseudo_cold_min_popularity,
        cold_threshold=float(cfg.cold_threshold),
        seed=run_config.seed,
        n_strata=4,
    )
    if not pseudo_plan.selected_item_ids:
        raise RuntimeError("C3K pseudo-cold plan selected no eligible warm courses")

    course_artifacts, course_stats, artifact_source = _build_course_artifacts(df, train_df, cfg)
    model = C3KFeedbackUSIM(cfg, content_emb).to(device)
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(item_train_popularity)
    model.set_pseudo_cold_plan(pseudo_plan)
    optimizer = torch.optim.Adam(_c3k_trainable_parameters(model), lr=run_config.learning_rate)

    train_history = legacy._add_user_seen_from_df({}, train_df)
    masked_train_history = mask_user_item_history(train_history, pseudo_plan.selected_mask)
    val_eval_df = select_validation_rows(
        val_df, max_rows=run_config.validation_max_rows, seed=run_config.seed
    )
    train_loader = DataLoader(
        legacy.StreamDataset(train_df, {}),
        batch_size=run_config.batch_size,
        shuffle=True,
        collate_fn=legacy.collate_fn,
    )
    val_loader = DataLoader(
        legacy.StreamDataset(val_eval_df, {}), batch_size=2048, shuffle=False, collate_fn=legacy.collate_fn
    )
    test_loader = DataLoader(
        legacy.StreamDataset(test_df, {}), batch_size=2048, shuffle=False, collate_fn=legacy.collate_fn
    )

    source_hash = c3k_source_hash()
    manifest = build_c3k_manifest_payload(
        seed=run_config.seed,
        source_hash=source_hash,
        pseudo_cold={
            "selection_source": "train_popularity_only",
            "plan_hash": pseudo_plan.plan_hash,
            "selected_item_count": len(pseudo_plan.selected_item_ids),
            "ratio": run_config.pseudo_cold_ratio,
            "min_popularity": run_config.pseudo_cold_min_popularity,
        },
        run_config=dataclasses.asdict(run_config),
    )
    manifest.update(
        {
            "split": split_info,
            "artifact_source": artifact_source,
            "course_stats": course_stats,
            "cold_definition": "strict item cold iff train_popularity < cold_threshold",
            "timing_policy": {
                "stable_train": "exclude first completed epoch; include train loop only",
                "inference": "checkpoint-restored full-catalog overall score, separated from metric replays",
            },
        }
    )
    _write_json(output_dir / "c3k_manifest.json", manifest)
    _write_json(output_dir / "pseudocold_plan.json", pseudo_plan.to_dict())

    print(
        f"[C3K] seed={run_config.seed} device={device} train={len(train_df)} val={len(val_eval_df)}/{len(val_df)} test={len(test_df)}"
    )
    print(
        f"[C3K] pseudo-cold={len(pseudo_plan.selected_item_ids)} plan={pseudo_plan.plan_hash[:12]} "
        f"ratio={run_config.pseudo_cold_ratio:.3f}; ID dropout disabled"
    )
    print(
        f"[C3K] epochs={run_config.n_epochs} patience={run_config.patience} "
        f"batch={run_config.batch_size} lr={run_config.learning_rate:.2e} "
        f"hot_guard={run_config.hot_tolerance:.4f}"
    )

    best_epoch: int | None = None
    best_selector: Mapping[str, Mapping[str, float]] | None = None
    best_path = checkpoint_dir / "best.pt"
    no_improvement = 0
    history_rows: list[dict[str, Any]] = []
    epoch_timing: list[dict[str, Any]] = []
    model.set_user_seen_index(masked_train_history)

    for epoch in range(1, run_config.n_epochs + 1):
        model.train()
        _sync(device)
        epoch_start = time.perf_counter()
        epoch_samples = 0
        epoch_batches = 0
        loss_sum = 0.0
        diagnostics_sum: dict[str, float] = {}
        for batch, pop, llm in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = model(
                batch,
                pop,
                llm,
                user_seen_items=masked_train_history,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"C3K produced a non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            epoch_samples += int(batch["u"].numel())
            epoch_batches += 1
            loss_sum += float(loss.detach().item())
            for key, value in diagnostics.items():
                diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + float(value)
        _sync(device)
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "seconds": float(epoch_seconds),
            "batches": epoch_batches,
            "samples": epoch_samples,
        }
        epoch_timing.append(epoch_record)

        model.set_user_seen_index(train_history)
        calibrated, calibrated_timing = _evaluate_partitioned_validation(
            model, val_loader, device, train_history, run_config, calibration=True
        )
        uncalibrated, uncalibrated_timing = _evaluate_partitioned_validation(
            model, val_loader, device, train_history, run_config, calibration=False
        )
        passes_guard = _passes_selector(calibrated, uncalibrated, run_config)
        updated = False
        if passes_guard and _is_better_selector_candidate(
            calibrated, best_selector, run_config.min_delta
        ):
            torch.save(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "method": _METHOD,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "selector": calibrated,
                    "source_hash": source_hash,
                    "pseudo_cold_plan_hash": pseudo_plan.plan_hash,
                },
                best_path,
            )
            best_epoch = epoch
            best_selector = calibrated
            no_improvement = 0
            updated = True
        else:
            no_improvement += 1

        epoch_summary = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, epoch_batches),
            "train_diagnostics": {
                key: value / max(1, epoch_batches) for key, value in diagnostics_sum.items()
            },
            "train_timing": epoch_record,
            "calibrated_validation": calibrated,
            "uncalibrated_validation": uncalibrated,
            "calibrated_validation_timing": calibrated_timing,
            "uncalibrated_validation_timing": uncalibrated_timing,
            "passes_hot_guard": passes_guard,
            "selector_updated": updated,
        }
        history_rows.append(epoch_summary)
        _write_json(output_dir / "training_history.json", {"epochs": history_rows})
        print(
            f"[C3K] epoch={epoch}/{run_config.n_epochs} train={epoch_seconds:.2f}s "
            f"loss={epoch_summary['train_loss']:.5f} coldN={_metric(calibrated['cold'], 'N@10'):.4f} "
            f"baseColdN={_metric(uncalibrated['cold'], 'N@10'):.4f} "
            f"hotN={_metric(calibrated['hot'], 'N@10'):.4f} "
            f"baseHotN={_metric(uncalibrated['hot'], 'N@10'):.4f} guard={passes_guard} update={updated}"
        )
        model.set_user_seen_index(masked_train_history)
        if no_improvement >= run_config.patience:
            print(f"[C3K] early stop after {epoch} epochs")
            break

    stable_timing = summarize_stable_epoch_timing(
        epoch_timing,
        seed=run_config.seed,
        source_hash=source_hash,
        selected_epoch=best_epoch,
    )
    _write_json(output_dir / "stable_epoch_timing.json", stable_timing)

    if best_epoch is None:
        result = {
            "schema_version": _SCHEMA_VERSION,
            "method": _METHOD,
            "status": "no_checkpoint_passed_validation_guard",
            "source_hash": source_hash,
            "pseudo_cold_plan_hash": pseudo_plan.plan_hash,
            "stable_epoch_timing": stable_timing,
        }
        _write_json(output_dir / "c3k_result.json", result)
        print("[C3K] no validation checkpoint passed the frozen hot-retention guard; test was not run")
        return result

    if run_config.validation_only:
        result = {
            "schema_version": _SCHEMA_VERSION,
            "method": _METHOD,
            "status": "validation_only_completed",
            "source_hash": source_hash,
            "selected_epoch": best_epoch,
            "selector": best_selector,
            "stable_epoch_timing": stable_timing,
            "pseudo_cold_plan_hash": pseudo_plan.plan_hash,
        }
        _write_json(output_dir / "c3k_result.json", result)
        print(f"[C3K] validation-only preflight completed selected_epoch={best_epoch}")
        return result

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_history = legacy._clone_user_seen(train_history)
    if run_config.test_history_policy == "train_val":
        legacy._add_user_seen_from_df(test_history, val_df)
    model.set_user_seen_index(test_history)
    test_metrics, inference_timing = _test_metrics(
        model, test_loader, device, test_history, run_config, output_dir
    )
    _write_json(output_dir / "inference_timing.json", inference_timing)
    result = {
        "schema_version": _SCHEMA_VERSION,
        "method": _METHOD,
        "status": "completed",
        "source_hash": source_hash,
        "selected_epoch": best_epoch,
        "selector": best_selector,
        "test": test_metrics,
        "stable_epoch_timing": stable_timing,
        "inference_timing": inference_timing,
        "pseudo_cold_plan_hash": pseudo_plan.plan_hash,
    }
    _write_json(output_dir / "c3k_result.json", result)
    print(
        f"[C3K] completed selected_epoch={best_epoch} "
        f"coldN@10={_metric(test_metrics['cold'], 'N@10'):.4f} "
        f"hotN@10={_metric(test_metrics['hot'], 'N@10'):.4f} "
        f"overallN@10={_metric(test_metrics['overall'], 'N@10'):.4f}"
    )
    return result


def main() -> None:
    run_config = C3KRunConfig.from_environment()
    if os.environ.get("C3K_DRY_RUN", "0") == "1":
        print(json.dumps(dataclasses.asdict(run_config), indent=2, sort_keys=True))
        return
    data_dir = Path(os.environ.get("USIM_DATA_DIR", "processed_data_hin"))
    stream_path = data_dir / "stream_data.pkl"
    if not stream_path.exists():
        raise FileNotFoundError(f"C3K data file not found: {stream_path}")
    with (data_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    df = pd.read_pickle(stream_path)
    content_emb = torch.load(data_dir / "content_emb.pt", map_location="cpu")
    cfg = Fast3Config(meta["n_users"], meta["n_items"], int(content_emb.shape[1]))
    device = legacy._resolve_torch_device()
    run_c3k_static_experiment(df, cfg, device, content_emb,)


if __name__ == "__main__":
    main()


__all__ = [
    "C3KRunConfig",
    "build_c3k_manifest_payload",
    "c3k_source_hash",
    "run_c3k_static_experiment",
    "select_validation_rows",
    "summarize_stable_epoch_timing",
]
