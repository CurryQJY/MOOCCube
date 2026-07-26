"""Serial validation-only campaign for auditable Stage-B component screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ckg_stageb_component_screen import (
    ComponentScreenConfig,
    component_source_files,
    run_component_screen,
)


_REPO_ROOT = Path(__file__).resolve().parent


def campaign_source_files() -> dict[str, Path]:
    """List controller code that can affect campaign promotion behavior."""
    return {
        "campaign_controller": Path(__file__).resolve(),
        "campaign_launcher": _REPO_ROOT / "run_ckg_stageb_component_campaign.ps1",
        **component_source_files(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign_source_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in campaign_source_files().items()}


@dataclass(frozen=True)
class CampaignStep:
    component_name: str
    seed: int
    requires_previous_acceptance: bool


@dataclass(frozen=True)
class CampaignConfig:
    campaign_id: str
    output_root: str = ""
    checkpoint_root: str = ""
    log_root: str = ""
    device: str = ""

    @classmethod
    def for_campaign_id(cls, campaign_id: str) -> "CampaignConfig":
        safe = str(campaign_id).strip()
        if not safe:
            raise ValueError("campaign_id must be non-empty")
        return cls(
            campaign_id=safe,
            output_root=f"outputs/ckg_stageb_component_campaign_{safe}",
            checkpoint_root=f"checkpoints/ckg_stageb_component_campaign_{safe}",
            log_root=f"background_logs/ckg_stageb_component_campaign_{safe}",
        )


def campaign_steps() -> tuple[CampaignStep, ...]:
    """Use seed 2027 for screening before the second strict-seed replication."""
    return (
        CampaignStep("soft_anchor_l2", 2027, False),
        CampaignStep("soft_anchor_l2", 2026, True),
    )


def validate_campaign_step_result(result: Mapping[str, Any], step: CampaignStep) -> None:
    """Require a runner result to attest to the exact queued strict step."""
    if not isinstance(result, Mapping):
        raise ValueError("campaign step result must be a mapping")
    expected_phase = "replication" if step.requires_previous_acceptance else "screen"
    if result.get("experiment") != "ckg_stageb_component_screen":
        raise ValueError("campaign step result experiment mismatch")
    if result.get("status") != "completed":
        raise ValueError("campaign step result status mismatch")
    if result.get("component_name") != step.component_name:
        raise ValueError("campaign step result component mismatch")
    if result.get("phase") != expected_phase:
        raise ValueError("campaign step result phase mismatch")
    if result.get("test_evaluation") is not False:
        raise ValueError("campaign step result test evaluation mismatch")
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("campaign step result lacks config")
    if int(config.get("seed", -1)) != int(step.seed):
        raise ValueError("campaign step result seed mismatch")
    if config.get("component_name") != step.component_name:
        raise ValueError("campaign step result config component mismatch")
    if config.get("phase") != expected_phase:
        raise ValueError("campaign step result config phase mismatch")
    if config.get("test_evaluation") is not False:
        raise ValueError("campaign step result config test evaluation mismatch")


def component_inventory() -> tuple[dict[str, str], ...]:
    """Record every audited legacy/configured behavior before launch."""
    return (
        {
            "component_name": "shared_content_projector",
            "status": "already_enabled",
            "reason": "The incumbent adapter already uses the shared zero-initialized content projector.",
        },
        {
            "component_name": "initial_content_anchor",
            "status": "already_enabled",
            "reason": "The incumbent adapter begins at the normalized frozen content representation.",
        },
        {
            "component_name": "hard_spherical_trust_region",
            "status": "already_enabled",
            "reason": "The incumbent caps the final content-relative chordal update at the fixed tau.",
        },
        {
            "component_name": "pseudo_cold_edge_masking",
            "status": "already_enabled",
            "reason": "The incumbent deterministically deletes every selected warm-course edge before training.",
        },
        {
            "component_name": "legacy_cbi_trust_cone",
            "status": "redundant",
            "reason": "The legacy cone is looser than the incumbent hard spherical trust region.",
        },
        {
            "component_name": "legacy_hot_id_content_gate",
            "status": "protocol_rejected",
            "reason": "It requires an item-ID tower and would mutate the frozen Hot bank.",
        },
        {
            "component_name": "legacy_id_content_infonce",
            "status": "protocol_rejected",
            "reason": "It depends on a mutable item-ID target absent from the content-only Stage-B contract.",
        },
        {
            "component_name": "legacy_simulator_rollout",
            "status": "protocol_rejected",
            "reason": "It is coupled to the Fast3 user-candidate state and legacy all-item inference family.",
        },
        {
            "component_name": "legacy_ppo",
            "status": "protocol_rejected",
            "reason": "Its policy objective is defined only for the legacy simulator trajectory.",
        },
        {
            "component_name": "legacy_course_reward",
            "status": "protocol_rejected",
            "reason": "Its reward is defined only for the legacy simulator and course-feedback state.",
        },
        {
            "component_name": "legacy_all_item_refinement",
            "status": "protocol_rejected",
            "reason": "It refines Hot and Cold banks together and violates frozen-Hot routing.",
        },
        {
            "component_name": "soft_anchor_l2",
            "status": "pending_screen",
            "reason": "A distinct final-space soft content anchor can be added without changing the Hot expert.",
        },
    )


def describe_campaign(cfg: CampaignConfig) -> dict[str, Any]:
    """Return the immutable launch contract without creating any artifacts."""
    return {
        "experiment": "ckg_stageb_component_campaign",
        "campaign_id": cfg.campaign_id,
        "scope": "validation_only",
        "test_evaluation": False,
        "execution": "serial",
        "components": list(component_inventory()),
        "steps": [asdict(step) for step in campaign_steps()],
        "acceptance": {
            "single_seed": "strict-2027 guard plus incumbent Cold N@10 gain >= .003 and nondecreasing Cold R@10",
            "final": "strict-2026-and-2027 guards plus mean Cold N@10 gain >= .003 and no Cold R@10 regression",
        },
    }


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _append_event_csv(path: Path, event: Mapping[str, Any]) -> None:
    fields = (
        "event", "campaign_id", "component_name", "seed", "phase", "record_status", "component_decision",
        "selected_epoch", "cold_r10", "cold_n10", "hot_r10", "hot_n10", "overall_r10", "overall_n10",
        "cold_r10_delta_vs_parent", "cold_n10_delta_vs_parent", "hot_r10_delta_vs_parent",
        "hot_n10_delta_vs_parent", "overall_r10_delta_vs_parent", "overall_n10_delta_vs_parent",
        "passes_retention_guards", "elapsed_seconds", "result_path", "parent_result_path",
        "parent_result_sha256", "hot_checkpoint_sha256", "stageb_source_sha256", "config_json", "error",
    )
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(event)


def _validate_campaign_config(cfg: CampaignConfig) -> None:
    if not str(cfg.campaign_id).strip():
        raise ValueError("campaign_id must be non-empty")
    roots = [_repo_path(cfg.output_root), _repo_path(cfg.checkpoint_root)]
    existing = [str(path) for path in roots if path.exists()]
    if existing:
        raise FileExistsError("component campaign requires fresh roots: " + ", ".join(existing))
    log_root = _repo_path(cfg.log_root)
    if log_root.exists() and not log_root.is_dir():
        raise ValueError("component campaign log root must be a directory")


def _result_record(
    *,
    campaign_id: str,
    step: CampaignStep,
    result: Mapping[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    selected = result.get("selected_validation_epoch")
    parent = result.get("parent_selected_validation_epoch")
    record: dict[str, Any] = {
        "campaign_id": campaign_id,
        "component_name": step.component_name,
        "seed": int(step.seed),
        "phase": "screen" if not step.requires_previous_acceptance else "replication",
        "record_status": str(result.get("status", "completed")),
        "component_decision": str(result.get("component_decision", "unknown")),
        "result_path": str(result_path),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "hot_checkpoint_contract": result.get("hot_checkpoint_contract"),
        "hot_checkpoint_sha256": (result.get("hot_checkpoint_contract") or {}).get("sha256"),
        "stageb_source_sha256": json.dumps(result.get("stageb_source_sha256"), sort_keys=True),
        "parent_result_path": result.get("parent_result_path"),
        "parent_result_sha256": result.get("parent_result_sha256"),
        "config_json": json.dumps(result.get("config"), sort_keys=True),
    }
    if isinstance(selected, Mapping):
        record["selected_epoch"] = int(selected["epoch"])
        record["passes_retention_guards"] = bool(selected.get("passes_retention_guards", False))
        for metric in ("cold_r10", "cold_n10", "hot_r10", "hot_n10", "overall_r10", "overall_n10"):
            record[metric] = float(selected[metric])
            if isinstance(parent, Mapping) and metric in parent:
                record[f"{metric}_delta_vs_parent"] = float(selected[metric]) - float(parent[metric])
    else:
        record["selected_epoch"] = None
        record["passes_retention_guards"] = False
    return record


def _write_ledgers(output_root: Path, payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("component ledger payload must be mutable")
    events = payload.setdefault("events", [])
    emitted = int(payload.get("_emitted_event_count", 0))
    if emitted < 0 or emitted > len(events):
        raise ValueError("component ledger event cursor is invalid")
    with (output_root / "component_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        for event in events[emitted:]:
            if not isinstance(event, Mapping):
                raise ValueError("component ledger events must be mappings")
            handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
            _append_event_csv(output_root / "component_ledger.csv", event)
    payload["_emitted_event_count"] = len(events)
    snapshot = {key: value for key, value in payload.items() if key != "_emitted_event_count"}
    _write_json(output_root / "component_ledger_snapshot.json", snapshot)


def finalize_component_decision(
    selected_by_seed: Mapping[int, Mapping[str, float | bool]],
    incumbent_by_seed: Mapping[int, Mapping[str, float]],
    *,
    mean_cold_n10_gain: float = 0.003,
) -> str:
    """Accept only a two-seed guarded mean Cold improvement."""
    expected_seeds = {2026, 2027}
    if set(selected_by_seed) != expected_seeds or set(incumbent_by_seed) != expected_seeds:
        return "rejected_incomplete_replication"
    if not all(bool(selected_by_seed[seed].get("passes_retention_guards", False)) for seed in expected_seeds):
        return "rejected_retention_guard"
    if any(
        float(selected_by_seed[seed]["cold_r10"]) < float(incumbent_by_seed[seed]["cold_r10"])
        for seed in expected_seeds
    ):
        return "rejected_cold_r10_regression"
    mean_gain = sum(
        float(selected_by_seed[seed]["cold_n10"]) - float(incumbent_by_seed[seed]["cold_n10"])
        for seed in expected_seeds
    ) / float(len(expected_seeds))
    if mean_gain < float(mean_cold_n10_gain):
        return "rejected_mean_cold_gain"
    return "accepted"


def build_final_incumbent(
    final_decision: str,
    *,
    component_name: str,
    campaign_id: str,
    result_paths: Mapping[int, str],
) -> dict[str, Any]:
    """Materialize the only configuration that may be treated as the next incumbent."""
    admitted = [str(component_name)] if str(final_decision) == "accepted" else []
    return {
        "schema_version": 1,
        "scope": "validation_only",
        "campaign_id": str(campaign_id),
        "base_model": "frozen_hot_edge_masked_pseudocold_shared_content_adapter",
        "always_enabled_components": [
            "shared_content_projector",
            "initial_content_anchor",
            "hard_spherical_trust_region",
            "pseudo_cold_edge_masking",
        ],
        "admitted_components": admitted,
        "decision": str(final_decision),
        "per_seed_component_results": {
            str(seed): str(path) for seed, path in sorted(result_paths.items())
        },
    }


def run_component_campaign(cfg: CampaignConfig) -> dict[str, Any]:
    """Run one auditable component candidate serially over the strict seeds."""
    _validate_campaign_config(cfg)
    campaign_source_sha256_before = _campaign_source_hashes()
    output_root = _repo_path(cfg.output_root)
    checkpoint_root = _repo_path(cfg.checkpoint_root)
    log_root = _repo_path(cfg.log_root)
    output_root.mkdir(parents=True, exist_ok=False)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=True)
    inventory = [dict(row) for row in component_inventory()]
    ledger: dict[str, Any] = {
        **describe_campaign(cfg),
        "campaign_status": "running",
        "inventory": inventory,
        "runs": [],
        "events": [
            {
                "event": "campaign_started",
                "campaign_id": cfg.campaign_id,
                "scope": "validation_only",
                "test_evaluation": False,
                "campaign_source_sha256": campaign_source_sha256_before,
            },
            *[
                {
                    "event": "component_inventory",
                    "campaign_id": cfg.campaign_id,
                    **row,
                }
                for row in inventory
            ],
        ],
        "final_component_decision": None,
    }
    _write_ledgers(output_root, ledger)
    completed_results: dict[int, Mapping[str, Any]] = {}
    failed = False
    for step in campaign_steps():
        if _campaign_source_hashes() != campaign_source_sha256_before:
            failed = True
            record = {
                "campaign_id": cfg.campaign_id,
                "component_name": step.component_name,
                "seed": int(step.seed),
                "phase": "replication" if step.requires_previous_acceptance else "screen",
                "record_status": "failed",
                "component_decision": "failed_source_drift",
                "error": "campaign runtime closure changed before the queued step",
            }
            ledger["runs"].append(record)
            ledger["events"].append({"event": "run_failed", **record})
            _write_ledgers(output_root, ledger)
            break
        if step.requires_previous_acceptance:
            first = completed_results.get(2027)
            if first is None or first.get("component_decision") != "provisionally_accepted":
                record = {
                    "campaign_id": cfg.campaign_id,
                    "component_name": step.component_name,
                    "seed": int(step.seed),
                    "phase": "replication",
                    "record_status": "skipped",
                    "component_decision": "skipped_previous_screen_not_accepted",
                }
                ledger["runs"].append(record)
                ledger["events"].append({"event": "run_skipped", **record})
                _write_ledgers(output_root, ledger)
                break
        phase = "replication" if step.requires_previous_acceptance else "screen"
        base_cfg = ComponentScreenConfig.for_seed(
            step.seed,
            component_name=step.component_name,
            phase=phase,
        )
        run_output = output_root / step.component_name / f"seed_{step.seed}"
        run_checkpoint = checkpoint_root / step.component_name / f"seed_{step.seed}"
        screen_cfg = replace(
            base_cfg,
            output_dir=str(run_output),
            checkpoint_dir=str(run_checkpoint),
            device=str(cfg.device),
        )
        result_path = run_output / "component_result.json"
        try:
            result = run_component_screen(screen_cfg)
            if _campaign_source_hashes() != campaign_source_sha256_before:
                raise RuntimeError("campaign runtime closure changed during the queued step")
            validate_campaign_step_result(result, step)
        except Exception as exc:
            failed = True
            record = {
                "campaign_id": cfg.campaign_id,
                "component_name": step.component_name,
                "seed": int(step.seed),
                "phase": "screen" if not step.requires_previous_acceptance else "replication",
                "record_status": "failed",
                "component_decision": "failed_execution_or_contract",
                "result_path": str(result_path),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            ledger["runs"].append(record)
            ledger["events"].append({"event": "run_failed", **record})
            _write_ledgers(output_root, ledger)
            break
        completed_results[int(step.seed)] = result
        record = _result_record(
            campaign_id=cfg.campaign_id,
            step=step,
            result=result,
            result_path=result_path,
        )
        ledger["runs"].append(record)
        ledger["events"].append({"event": "run_completed", **record})
        _write_ledgers(output_root, ledger)
    campaign_source_sha256_after = _campaign_source_hashes()
    if campaign_source_sha256_after != campaign_source_sha256_before:
        failed = True
        ledger["events"].append(
            {
                "event": "campaign_source_drift",
                "campaign_id": cfg.campaign_id,
                "campaign_source_sha256_before": campaign_source_sha256_before,
                "campaign_source_sha256_after": campaign_source_sha256_after,
            }
        )
    if failed:
        final_decision = "failed"
        campaign_status = "failed"
    elif set(completed_results) == {2026, 2027}:
        selected_by_seed = {
            seed: result["selected_validation_epoch"]
            for seed, result in completed_results.items()
            if isinstance(result.get("selected_validation_epoch"), Mapping)
        }
        incumbent_by_seed = {
            seed: result["parent_selected_validation_epoch"]
            for seed, result in completed_results.items()
            if isinstance(result.get("parent_selected_validation_epoch"), Mapping)
        }
        final_decision = finalize_component_decision(selected_by_seed, incumbent_by_seed)
        campaign_status = "completed"
    else:
        first = completed_results.get(2027)
        final_decision = (
            str(first.get("component_decision"))
            if first is not None
            else "rejected_no_completed_screen"
        )
        campaign_status = "completed"
    for row in inventory:
        if row["component_name"] == "soft_anchor_l2":
            row["status"] = final_decision
            row["decision"] = final_decision
    result_paths = {
        seed: str(output_root / "soft_anchor_l2" / f"seed_{seed}" / "component_result.json")
        for seed in completed_results
    }
    final_incumbent = build_final_incumbent(
        final_decision,
        component_name="soft_anchor_l2",
        campaign_id=cfg.campaign_id,
        result_paths=result_paths,
    )
    final_incumbent_path = output_root / "final_incumbent.json"
    _write_json(final_incumbent_path, final_incumbent)
    ledger["campaign_status"] = campaign_status
    ledger["final_component_decision"] = final_decision
    ledger["completed_seed_count"] = len(completed_results)
    ledger["final_incumbent_path"] = str(final_incumbent_path)
    ledger["events"].append(
        {
            "event": "campaign_completed",
            "campaign_id": cfg.campaign_id,
            "campaign_status": campaign_status,
            "final_component_decision": final_decision,
            "completed_seed_count": len(completed_results),
            "campaign_source_sha256": campaign_source_sha256_before,
        }
    )
    ledger["events"].append(
        {
            "event": "incumbent_promoted" if final_decision == "accepted" else "incumbent_unchanged",
            "campaign_id": cfg.campaign_id,
            "final_incumbent_path": str(final_incumbent_path),
            "admitted_components": final_incumbent["admitted_components"],
        }
    )
    _write_ledgers(output_root, ledger)
    summary = {
        "campaign_status": campaign_status,
        "final_component_decision": final_decision,
        "ledger_path": str(output_root / "component_ledger.jsonl"),
        "run_csv_path": str(output_root / "component_ledger.csv"),
        "log_root": str(log_root),
        "final_incumbent_path": str(final_incumbent_path),
        "campaign_source_sha256": campaign_source_sha256_before,
    }
    _write_json(output_root / "campaign_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict serial Stage-B component campaign")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--log-root", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> CampaignConfig:
    cfg = CampaignConfig.for_campaign_id(args.campaign_id)
    return replace(
        cfg,
        output_root=str(args.output_root or cfg.output_root),
        checkpoint_root=str(args.checkpoint_root or cfg.checkpoint_root),
        log_root=str(args.log_root or cfg.log_root),
        device=str(args.device),
    )


def main() -> None:
    args = _parser().parse_args()
    cfg = _config_from_args(args)
    if args.dry_run:
        print(json.dumps(describe_campaign(cfg), indent=2, sort_keys=True))
        return
    summary = run_component_campaign(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["campaign_status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
