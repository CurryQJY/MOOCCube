"""Frozen, test-only replay for registered Hot graph-expert checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader

import cgrc_paper_static_hin as cgrc
from hin_data_common import InteractionDataset, build_user_seen, collate_interactions
from hin_eval_common import evaluate_embedding_ranker


_REPO_ROOT = Path(__file__).resolve().parent
_REGISTERED_SEEDS = (2025, 2026, 2027)
_REGISTERED_CHECKPOINT_SHA256 = {
    2025: "a41c466d8244fa08e043cfd8dc0289e3f99f5dd5af351f4b891d62780a2c258f",
    2026: "97c899eeffdca01e446389a0f5b78f1d9a0b56f6fe38b23ab6da62a1b3c1564d",
    2027: "dbc0548139ee6a7c2bc4a15044201c73098c277f7accaf5d4a53b844d443bfd4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_replay_inputs(
    data_dir: str | Path, split_dir: str | Path
) -> tuple[dict[str, Any], torch.Tensor, pd.DataFrame, pd.DataFrame]:
    """Load exactly metadata, content, train, and held-out test inputs."""
    data_root = Path(data_dir)
    split_root = Path(split_dir)
    paths = {
        "meta": data_root / "meta.json",
        "content": data_root / "content_emb.pt",
        "train": split_root / "static_train.pkl",
        "test": split_root / "static_test.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Hot test replay input: " + ", ".join(missing))
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    try:
        content = torch.load(paths["content"], map_location="cpu", weights_only=False)
    except TypeError:
        content = torch.load(paths["content"], map_location="cpu")
    if not isinstance(content, torch.Tensor):
        raise ValueError("content embedding must be a tensor")
    train = pd.read_pickle(paths["train"]).copy()
    test = pd.read_pickle(paths["test"]).copy()
    required = {"u_idx", "i_idx"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError("train and test frames must contain u_idx and i_idx")
    counts = train["i_idx"].astype(int).value_counts().astype(int)
    for frame in (train, test):
        frame["popularity"] = frame["i_idx"].astype(int).map(counts).fillna(0).astype(int)
    return meta, content.float(), train, test


def build_test_result(
    *,
    frozen: dict[str, Any],
    cold: dict[str, float],
    cold_count: int,
    hot: dict[str, float],
    hot_count: int,
) -> dict[str, Any]:
    """Build a test report using item-macro Cold/Hot counts for Overall."""
    cold_count = int(cold_count)
    hot_count = int(hot_count)
    total = cold_count + hot_count
    if total <= 0:
        raise ValueError("test replay requires at least one evaluated item")
    result: dict[str, Any] = {
        "test_evaluation": True,
        "selected_validation_epoch": int(frozen["epoch"]),
        "selected_checkpoint_sha256": str(frozen["sha256"]),
        "cold_item_count": cold_count,
        "hot_item_count": hot_count,
        "overall_item_count": total,
    }
    for metric in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20"):
        field = metric.lower().replace("@", "")
        cold_value = float(cold.get(metric, 0.0))
        hot_value = float(hot.get(metric, 0.0))
        result[f"cold_{field}"] = cold_value
        result[f"hot_{field}"] = hot_value
        result[f"overall_{field}"] = (
            cold_value * cold_count + hot_value * hot_count
        ) / total
    return result


@dataclass(frozen=True)
class HotTestReplayConfig:
    """Immutable inputs for one frozen, held-out Hot-expert test replay."""

    seed: int
    data_dir: str | Path
    split_dir: str | Path
    source_result_path: str | Path
    checkpoint_dir: str | Path
    output_dir: str | Path
    device: str = ""
    expected_sha256: str = ""


def registered_test_replay_configs(
    *, output_root: str | Path, device: str = ""
) -> list[HotTestReplayConfig]:
    """Register the three validation-selected Hot checkpoints for one test replay."""
    output_root = Path(output_root)
    data_dir = _REPO_ROOT / "processed_data_hin_clean_pop5"
    configs: list[HotTestReplayConfig] = []
    for seed in _REGISTERED_SEEDS:
        split_dir = (
            _REPO_ROOT
            / "outputs"
            / "content_delta_pop5"
            / "static_item_cold_balanced"
            / f"strict_item_cold_balanced_thr1_seed_{seed}"
        )
        if seed == 2025:
            source_root = _REPO_ROOT / "outputs" / "ckg_hot_graph_preflight_seed2025"
            checkpoint_dir = _REPO_ROOT / "checkpoints" / "ckg_hot_graph_preflight_seed2025"
        else:
            source_root = _REPO_ROOT / "outputs" / f"ckg_hot_graph_preflight_replication_seed{seed}"
            checkpoint_dir = _REPO_ROOT / "checkpoints" / f"ckg_hot_graph_preflight_replication_seed{seed}"
        configs.append(
            HotTestReplayConfig(
                seed=seed,
                data_dir=data_dir,
                split_dir=split_dir,
                source_result_path=source_root / "preflight_result.json",
                checkpoint_dir=checkpoint_dir,
                output_dir=output_root / f"seed{seed}",
                device=device,
                expected_sha256=_REGISTERED_CHECKPOINT_SHA256[seed],
            )
        )
    return configs


def build_test_replay_dry_run(*, output_root: str | Path, device: str = "") -> dict[str, Any]:
    """Describe the fixed test-only contract without reading a checkpoint or test row."""
    configs = registered_test_replay_configs(output_root=output_root, device=device)
    return {
        "experiment": "ckg_hot_graph_test_replay",
        "test_evaluation": True,
        "test_history": "train_only",
        "seeds": [cfg.seed for cfg in configs],
        "runs": [
            {
                "seed": cfg.seed,
                "selection": "frozen_validation_checkpoint",
                "source_result": str(cfg.source_result_path),
                "checkpoint_dir": str(cfg.checkpoint_dir),
                "expected_checkpoint_sha256": cfg.expected_sha256,
                "split_dir": str(cfg.split_dir),
                "output_dir": str(cfg.output_dir),
            }
            for cfg in configs
        ],
    }


def _resolve_device(requested: str) -> torch.device:
    raw = str(requested).strip().lower()
    if raw == "cpu":
        return torch.device("cpu")
    if raw.startswith("cuda") and torch.cuda.is_available():
        return torch.device(raw)
    if raw:
        raise RuntimeError(f"requested unavailable device: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not isinstance(state.get("model_state"), dict):
        raise ValueError("selected checkpoint must contain model_state")
    if not isinstance(state.get("config"), dict):
        raise ValueError("selected checkpoint must contain config")
    return state


def _evaluate_frozen_checkpoint(
    *,
    model: cgrc.CGRCNet,
    layers_full: int,
    cold_threshold: int,
    device: torch.device,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, float], int, dict[str, float], int]:
    train_seen = build_user_seen(train_df)
    test_loader = DataLoader(
        InteractionDataset(test_df),
        batch_size=4096,
        shuffle=False,
        collate_fn=collate_interactions,
    )
    r_base = cgrc._build_interaction_csr(train_df, model.n_users, model.n_items)
    sparse_full = cgrc._sparse_adj_tensor(
        cgrc._normalize_graph_mat(cgrc._bip_adj_from_R(r_base, model.n_users, model.n_items)),
        device,
    )
    model.eval()
    with torch.no_grad():
        all_u, all_i = cgrc._lightgcn_mean_all_layers(
            sparse_full,
            model.user_emb,
            model.item_x(),
            model.n_users,
            int(layers_full),
        )
        all_u = F.normalize(all_u, dim=1)
        all_i = F.normalize(all_i, dim=1)
        get_user = lambda batch: all_u[batch["u"]]
        cold, cold_count = evaluate_embedding_ranker(
            test_loader,
            device=device,
            n_items=model.n_items,
            cold_threshold=int(cold_threshold),
            get_user_vectors_fn=get_user,
            all_item_vectors=all_i,
            k_list=(5, 10, 20),
            eval_type="cold",
            full_ranking=True,
            user_seen_items=train_seen,
            average_mode="item_macro",
            export_item_metrics_path=str(output_dir / "per_item_test_cold.csv"),
        )
        hot, hot_count = evaluate_embedding_ranker(
            test_loader,
            device=device,
            n_items=model.n_items,
            cold_threshold=int(cold_threshold),
            get_user_vectors_fn=get_user,
            all_item_vectors=all_i,
            k_list=(5, 10, 20),
            eval_type="hot",
            full_ranking=True,
            user_seen_items=train_seen,
            average_mode="item_macro",
            export_item_metrics_path=str(output_dir / "per_item_test_hot.csv"),
        )
    if cold is None or hot is None:
        raise RuntimeError("test split did not yield both Cold and Hot item-macro metrics")
    return cold, int(cold_count), hot, int(hot_count)


def run_test_replay(cfg: HotTestReplayConfig) -> dict[str, Any]:
    """Evaluate one validation-selected Hot checkpoint exactly once on test."""
    output_dir = Path(cfg.output_dir)
    result_path = output_dir / "test_result.json"
    if result_path.exists():
        raise FileExistsError(f"test replay result already exists: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = resolve_frozen_checkpoint(
        result_path=cfg.source_result_path,
        checkpoint_dir=cfg.checkpoint_dir,
        seed=cfg.seed,
        expected_sha256=cfg.expected_sha256 or None,
    )
    state = _load_checkpoint_state(Path(frozen["path"]))
    checkpoint_cfg = state["config"]
    if int(checkpoint_cfg.get("seed", -1)) != int(cfg.seed):
        raise ValueError("selected checkpoint seed does not match replay seed")
    meta, content, train_df, test_df = load_test_replay_inputs(cfg.data_dir, cfg.split_dir)
    device = _resolve_device(cfg.device)
    model = cgrc.CGRCNet(
        int(meta["n_users"]),
        int(meta["n_items"]),
        int(content.shape[1]),
        int(checkpoint_cfg["emb_dim"]),
        int(checkpoint_cfg["mlp_hidden"]),
        content,
    ).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    cold, cold_count, hot, hot_count = _evaluate_frozen_checkpoint(
        model=model,
        layers_full=int(checkpoint_cfg["layers_full"]),
        cold_threshold=int(checkpoint_cfg.get("cold_threshold", 1)),
        device=device,
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir,
    )
    result = build_test_result(
        frozen=frozen,
        cold=cold,
        cold_count=cold_count,
        hot=hot,
        hot_count=hot_count,
    )
    result.update(
        {
            "experiment": "ckg_hot_graph_test_replay",
            "seed": int(cfg.seed),
            "input_protocol": "meta_content_static_train_static_test",
            "test_history": "train_only",
            "source_result_path": str(Path(cfg.source_result_path)),
            "selected_checkpoint_path": str(Path(frozen["path"])),
        }
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_registered_test_replays(
    *, output_root: str | Path, device: str = ""
) -> dict[str, Any]:
    """Run the three registered frozen checkpoints once on their held-out tests."""
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"test replay requires a fresh output root: {output_root}")
    output_root.mkdir(parents=True)
    configs = registered_test_replay_configs(output_root=output_root, device=device)
    manifest: dict[str, Any] = {
        "experiment": "ckg_hot_graph_test_replay",
        "test_evaluation": True,
        "test_history": "train_only",
        "seeds": [cfg.seed for cfg in configs],
        "sources": [
            {
                "seed": cfg.seed,
                "source_result_path": str(Path(cfg.source_result_path)),
                "checkpoint_dir": str(Path(cfg.checkpoint_dir)),
                "expected_checkpoint_sha256": cfg.expected_sha256,
                "seed_output_dir": str(Path(cfg.output_dir)),
            }
            for cfg in configs
        ],
    }
    manifest_path = output_root / "test_replay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results = [run_test_replay(cfg) for cfg in configs]
    summary = {
        "experiment": "ckg_hot_graph_test_replay",
        "test_evaluation": True,
        "test_history": "train_only",
        "seeds": [cfg.seed for cfg in configs],
        "runs": results,
    }
    (output_root / "test_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(output_root / "test_replay_summary.csv", index=False)
    manifest["completed_runs"] = [
        {
            "seed": result["seed"],
            "selected_validation_epoch": result["selected_validation_epoch"],
            "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
        }
        for result in results
    ]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def resolve_frozen_checkpoint(
    *,
    result_path: str | Path,
    checkpoint_dir: str | Path,
    seed: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve only the validation-registered checkpoint and verify its digest."""
    result_path = Path(result_path)
    checkpoint_dir = Path(checkpoint_dir)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if bool(result.get("test_evaluation")):
        raise ValueError("test replay requires a validation-only source result")
    if not bool(result.get("passed_hot_preflight")):
        raise ValueError("test replay requires a passed Hot preflight")
    selected = result.get("selected_validation_epoch") or {}
    selected_epoch = int(selected.get("epoch", -1))
    contract = result.get("selected_checkpoint_contract")
    if not isinstance(contract, dict):
        source_config = result.get("config") or {}
        if int(source_config.get("seed", -1)) != int(seed):
            raise ValueError("legacy source result seed does not match replay seed")
        contract = {
            "seed": int(seed),
            "epoch": selected_epoch,
            "relative_path": f"epoch_{selected_epoch:03d}.pt",
        }
    if int(contract.get("seed", -1)) != int(seed):
        raise ValueError("selected checkpoint seed does not match replay seed")
    epoch = int(contract.get("epoch", -1))
    if epoch < 1 or selected_epoch != epoch:
        raise ValueError("selected checkpoint epoch does not match validation selection")
    relative_path = str(contract.get("relative_path", ""))
    path = checkpoint_dir / relative_path
    if path.name != f"epoch_{epoch:03d}.pt" or not path.is_file():
        raise ValueError("selected checkpoint path is invalid")
    actual = _sha256(path)
    expected = str(contract.get("sha256", "")).lower()
    registered = str(expected_sha256 or "").lower()
    if registered and expected and expected != registered:
        raise ValueError("registered checkpoint sha256 does not match its source contract")
    expected = registered or expected or actual
    if expected != actual:
        raise ValueError("selected checkpoint sha256 does not match its registered contract")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    checkpoint_config = state.get("config") if isinstance(state, dict) else None
    if not isinstance(checkpoint_config, dict) or int(checkpoint_config.get("seed", -1)) != int(seed):
        raise ValueError("selected checkpoint seed does not match replay seed")
    if int(state.get("epoch", -1)) != epoch:
        raise ValueError("selected checkpoint epoch does not match validation selection")
    return {"epoch": epoch, "path": path, "sha256": actual}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "outputs" / "ckg_hot_graph_test_replay_3seed",
        help="New directory for the frozen test replay outputs.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Evaluation device, e.g. cuda:0 or cpu. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the immutable test contract without reading test rows or writing outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                build_test_replay_dry_run(output_root=args.output_root, device=args.device),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = run_registered_test_replays(
        output_root=args.output_root,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
