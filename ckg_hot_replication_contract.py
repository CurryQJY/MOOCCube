"""Register the actual selected Hot checkpoint for a strict replication seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import cgrc_paper_static_hin as cgrc
from ckg_hot_graph_preflight_replication import load_hot_replication_inputs


FIXED_TRUST_TAU = 0.24929234
_REPLICATION_SEEDS = (2026, 2027)
_ARCHITECTURE = {"emb_dim": 64, "mlp_hidden": 64, "layers_full": 2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checkpoint(path: Path, *, expected_seed: int) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("selected Hot checkpoint must contain a model_state mapping")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("selected Hot checkpoint must contain its config")
    if int(config.get("seed", -1)) != int(expected_seed):
        raise ValueError("selected Hot checkpoint seed does not match the replication seed")
    for key, value in _ARCHITECTURE.items():
        if int(config.get(key, -1)) != value:
            raise ValueError(f"selected Hot checkpoint has incompatible {key}")
    return payload


def build_selected_checkpoint_contract(
    *,
    seed: int,
    result: Mapping[str, Any],
    checkpoint_dir: str | Path,
    warm_q75_audit: float,
) -> dict[str, Any]:
    """Build a provenance record for the actual Hot-selected checkpoint."""
    if int(seed) not in _REPLICATION_SEEDS:
        raise ValueError("Hot contract seed is not registered for replication")
    if result.get("passed_hot_preflight") is not True or result.get("gate_status") != "completed":
        raise ValueError("a completed passed Hot preflight is required")
    selected = result.get("selected_validation_epoch")
    if not isinstance(selected, Mapping):
        raise ValueError("Hot result has no selected validation epoch")
    try:
        epoch = int(selected["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Hot selected validation epoch is invalid") from exc
    checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt"
    if not checkpoint_path.is_file():
        raise ValueError(f"selected Hot checkpoint is missing: {checkpoint_path}")
    payload = _load_checkpoint(checkpoint_path, expected_seed=seed)
    if int(payload.get("epoch", -1)) != epoch:
        raise ValueError("selected Hot checkpoint payload epoch does not match result")
    if not np.isfinite(float(warm_q75_audit)) or float(warm_q75_audit) < 0.0:
        raise ValueError("warm q75 audit statistic must be finite and non-negative")
    return {
        "schema_version": 1,
        "seed": int(seed),
        "epoch": epoch,
        "relative_path": checkpoint_path.name,
        "sha256": _sha256(checkpoint_path),
        "architecture": dict(_ARCHITECTURE),
        "fixed_trust_tau": float(FIXED_TRUST_TAU),
        "warm_q75_audit": float(warm_q75_audit),
    }


def compute_warm_q75_audit(
    *,
    seed: int,
    data_dir: str | Path,
    split_dir: str | Path,
    checkpoint_path: str | Path,
    device: str = "",
) -> float:
    """Compute q75 in normalized Hot/content space from train-warm IDs only."""
    if int(seed) not in _REPLICATION_SEEDS:
        raise ValueError("q75 audit seed is not registered for replication")
    meta, content, train_df, _ = load_hot_replication_inputs(data_dir, split_dir)
    selected = _load_checkpoint(Path(checkpoint_path), expected_seed=seed)
    requested = str(device).strip().lower()
    if requested:
        if requested == "cpu":
            resolved = torch.device("cpu")
        elif requested.startswith("cuda") and torch.cuda.is_available():
            resolved = torch.device(requested)
        else:
            raise RuntimeError(f"requested unavailable device: {device}")
    else:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cgrc.CGRCNet(
        int(meta["n_users"]), int(meta["n_items"]), int(content.shape[1]), 64, 64, content
    ).to(resolved)
    model.load_state_dict(selected["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        R = cgrc._build_interaction_csr(train_df, model.n_users, model.n_items)
        sparse = cgrc._sparse_adj_tensor(
            cgrc._normalize_graph_mat(cgrc._bip_adj_from_R(R, model.n_users, model.n_items)), resolved
        )
        _, items = cgrc._lightgcn_mean_all_layers(sparse, model.user_emb, model.item_x(), model.n_users, 2)
        items = F.normalize(items, dim=1)
        content_bank = F.normalize(model.item_x(), dim=1)
        counts = train_df["i_idx"].astype(int).value_counts()
        warm = np.asarray([int(counts.get(index, 0)) > 0 for index in range(model.n_items)], dtype=bool)
        if int(warm.sum()) != 596 or int((~warm).sum()) != 102:
            raise ValueError("replication split does not have the registered 596/102 catalog composition")
        return float(torch.quantile((items[warm] - content_bank[warm]).norm(dim=1), 0.75).item())


def register_hot_contract(
    *,
    seed: int,
    result_path: str | Path,
    checkpoint_dir: str | Path,
    data_dir: str | Path,
    split_dir: str | Path,
    device: str = "",
) -> dict[str, Any]:
    """Append the dynamic checkpoint contract to a completed Hot result."""
    path = Path(result_path)
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError("Hot result must be a JSON object")
    config = result.get("config")
    if not isinstance(config, Mapping) or int(config.get("seed", -1)) != int(seed):
        raise ValueError("Hot result seed does not match contract seed")
    selected = result.get("selected_validation_epoch")
    if not isinstance(selected, Mapping):
        raise ValueError("Hot result has no selected checkpoint")
    checkpoint_path = Path(checkpoint_dir) / f"epoch_{int(selected['epoch']):03d}.pt"
    q75 = compute_warm_q75_audit(
        seed=seed,
        data_dir=data_dir,
        split_dir=split_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    result["selected_checkpoint_contract"] = build_selected_checkpoint_contract(
        seed=seed,
        result=result,
        checkpoint_dir=checkpoint_dir,
        warm_q75_audit=q75,
    )
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result["selected_checkpoint_contract"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--device", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    contract = register_hot_contract(
        seed=args.seed,
        result_path=args.result_path,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        device=args.device,
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
