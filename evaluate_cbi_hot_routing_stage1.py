"""Stage-1 candidate-level routing evaluator (CBI cold / Hot-expert hot).

New, self-contained script. Does NOT modify any protected source file. It faithfully
reuses the frozen forward logic of both experts:

  * CBI expert  : builds the exact all-refined item bank
                  (evaluate_cbi_all_refined_seed2025.build_all_refined_item_bank) and
                  scores with cosine  z_u = F.normalize(user_proj(user_emb(u))) @ bank.T
                  (fast3_delta/eval.evaluate_usim, lines 466-468).
  * Hot expert  : LightGCN mean-of-all-layers user/item embeddings, L2-normalized
                  (ckg_hot_graph_test_replay._evaluate_frozen_checkpoint, lines 226-235).

For every test interaction we build a routed 698-dim score vector: cold candidates
(train popularity < cold_threshold) take the calibrated CBI score, hot candidates take
the calibrated Hot-expert score. Ranking metrics use the SAME
hin_eval_common.compute_ranking_metric_values and the SAME item_macro aggregation as
the main tables, so the produced Overall is a like-for-like, deployable number to
compare against the frozen oracle stitch upper bound (Overall R@10=0.2492 / N@10=0.1650).

Two per-user, parameter-free, deterministic calibrations are produced:
  * percentile : within-bucket rank in [0, 1]
  * zscore     : within-bucket standardization
Seen-masked candidates (-1e9) are excluded from the calibration distribution and get a
sentinel -1e9 in the merged vector (they were already excluded from ranking).

Run with the official env:
  D:\\anaconda3\\envs\\zw\\python.exe evaluate_cbi_hot_routing_stage1.py [--seeds 2025 ...] [--validate-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent

# ---- faithful reuse of frozen forward logic (import, do not modify) ----
import cgrc_paper_static_hin as cgrc  # noqa: E402
from ckg_hot_graph_test_replay import load_test_replay_inputs  # noqa: E402
from hin_eval_common import compute_ranking_metric_values  # noqa: E402

import fast3_delta.eval as eval_mod  # noqa: E402
from fast3_delta.config import Fast3Config  # noqa: E402
from fast3_delta.course_artifacts import _empty_course_stats, build_course_artifacts  # noqa: E402
from fast3_delta.eval import prepare_llm_scores  # noqa: E402
from fast3_delta.static_protocol import add_user_seen_from_df, apply_train_popularity  # noqa: E402
from usim_feedback_fast3_content_delta import (  # noqa: E402
    Fast3FeedbackUSIM,
    _resolve_torch_device,
    load_llm_scores_for_stream,
)
from evaluate_cbi_all_refined_seed2025 import (  # noqa: E402
    build_all_refined_item_bank,
    _reset_usim_env,
    _set_seed,
)

K_LIST = (5, 10, 20)
NEG_INF = -1e9
SENTINEL = -1e9

# Frozen reference numbers (item-macro) used purely as a faithfulness gate.
_REFERENCE = {
    2025: {
        "cbi_cold_r10": 0.3613185488001842,
        "cbi_hot_r10": 0.1222871398145451,
        "hot_cold_r10": 0.22316119877283838,
        "hot_hot_r10": 0.23120992790080772,
    }
}

# ---- checkpoint / manifest registry -----------------------------------------

def _cbi_paths(seed: int) -> tuple[Path, Path]:
    if seed == 2025:
        root = _REPO_ROOT / "outputs" / "cbi_anchor_sim_single_seed2025" / (
            "strict_item_cold_balanced_thr1_seed_2025"
        )
        ckpt = _REPO_ROOT / "checkpoints" / "cbi_anchor_sim_single_seed2025" / (
            "strict_item_cold_balanced_thr1_seed_2025"
        ) / "finished.pt"
    else:
        root = _REPO_ROOT / "outputs" / "cbi_anchor_sim_3seed_serial" / (
            f"strict_item_cold_balanced_thr1_seed_{seed}"
        )
        ckpt = _REPO_ROOT / "checkpoints" / "cbi_anchor_sim_3seed_serial" / (
            f"strict_item_cold_balanced_thr1_seed_{seed}"
        ) / "finished.pt"
    manifest = root / "static_protocol_manifest.json"
    return manifest, ckpt


def _hot_paths(seed: int) -> tuple[Path, Path]:
    split_dir = (
        _REPO_ROOT
        / "outputs"
        / "content_delta_pop5"
        / "static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{seed}"
    )
    if seed == 2025:
        ckpt = _REPO_ROOT / "checkpoints" / "ckg_hot_graph_preflight_seed2025" / "epoch_015.pt"
    elif seed == 2026:
        ckpt = (
            _REPO_ROOT / "checkpoints" / "ckg_hot_graph_preflight_replication_seed2026" / "epoch_015.pt"
        )
    elif seed == 2027:
        ckpt = (
            _REPO_ROOT / "checkpoints" / "ckg_hot_graph_preflight_replication_seed2027" / "epoch_013.pt"
        )
    else:
        raise ValueError(f"no registered Hot checkpoint for seed {seed}")
    return split_dir, ckpt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- CBI expert reconstruction (faithful to run_cbi_anchor_sim + blueprint) --

def build_cbi_expert(seed: int, device: torch.device):
    manifest_path, ckpt_path = _cbi_paths(seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing CBI manifest: {manifest_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing CBI checkpoint: {ckpt_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    _reset_usim_env(manifest.get("env", {}))
    _set_seed(int(manifest["split"]["seed"]))

    data_dir = Path(manifest["data"]["data_dir"])
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    frame = pd.read_pickle(data_dir / "stream_data.pkl")
    content = torch.load(data_dir / "content_emb.pt", map_location="cpu", weights_only=False)

    llm_scores, llm_path, _ = load_llm_scores_for_stream(
        str(data_dir),
        frame,
        cold_threshold=int(manifest["split"]["cold_threshold"]),
        n_users=meta["n_users"],
        n_items=meta["n_items"],
        fallback_data_dirs=["processed_data"],
        verbose=False,
    )
    cfg = Fast3Config(meta["n_users"], meta["n_items"], content.shape[1])
    llm_scores, llm_summary = prepare_llm_scores(llm_scores, cfg)
    cfg.llm_bank_mode = llm_summary["mode"]

    exports = manifest["exports"]
    train_df = pd.read_pickle(exports["train_split"])
    val_df = pd.read_pickle(exports["val_split"])
    test_df = pd.read_pickle(exports["test_split"])
    train_df, val_df, test_df, train_pop = apply_train_popularity(train_df, val_df, test_df, cfg)

    artifact_df = frame if manifest["split"].get("artifact_source") == "all_metadata" else train_df
    if cfg.feedback_load_course_artifacts:
        course_artifacts, _ = build_course_artifacts(
            artifact_df,
            cfg.n_items,
            relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations"),
            prereq_min_support=cfg.prereq_min_support,
            prereq_max_per_item=cfg.prereq_max_per_item,
            prereq_min_items=cfg.prereq_min_items,
            prereq_max_forward=cfg.prereq_max_forward,
        )
    else:
        course_artifacts = None

    model = Fast3FeedbackUSIM(cfg, content).to(device)
    model.device = device
    if course_artifacts is not None:
        model.set_course_artifacts(course_artifacts)
    model.set_feedback_item_stats(train_pop)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_state = checkpoint.get("es_best_state")
    if best_state is None:
        raise RuntimeError("CBI checkpoint does not contain es_best_state")
    incompatible = model.load_state_dict(best_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"CBI checkpoint mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    assert int(cfg.cold_threshold) == 1, f"CBI cold_threshold must be 1, got {cfg.cold_threshold}"

    train_seen = add_user_seen_from_df({}, train_df)
    model.set_user_seen_index(train_seen)

    with torch.no_grad():
        item_bank, bank_stats = build_all_refined_item_bank(
            model, device, llm_scores=llm_scores, item_batch=1024
        )
    item_bank = item_bank.detach()
    assert item_bank.shape == (meta["n_items"], int(cfg.emb_dim)), item_bank.shape

    return {
        "model": model,
        "item_bank": item_bank,
        "train_df": train_df,
        "test_df": test_df,
        "train_pop": np.asarray(train_pop, dtype=np.int64),
        "cold_threshold": int(cfg.cold_threshold),
        "n_items": int(meta["n_items"]),
        "n_users": int(meta["n_users"]),
        "ckpt_path": ckpt_path,
        "manifest_path": manifest_path,
        "llm_path": llm_path,
        "llm_mode": llm_summary["mode"],
        "bank_stats": bank_stats,
        "emb_dim": int(cfg.emb_dim),
    }


@torch.no_grad()
def cbi_user_vectors(model, users: torch.Tensor) -> torch.Tensor:
    """z_u = F.normalize(user_proj(user_emb(u))) -- eval.py:466."""
    return F.normalize(model.user_proj(model.user_emb(users)), dim=1)


# ---- Hot expert reconstruction (faithful to test replay) --------------------

def build_hot_expert(seed: int, device: torch.device):
    split_dir, ckpt_path = _hot_paths(seed)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing Hot checkpoint: {ckpt_path}")
    data_dir = _REPO_ROOT / "processed_data_hin_clean_pop5"
    meta, content, train_df, test_df = load_test_replay_inputs(data_dir, split_dir)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_cfg = state["config"]
    if int(ck_cfg.get("seed", -1)) != int(seed):
        raise ValueError("Hot checkpoint seed mismatch")
    assert int(ck_cfg.get("cold_threshold", 1)) == 1

    model = cgrc.CGRCNet(
        int(meta["n_users"]),
        int(meta["n_items"]),
        int(content.shape[1]),
        int(ck_cfg["emb_dim"]),
        int(ck_cfg["mlp_hidden"]),
        content,
    ).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()

    layers_full = int(ck_cfg["layers_full"])
    r_base = cgrc._build_interaction_csr(train_df, model.n_users, model.n_items)
    sparse_full = cgrc._sparse_adj_tensor(
        cgrc._normalize_graph_mat(cgrc._bip_adj_from_R(r_base, model.n_users, model.n_items)),
        device,
    )
    with torch.no_grad():
        all_u, all_i = cgrc._lightgcn_mean_all_layers(
            sparse_full, model.user_emb, model.item_x(), model.n_users, layers_full
        )
        all_u = F.normalize(all_u, dim=1).detach()
        all_i = F.normalize(all_i, dim=1).detach()

    return {
        "all_u": all_u,
        "all_i": all_i,
        "ckpt_path": ckpt_path,
        "split_dir": split_dir,
        "test_df": test_df,
        "train_df": train_df,
        "emb_dim": int(ck_cfg["emb_dim"]),
    }


# ---- shared structures ------------------------------------------------------

def build_seen_bool(train_df: pd.DataFrame, unique_users: np.ndarray, n_items: int) -> torch.Tensor:
    """[n_unique, n_items] bool matrix of train-seen items for the given users."""
    seen_map: dict[int, list[int]] = {}
    us = train_df["u_idx"].astype(int).to_numpy()
    is_ = train_df["i_idx"].astype(int).to_numpy()
    for u, i in zip(us, is_):
        seen_map.setdefault(u, []).append(i)
    seen_bool = torch.zeros((len(unique_users), n_items), dtype=torch.bool)
    for row, u in enumerate(unique_users):
        items = seen_map.get(int(u))
        if items:
            seen_bool[row, items] = True
    return seen_bool


# ---- calibration ------------------------------------------------------------

def _percentile_within(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Rank in [0,1] among valid entries per row. Masked -> SENTINEL.

    Masked scores are set to a value strictly below every valid score so that a
    valid item's global ascending rank equals (#masked_in_row + #valid_below_it);
    subtracting #masked recovers the within-valid rank.
    """
    n = scores.size(1)
    filled = torch.where(valid, scores, torch.full_like(scores, NEG_INF))
    order = filled.argsort(dim=1)
    rank_all = order.argsort(dim=1).float()  # 0-based ascending rank over all cols
    n_masked = (~valid).sum(dim=1, keepdim=True).float()
    n_valid = valid.sum(dim=1, keepdim=True).float()
    denom = (n_valid - 1.0).clamp_min(1.0)
    pct = (rank_all - n_masked) / denom
    pct = torch.where(valid, pct, torch.full_like(pct, SENTINEL))
    # rows with a single valid item -> put it at 0.5 (neutral) rather than 0.
    single = (n_valid <= 1).expand_as(pct)
    pct = torch.where(single & valid, torch.full_like(pct, 0.5), pct)
    return pct


def _zscore_within(scores: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    vfloat = valid.float()
    n_valid = vfloat.sum(dim=1, keepdim=True).clamp_min(1.0)
    masked_scores = torch.where(valid, scores, torch.zeros_like(scores))
    mean = masked_scores.sum(dim=1, keepdim=True) / n_valid
    centered = torch.where(valid, scores - mean, torch.zeros_like(scores))
    var = (centered.pow(2)).sum(dim=1, keepdim=True) / n_valid
    std = (var + eps).sqrt()
    z = (scores - mean) / std
    z = torch.where(valid, z, torch.full_like(z, SENTINEL))
    return z


# ---- item-macro accumulator -------------------------------------------------

class MacroAccumulator:
    def __init__(self):
        # bucket -> metric_key -> {item_id: sum}, plus bucket -> {item_id: count}
        self.sum = {"cold": {}, "hot": {}}
        self.count = {"cold": {}, "hot": {}}
        for b in ("cold", "hot"):
            for m in ("R", "N"):
                for k in K_LIST:
                    self.sum[b][f"{m}@{k}"] = {}

    def add_batch(self, bucket_is_cold: torch.Tensor, target_items: torch.Tensor, metric_vals: dict):
        bic = bucket_is_cold.cpu().numpy()
        items = target_items.cpu().numpy()
        vals = {k: v.cpu().numpy() for k, v in metric_vals.items()}
        for row in range(len(items)):
            bucket = "cold" if bic[row] else "hot"
            item_id = int(items[row])
            self.count[bucket][item_id] = self.count[bucket].get(item_id, 0) + 1
            for key, arr in vals.items():
                d = self.sum[bucket][key]
                d[item_id] = d.get(item_id, 0.0) + float(arr[row])

    def finalize(self):
        result = {}
        n_items = {}
        for bucket in ("cold", "hot"):
            counts = self.count[bucket]
            n_items[bucket] = len(counts)
            macro = {}
            for key, sums in self.sum[bucket].items():
                if counts:
                    per_item = [sums.get(i, 0.0) / counts[i] for i in counts]
                    macro[key] = sum(per_item) / len(per_item)
                else:
                    macro[key] = 0.0
            result[bucket] = macro
        # overall = weighted by distinct target-item count (matches main table)
        n_cold, n_hot = n_items["cold"], n_items["hot"]
        total = n_cold + n_hot
        overall = {}
        for key in result["cold"]:
            overall[key] = (
                result["cold"][key] * n_cold + result["hot"][key] * n_hot
            ) / max(1, total)
        result["overall"] = overall
        return result, n_cold, n_hot


# ---- main routing evaluation for one seed -----------------------------------

def evaluate_seed(seed: int, device: torch.device, batch_size: int = 2048, validate: bool = False):
    t0 = time.perf_counter()
    print(f"\n===== seed {seed} =====", flush=True)
    hot = build_hot_expert(seed, device)  # build Hot first (no USIM env dependence)
    cbi = build_cbi_expert(seed, device)  # sets USIM_* env

    n_items = cbi["n_items"]
    assert hot["all_i"].shape[0] == n_items

    # Verify byte-identical test/train and reuse a single source of truth.
    cbi_test = cbi["test_df"].reset_index(drop=True)
    hot_test = hot["test_df"].reset_index(drop=True)
    cbi_pairs = set(zip(cbi_test["u_idx"].astype(int), cbi_test["i_idx"].astype(int)))
    hot_pairs = set(zip(hot_test["u_idx"].astype(int), hot_test["i_idx"].astype(int)))
    assert cbi_pairs == hot_pairs, "CBI/Hot test interactions differ"

    train_pop = cbi["train_pop"]
    cold_threshold = cbi["cold_threshold"]
    cold_item_mask = torch.from_numpy(train_pop < cold_threshold).to(device)  # [n_items] bool
    cold_idx = torch.nonzero(cold_item_mask, as_tuple=False).view(-1)
    hot_idx = torch.nonzero(~cold_item_mask, as_tuple=False).view(-1)
    print(f"  cold candidates={cold_idx.numel()} hot candidates={hot_idx.numel()}", flush=True)

    # test users/items
    test_u = torch.from_numpy(cbi_test["u_idx"].astype(np.int64).to_numpy())
    test_i = torch.from_numpy(cbi_test["i_idx"].astype(np.int64).to_numpy())
    n_rows = test_u.numel()

    unique_users = np.unique(test_u.numpy())
    user_to_row = {int(u): r for r, u in enumerate(unique_users)}
    seen_bool_unique = build_seen_bool(cbi["train_df"], unique_users, n_items).to(device)
    inv = torch.from_numpy(
        np.asarray([user_to_row[int(u)] for u in test_u.numpy()], dtype=np.int64)
    )

    cbi_bank = cbi["item_bank"].to(device)          # [n_items, 128]
    hot_bank = hot["all_i"].to(device)              # [n_items, 64]
    all_u_hot = hot["all_u"].to(device)             # [n_users, 64]
    cbi_model = cbi["model"]

    accum = {"percentile": MacroAccumulator(), "zscore": MacroAccumulator()}
    # faithfulness self-checks: pure CBI and pure Hot full-ranking, same aggregation
    check = {"cbi_only": MacroAccumulator(), "hot_only": MacroAccumulator()}

    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        u = test_u[start:end].to(device)
        i = test_i[start:end].to(device)
        rows_seen = seen_bool_unique[inv[start:end].to(device)]  # [B, n_items] bool

        z_cbi = cbi_user_vectors(cbi_model, u)          # [B,128]
        s_cbi = torch.mm(z_cbi, cbi_bank.t())            # [B, n_items]
        z_hot = all_u_hot[u]                             # [B,64]
        s_hot = torch.mm(z_hot, hot_bank.t())            # [B, n_items]

        # seen-mask (keep target). target is a held-out edge, never train-seen, but
        # mirror the reference code which restores it explicitly.
        rowsB = torch.arange(u.size(0), device=device)
        tgt_cbi = s_cbi[rowsB, i].clone()
        tgt_hot = s_hot[rowsB, i].clone()
        s_cbi = s_cbi.masked_fill(rows_seen, NEG_INF)
        s_hot = s_hot.masked_fill(rows_seen, NEG_INF)
        s_cbi[rowsB, i] = tgt_cbi
        s_hot[rowsB, i] = tgt_hot

        target_is_cold = cold_item_mask[i]

        # self-checks: pure expert full-ranking metrics (standalone reproduction)
        check["cbi_only"].add_batch(
            target_is_cold, i, compute_ranking_metric_values(s_cbi, i, k_list=K_LIST)
        )
        check["hot_only"].add_batch(
            target_is_cold, i, compute_ranking_metric_values(s_hot, i, k_list=K_LIST)
        )

        # split into cold/hot candidate subsets
        cbi_cold = s_cbi.index_select(1, cold_idx)      # [B, Nc]
        hot_hot = s_hot.index_select(1, hot_idx)        # [B, Nh]
        valid_cold = cbi_cold > (NEG_INF / 2)
        valid_hot = hot_hot > (NEG_INF / 2)

        for method, cal in (("percentile", _percentile_within), ("zscore", _zscore_within)):
            cold_cal = cal(cbi_cold, valid_cold)
            hot_cal = cal(hot_hot, valid_hot)
            s_route = torch.full((u.size(0), n_items), SENTINEL, device=device)
            s_route.index_copy_(1, cold_idx, cold_cal)
            s_route.index_copy_(1, hot_idx, hot_cal)
            metric_vals = compute_ranking_metric_values(s_route, i, k_list=K_LIST)
            accum[method].add_batch(target_is_cold, i, metric_vals)

    results = {}
    for method in ("percentile", "zscore"):
        res, n_cold, n_hot = accum[method].finalize()
        results[method] = {"metrics": res, "n_cold": n_cold, "n_hot": n_hot}

    # self-checks
    checks_out = {}
    for name in ("cbi_only", "hot_only"):
        res, n_cold, n_hot = check[name].finalize()
        checks_out[name] = res

    # print validation summary
    print(
        f"  [self-check] CBI-only  cold R@10={checks_out['cbi_only']['cold']['R@10']:.4f} "
        f"hot R@10={checks_out['cbi_only']['hot']['R@10']:.4f}",
        flush=True,
    )
    print(
        f"  [self-check] Hot-only  cold R@10={checks_out['hot_only']['cold']['R@10']:.4f} "
        f"hot R@10={checks_out['hot_only']['hot']['R@10']:.4f}",
        flush=True,
    )
    if seed in _REFERENCE:
        ref = _REFERENCE[seed]
        gate = {
            "cbi_cold_r10": checks_out["cbi_only"]["cold"]["R@10"],
            "cbi_hot_r10": checks_out["cbi_only"]["hot"]["R@10"],
            "hot_cold_r10": checks_out["hot_only"]["cold"]["R@10"],
            "hot_hot_r10": checks_out["hot_only"]["hot"]["R@10"],
        }
        for key, got in gate.items():
            exp = ref[key]
            drift = abs(got - exp)
            flag = "OK" if drift < 2e-3 else "WARN"
            print(f"    gate {key}: got={got:.4f} ref={exp:.4f} drift={drift:.4f} [{flag}]", flush=True)
    for method in ("percentile", "zscore"):
        m = results[method]["metrics"]
        print(
            f"  [{method}] cold R@10={m['cold']['R@10']:.4f} N@10={m['cold']['N@10']:.4f} | "
            f"hot R@10={m['hot']['R@10']:.4f} N@10={m['hot']['N@10']:.4f} | "
            f"overall R@10={m['overall']['R@10']:.4f} N@10={m['overall']['N@10']:.4f}",
            flush=True,
        )
    print(f"  seed {seed} done in {time.perf_counter() - t0:.1f}s", flush=True)

    return {
        "seed": seed,
        "results": results,
        "checks": checks_out,
        "cbi_ckpt": cbi["ckpt_path"],
        "cbi_manifest": cbi["manifest_path"],
        "hot_ckpt": hot["ckpt_path"],
        "llm_mode": cbi["llm_mode"],
        "cbi_emb_dim": cbi["emb_dim"],
        "hot_emb_dim": hot["emb_dim"],
        "n_cold_candidates": int(cold_idx.numel()),
        "n_hot_candidates": int(hot_idx.numel()),
    }


# ---- output writers ---------------------------------------------------------

_METRIC_ORDER = [f"{m}@{k}" for k in K_LIST for m in ("R", "N")]  # R@5,N@5,R@10,N@10,R@20,N@20


def _per_seed_row(seed_result: dict, method: str) -> dict:
    r = seed_result["results"][method]
    m = r["metrics"]
    row = {"seed": seed_result["seed"], "n_cold": r["n_cold"], "n_hot": r["n_hot"]}
    for bucket in ("cold", "hot", "overall"):
        for metric in _METRIC_ORDER:
            row[f"{bucket}_{metric}"] = m[bucket][metric]
    return row


def write_outputs(seed_results: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv

    fieldnames = ["seed", "n_cold", "n_hot"] + [
        f"{b}_{metric}" for b in ("cold", "hot", "overall") for metric in _METRIC_ORDER
    ]
    for method in ("percentile", "zscore"):
        path = out_dir / f"routing_per_seed_{method}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for sr in seed_results:
                writer.writerow(_per_seed_row(sr, method))

    # summary: method,bucket,metric,mean,std (sample std, n-1)
    summary_path = out_dir / "routing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "bucket", "metric", "mean", "std"])
        for method in ("percentile", "zscore"):
            for bucket in ("cold", "hot", "overall"):
                for metric in _METRIC_ORDER:
                    vals = [
                        sr["results"][method]["metrics"][bucket][metric] for sr in seed_results
                    ]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    writer.writerow([method, bucket, metric, f"{mean:.6f}", f"{std:.6f}"])

    # provenance
    prov = {
        "experiment": "cbi_hot_routing_stage1",
        "description": (
            "Candidate-level routing evaluator: cold candidates (train popularity < "
            "cold_threshold=1) scored by the CBI expert, hot candidates by the Hot graph "
            "expert; per-user parameter-free calibration merges them into one 698-dim "
            "ranking. Metrics use hin_eval_common.compute_ranking_metric_values and the "
            "same item_macro aggregation as the main tables."
        ),
        "interpretation": (
            "This is the deployable, real routing value. It should fall between the raw "
            "CKG-RL baseline (Overall R@10 ~ 0.157) and the frozen oracle stitch upper "
            "bound (Overall R@10 = 0.2492 / N@10 = 0.1650)."
        ),
        "calibration_methods": {
            "percentile": "within-bucket rank in [0,1] per user (CBI over cold candidates, "
            "Hot over hot candidates); seen-masked candidates excluded from the "
            "distribution and set to -1e9.",
            "zscore": "within-bucket z-score standardization per user; seen-masked "
            "candidates excluded from mean/std and set to -1e9.",
        },
        "scoring": "both experts are pure cosine similarity (L2-normalized user and item vectors)",
        "seeds": [sr["seed"] for sr in seed_results],
        "checkpoints": [],
        "self_checks": {},
    }
    for sr in seed_results:
        prov["checkpoints"].append(
            {
                "seed": sr["seed"],
                "cbi_checkpoint": str(sr["cbi_ckpt"]),
                "cbi_checkpoint_sha256": _sha256(sr["cbi_ckpt"]),
                "cbi_manifest": str(sr["cbi_manifest"]),
                "cbi_manifest_sha256": _sha256(sr["cbi_manifest"]),
                "cbi_emb_dim": sr["cbi_emb_dim"],
                "hot_checkpoint": str(sr["hot_ckpt"]),
                "hot_checkpoint_sha256": _sha256(sr["hot_ckpt"]),
                "hot_emb_dim": sr["hot_emb_dim"],
                "llm_mode": sr["llm_mode"],
                "n_cold_candidates": sr["n_cold_candidates"],
                "n_hot_candidates": sr["n_hot_candidates"],
            }
        )
        prov["self_checks"][str(sr["seed"])] = {
            "cbi_only_cold_R@10": sr["checks"]["cbi_only"]["cold"]["R@10"],
            "cbi_only_hot_R@10": sr["checks"]["cbi_only"]["hot"]["R@10"],
            "hot_only_cold_R@10": sr["checks"]["hot_only"]["cold"]["R@10"],
            "hot_only_hot_R@10": sr["checks"]["hot_only"]["hot"]["R@10"],
        }
    (out_dir / "routing_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote outputs to {out_dir}", flush=True)


# ---- entry ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--validate-only", action="store_true", help="run only seed 2025")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--output-dir", type=Path, default=_REPO_ROOT / "outputs" / "cbi_hot_routing_stage1"
    )
    args = parser.parse_args()

    seeds = [2025] if args.validate_only else args.seeds
    device = _resolve_torch_device()
    print(f"device={device} torch={torch.__version__} seeds={seeds}", flush=True)

    seed_results = [evaluate_seed(s, device, batch_size=args.batch_size) for s in seeds]

    if args.validate_only:
        print("\n[validate-only] skipping output write; re-run without --validate-only for full CSVs.")
        return
    write_outputs(seed_results, args.output_dir)


if __name__ == "__main__":
    main()
