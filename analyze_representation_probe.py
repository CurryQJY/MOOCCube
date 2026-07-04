from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.4,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.unicode_minus": False,
        "savefig.dpi": 600,
    }
)


SEEDS = (2025, 2026, 2027)
PALETTE = {
    "content": "#8A9099",
    "cgrc": "#E69F00",
    "ckg": "#0072B2",
    "gain": "#009E73",
    "loss": "#D55E00",
    "tie": "#C9CDD3",
    "neutral": "#3D4451",
    "grid": "#E6E8EB",
    "category": "#56B4E9",
    "concept": "#CC79A7",
}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norm > eps, norm, 1.0)
    out = arr / safe
    out[norm[:, 0] <= eps] = 0.0
    return out


def merge_per_item_pair(
    ours: pd.DataFrame,
    baseline: pd.DataFrame,
    ours_name: str,
    baseline_name: str,
) -> pd.DataFrame:
    merged = ours.merge(
        baseline,
        on=["seed", "item_id"],
        suffixes=("_ours", "_baseline"),
        how="inner",
        validate="one_to_one",
    )
    for metric in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20"):
        if f"{metric}_ours" not in merged.columns or f"{metric}_baseline" not in merged.columns:
            continue
        merged[f"delta_{metric}"] = np.round(
            merged[f"{metric}_ours"].astype(float) - merged[f"{metric}_baseline"].astype(float),
            12,
        )
    merged["ours"] = ours_name
    merged["baseline"] = baseline_name
    return merged


def summarize_signed_delta(values: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    wins = int((arr > eps).sum())
    ties = int((np.abs(arr) <= eps).sum())
    losses = int((arr < -eps).sum())
    n = int(arr.size)
    denom = max(n, 1)
    return {
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_ratio": wins / denom,
        "tie_ratio": ties / denom,
        "loss_ratio": losses / denom,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    seed: int = 2025,
    ci: float = 0.95,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "n": int(arr.size),
    }


def neighbor_label_purity(
    embeddings: np.ndarray,
    labels: list[set[str]],
    query_ids: list[int],
    candidate_ids: list[int],
    k: int = 10,
) -> dict[str, float]:
    detail, empty_label_queries = neighbor_label_purity_values(
        embeddings,
        labels,
        query_ids,
        candidate_ids,
        k=k,
    )
    purities = detail["purity"].astype(float).to_numpy() if not detail.empty else np.asarray([])
    return {
        "mean_purity": float(np.mean(purities)) if purities.size else 0.0,
        "std_purity": float(np.std(purities)) if purities.size else 0.0,
        "n_queries": int(len(query_ids)),
        "n_evaluated": int(purities.size),
        "n_empty_label_queries": int(empty_label_queries),
        "k": int(k),
    }


def neighbor_label_purity_values(
    embeddings: np.ndarray,
    labels: list[set[str]],
    query_ids: list[int],
    candidate_ids: list[int],
    k: int = 10,
) -> tuple[pd.DataFrame, int]:
    emb = l2_normalize(embeddings)
    cand = np.asarray(candidate_ids, dtype=np.int64)
    cand_emb = emb[cand]
    rows = []
    empty_label_queries = 0
    for q in query_ids:
        q_int = int(q)
        q_labels = labels[q_int]
        if not q_labels:
            empty_label_queries += 1
            continue
        scores = cand_emb @ emb[q_int]
        same = cand == q_int
        if same.any():
            scores = scores.copy()
            scores[same] = -np.inf
        take = min(k, int(np.isfinite(scores).sum()))
        if take <= 0:
            continue
        top_pos = np.argpartition(-scores, take - 1)[:take]
        hits = 0
        for pos in top_pos:
            if q_labels & labels[int(cand[pos])]:
                hits += 1
        rows.append(
            {
                "query_id": q_int,
                "purity": hits / take,
                "k": int(k),
                "neighbors_used": int(take),
            }
        )
    return pd.DataFrame(rows), int(empty_label_queries)


def save_pub_py(
    fig: mpl.figure.Figure,
    out_base: Path,
    dpi: int = 600,
    gray_preview: bool = True,
) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    png_path = out_base.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if gray_preview:
        _save_grayscale_preview(png_path, out_base.with_name(f"{out_base.name}_gray").with_suffix(".png"))


def _save_grayscale_preview(src: Path, dst: Path) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return
    with Image.open(src) as image:
        ImageOps.grayscale(image).save(dst)


def style_axis(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.55)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
    )


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu")


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = _torch_load(path)
    state = (
        ckpt.get("model_state")
        or ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt.get("best_state")
    )
    if not isinstance(state, dict):
        raise ValueError(f"No model state found in {path}")
    return state


def content_projection_from_state(state: dict[str, torch.Tensor], content: torch.Tensor) -> torch.Tensor:
    x = content.float()
    x = F.linear(x, state["content_proj.0.weight"], state["content_proj.0.bias"])
    x = F.gelu(x)
    x = F.linear(x, state["content_proj.3.weight"], state["content_proj.3.bias"])
    x = F.gelu(x)
    x = F.linear(x, state["content_proj.6.weight"], state["content_proj.6.bias"])
    x = F.layer_norm(
        x,
        normalized_shape=(x.shape[1],),
        weight=state["content_proj.7.weight"],
        bias=state["content_proj.7.bias"],
    )
    return x


def ckg_cold_item_embeddings(
    state: dict[str, torch.Tensor],
    content: torch.Tensor,
    manifest: dict,
) -> np.ndarray:
    cfg = manifest.get("model_config", {})
    env = manifest.get("env", {})
    content_e = content_projection_from_state(state, content)
    if _bool_cfg(cfg, env, "content_delta_normalize_base", "USIM_CONTENT_DELTA_NORMALIZE_BASE", True):
        base_e = F.normalize(content_e, dim=1)
    else:
        base_e = content_e

    use_delta = _bool_cfg(cfg, env, "use_content_delta", "USIM_USE_CONTENT_DELTA", False)
    if use_delta:
        delta = state.get("content_delta.weight", torch.zeros_like(base_e))
        max_norm = _float_cfg(cfg, env, "content_delta_max_norm", "USIM_CONTENT_DELTA_MAX_NORM", 0.05)
        if max_norm >= 0.0:
            delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
            delta = delta * (max_norm / delta_norm).clamp(max=1.0)
        scale = _float_cfg(cfg, env, "content_delta_scale", "USIM_CONTENT_DELTA_SCALE", 1.0)
        content_e = base_e + scale * delta
        if _bool_cfg(cfg, env, "content_delta_normalize_output", "USIM_CONTENT_DELTA_NORMALIZE_OUTPUT", True):
            content_e = F.normalize(content_e, dim=1)
    else:
        content_e = F.normalize(base_e, dim=1)

    if _bool_cfg(cfg, env, "content_delta_replace_item", "USIM_CONTENT_DELTA_REPLACE_ITEM", False):
        fused = content_e
    else:
        zeros = torch.zeros_like(content_e)
        gate_in = torch.cat([zeros, content_e], dim=1)
        alpha = torch.sigmoid(F.linear(gate_in, state["gate_net.0.weight"], state["gate_net.0.bias"]))
        fused = (1.0 - alpha) * content_e
    return l2_normalize(fused.detach().cpu().numpy())


def cgrc_item_embeddings(state: dict[str, torch.Tensor], content: torch.Tensor) -> np.ndarray:
    emb = F.linear(content.float(), state["item_lin.weight"], state["item_lin.bias"])
    return l2_normalize(emb.detach().cpu().numpy())


def _bool_cfg(cfg: dict, env: dict, cfg_key: str, env_key: str, default: bool) -> bool:
    if cfg_key in cfg:
        return bool(cfg[cfg_key])
    if env_key in env:
        return str(env[env_key]).strip().lower() in {"1", "true", "yes", "on"}
    return default


def _float_cfg(cfg: dict, env: dict, cfg_key: str, env_key: str, default: float) -> float:
    if cfg_key in cfg:
        return float(cfg[cfg_key])
    if env_key in env:
        return float(env[env_key])
    return default


def load_item_map(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    return {str(row.course_id): int(row.i_idx) for row in frame.itertuples(index=False)}


def load_relation_label_sets(
    relation_path: Path,
    item_map: dict[str, int],
    n_items: int,
    prefix: str | None = None,
) -> list[set[str]]:
    labels = [set() for _ in range(n_items)]
    if not relation_path.exists():
        return labels
    with relation_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or "\t" not in line:
                continue
            course_id, label = line.split("\t", 1)
            if prefix and not label.startswith(prefix):
                continue
            item_idx = item_map.get(str(course_id))
            if item_idx is None:
                continue
            labels[item_idx].add(label)
    return labels


def load_per_item_for_seeds(path_template: str, seeds: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = Path(path_template.format(seed=seed))
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cold_item_ids_from_per_item(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    return [int(x) for x in frame["item_id"].tolist()]


def warm_item_ids_from_split(path: Path) -> list[int]:
    frame = pd.read_csv(path, usecols=["i_idx", "split"])
    return sorted(int(x) for x in frame.loc[frame["split"] == "train", "i_idx"].unique())


def primary_labels(label_sets: list[set[str]], query_ids: list[int], top_n: int = 7) -> tuple[list[str], list[str]]:
    counts: dict[str, int] = {}
    for idx in query_ids:
        label = sorted(label_sets[idx])[0] if label_sets[idx] else "unlabeled"
        counts[label] = counts.get(label, 0) + 1
    top = [x for x, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    assigned = []
    for idx in query_ids:
        label = sorted(label_sets[idx])[0] if label_sets[idx] else "unlabeled"
        assigned.append(label if label in top else "other")
    return assigned, top + (["other"] if len(counts) > top_n else [])


def build_neighbor_summary(
    embeddings: dict[str, np.ndarray],
    label_groups: dict[str, list[set[str]]],
    query_ids: list[int],
    candidate_ids: list[int],
    ks: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    rows = []
    for method, emb in embeddings.items():
        for label_name, labels in label_groups.items():
            for k in ks:
                stats = neighbor_label_purity(emb, labels, query_ids, candidate_ids, k=k)
                rows.append({"method": method, "label_group": label_name, **stats})
    return pd.DataFrame(rows)


def build_neighbor_detail(
    embeddings: dict[str, np.ndarray],
    label_groups: dict[str, list[set[str]]],
    query_ids: list[int],
    candidate_ids: list[int],
    ks: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    frames = []
    for method, emb in embeddings.items():
        for label_name, labels in label_groups.items():
            for k in ks:
                detail, empty_count = neighbor_label_purity_values(
                    emb,
                    labels,
                    query_ids,
                    candidate_ids,
                    k=k,
                )
                if detail.empty:
                    continue
                detail = detail.copy()
                detail["method"] = method
                detail["label_group"] = label_name
                detail["n_empty_label_queries"] = int(empty_count)
                frames.append(detail)
    if not frames:
        return pd.DataFrame(
            columns=[
                "query_id",
                "purity",
                "k",
                "neighbors_used",
                "method",
                "label_group",
                "n_empty_label_queries",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def plot_neighbor_summary(
    summary: pd.DataFrame,
    out_base: Path,
    detail: pd.DataFrame | None = None,
) -> None:
    methods = ["Content", "CGRC", "CKG-RL"]
    method_colors = {"Content": PALETTE["content"], "CGRC": PALETTE["cgrc"], "CKG-RL": PALETTE["ckg"]}
    groups = [("category", "Category labels"), ("concept", "Concept labels")]

    if detail is None or detail.empty:
        fig, ax = plt.subplots(figsize=(3.45, 2.15), layout="constrained")
        x = np.arange(len(methods))
        width = 0.36
        colors = {"category": PALETTE["category"], "concept": PALETTE["concept"]}
        for offset, group in [(-width / 2, "category"), (width / 2, "concept")]:
            values = []
            for method in methods:
                sub = summary[
                    (summary["method"] == method)
                    & (summary["label_group"] == group)
                    & (summary["k"] == 10)
                ]
                values.append(float(sub["mean_purity"].iloc[0]) if not sub.empty else 0.0)
            ax.bar(x + offset, values, width=width, color=colors[group], label=f"{group} @10")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Warm-neighbor purity@10")
        ax.set_ylim(0, 1.0)
        style_axis(ax)
        ax.legend(loc="upper left", ncols=2, handlelength=1.2, columnspacing=0.8)
        save_pub_py(fig, out_base)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), sharey=True, layout="constrained")
    rng = np.random.default_rng(2025)
    for ax, (group, title), label in zip(axes, groups, ["a", "b"]):
        panel_data = []
        for method in methods:
            values = detail[
                (detail["method"] == method)
                & (detail["label_group"] == group)
                & (detail["k"] == 10)
            ]["purity"].astype(float).to_numpy()
            panel_data.append(values)

        positions = np.arange(len(methods))
        violin = ax.violinplot(
            panel_data,
            positions=positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, method in zip(violin["bodies"], methods):
            body.set_facecolor(method_colors[method])
            body.set_edgecolor("none")
            body.set_alpha(0.18)

        boxes = ax.boxplot(
            panel_data,
            positions=positions,
            widths=0.28,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#20242A", "linewidth": 0.9},
            boxprops={"facecolor": "white", "edgecolor": "#20242A", "linewidth": 0.7},
            whiskerprops={"color": "#20242A", "linewidth": 0.7},
            capprops={"color": "#20242A", "linewidth": 0.7},
        )
        for patch in boxes["boxes"]:
            patch.set_alpha(0.86)

        for pos, method, values in zip(positions, methods, panel_data):
            if values.size == 0:
                continue
            shown = values
            if values.size > 240:
                shown = rng.choice(values, size=240, replace=False)
            jitter = rng.normal(0.0, 0.035, size=shown.size)
            ax.scatter(
                np.full(shown.size, pos) + jitter,
                shown,
                s=4.5,
                color=method_colors[method],
                alpha=0.16,
                linewidth=0,
            )
            ci = bootstrap_mean_ci(values, seed=2025 + int(pos) + (0 if group == "category" else 10))
            ax.errorbar(
                [pos],
                [ci["mean"]],
                yerr=[[ci["mean"] - ci["ci_low"]], [ci["ci_high"] - ci["mean"]]],
                fmt="o",
                color="#20242A",
                ecolor="#20242A",
                elinewidth=0.9,
                capsize=2.2,
                markersize=4.2,
                markerfacecolor=method_colors[method],
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=5,
            )
            ax.text(
                pos + 0.08,
                min(1.04, ci["mean"] + 0.045),
                f"{ci['mean']:.2f}",
                ha="left",
                va="center",
                fontsize=5.8,
                color="#20242A",
            )

        panel_label(ax, label)
        ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods)
        ax.set_ylim(-0.04, 1.08)
        style_axis(ax)
    axes[0].set_ylabel("Warm-neighbor purity@10")
    save_pub_py(fig, out_base)
    plt.close(fig)


def plot_risk_distribution(merged: pd.DataFrame, out_base: Path) -> dict[str, float]:
    delta = merged["delta_N@10"].astype(float).to_numpy()
    signed = summarize_signed_delta(delta)
    ci_stats = bootstrap_mean_ci(delta)
    median_delta = float(np.median(delta))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 2.35),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.18, 0.92, 1.05]},
    )
    ax = axes[0]
    ordered = np.sort(delta)
    idx = np.arange(len(ordered))
    ax.fill_between(idx, 0, ordered, where=ordered >= 0, color=PALETTE["gain"], linewidth=0)
    ax.fill_between(idx, 0, ordered, where=ordered < 0, color=PALETTE["loss"], linewidth=0)
    ax.axhline(0, color="#2F3542", linewidth=0.8)
    ax.set_title("Ranked paired delta")
    ax.set_xlabel("Seed-course units")
    ax.set_ylabel("CKG-RL - CGRC N@10")
    ax.text(
        0.03,
        0.95,
        f"mean {ci_stats['mean']:.3f}\n95% CI [{ci_stats['ci_low']:.3f}, {ci_stats['ci_high']:.3f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.6},
    )
    panel_label(ax, "a")
    style_axis(ax)

    ax = axes[1]
    ecdf_x = np.sort(delta)
    ecdf_y = np.arange(1, len(ecdf_x) + 1) / max(len(ecdf_x), 1)
    ax.step(ecdf_x, ecdf_y, where="post", color=PALETTE["ckg"], linewidth=1.2)
    ax.axvline(0, color="#2F3542", linewidth=0.8)
    ax.axvline(ci_stats["mean"], color="#1F6F50", linewidth=1.0)
    ax.axhline(0.5, color="#B9BEC7", linewidth=0.6, linestyle="--")
    ax.set_title("Delta ECDF")
    ax.set_xlabel("N@10 delta")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.0)
    ax.text(
        0.04,
        0.08,
        f"win/tie/loss\n{signed['wins']}/{signed['ties']}/{signed['losses']}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color="#20242A",
    )
    panel_label(ax, "b")
    style_axis(ax, grid_axis="both")

    ax = axes[2]
    base = merged["N@10_baseline"].astype(float).to_numpy()
    ours = merged["N@10_ours"].astype(float).to_numpy()
    lim = max(float(base.max()), float(ours.max()), 1e-6)
    hb = ax.hexbin(
        base,
        ours,
        gridsize=34,
        extent=(0, lim, 0, lim),
        mincnt=1,
        cmap="Blues",
        bins="log",
        linewidths=0,
    )
    ax.plot([0, lim], [0, lim], color="#9AA3AF", linewidth=0.8)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Seed-course units/bin")
    cbar.ax.tick_params(labelsize=5.8, width=0.6, length=2)
    ax.set_title("Paired score density")
    ax.set_xlabel("CGRC N@10")
    ax.set_ylabel("CKG-RL N@10")
    ax.set_xlim(-0.01, lim + 0.02)
    ax.set_ylim(-0.01, lim + 0.02)
    panel_label(ax, "c")
    style_axis(ax, grid_axis="both")
    save_pub_py(fig, out_base)
    plt.close(fig)
    return {
        "paired_units": int(len(merged)),
        "gain_ratio": float(signed["win_ratio"]),
        "tie_ratio": float(signed["tie_ratio"]),
        "loss_ratio": float(signed["loss_ratio"]),
        "wins": int(signed["wins"]),
        "ties": int(signed["ties"]),
        "losses": int(signed["losses"]),
        "mean_delta_n10": float(ci_stats["mean"]),
        "mean_delta_n10_ci_low": float(ci_stats["ci_low"]),
        "mean_delta_n10_ci_high": float(ci_stats["ci_high"]),
        "median_delta_n10": median_delta,
    }


def plot_topconf_composite(merged: pd.DataFrame, neighbor_summary: pd.DataFrame, out_base: Path) -> None:
    delta = merged["delta_N@10"].astype(float).to_numpy()
    signed = summarize_signed_delta(delta)
    ci_stats = bootstrap_mean_ci(delta)
    ordered = np.sort(delta)

    fig = plt.figure(figsize=(7.05, 3.25))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.78, 0.92], height_ratios=[1.0, 1.0])
    ax_delta = fig.add_subplot(gs[:, 0])
    ax_wtl = fig.add_subplot(gs[0, 1])
    ax_ci = fig.add_subplot(gs[1, 1])
    ax_purity = fig.add_subplot(gs[:, 2])

    bar_colors = np.where(ordered > 1e-12, PALETTE["gain"], np.where(ordered < -1e-12, PALETTE["loss"], PALETTE["tie"]))
    ax_delta.bar(np.arange(len(ordered)), ordered, width=1.0, color=bar_colors, linewidth=0)
    ax_delta.axhline(0, color="#20242A", linewidth=0.8)
    ax_delta.set_title("a  Paired cold-course delta")
    ax_delta.set_xlabel("Seed-course units")
    ax_delta.set_ylabel("CKG-RL - CGRC N@10")
    ax_delta.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax_delta.text(
        0.03,
        0.96,
        f"mean {ci_stats['mean']:.3f}\n95% CI [{ci_stats['ci_low']:.3f}, {ci_stats['ci_high']:.3f}]",
        transform=ax_delta.transAxes,
        va="top",
        ha="left",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )

    wtl_labels = ["loss", "tie", "win"]
    wtl_values = [signed["loss_ratio"], signed["tie_ratio"], signed["win_ratio"]]
    wtl_colors = [PALETTE["loss"], PALETTE["tie"], PALETTE["gain"]]
    ax_wtl.barh(wtl_labels, wtl_values, color=wtl_colors, height=0.62)
    for y, value in enumerate(wtl_values):
        ax_wtl.text(value + 0.015, y, f"{value:.2f}", va="center", ha="left", fontsize=6.2)
    ax_wtl.set_xlim(0, 0.55)
    ax_wtl.set_xlabel("Fraction")
    ax_wtl.set_title(f"b  Win/tie/loss (n={signed['n']:,})")
    ax_wtl.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)

    ax_ci.errorbar(
        [ci_stats["mean"]],
        [0],
        xerr=[[ci_stats["mean"] - ci_stats["ci_low"]], [ci_stats["ci_high"] - ci_stats["mean"]]],
        fmt="o",
        color=PALETTE["ckg"],
        ecolor=PALETTE["ckg"],
        elinewidth=1.2,
        capsize=3,
        markersize=4,
    )
    ax_ci.axvline(0, color="#20242A", linewidth=0.8)
    ax_ci.set_xlim(-0.001, max(0.022, ci_stats["ci_high"] * 1.1))
    ax_ci.set_ylim(-0.5, 0.5)
    ax_ci.set_yticks([])
    ax_ci.set_xlabel("Mean N@10 delta")
    ax_ci.set_title("c  Bootstrap CI")
    ax_ci.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
    ax_ci.spines["left"].set_visible(False)

    methods = ["Content", "CGRC", "CKG-RL"]
    groups = ["category", "concept"]
    x = np.arange(len(methods))
    width = 0.36
    colors = {"category": "#8DA9CC", "concept": "#E6AA6D"}
    for offset, group in [(-width / 2, "category"), (width / 2, "concept")]:
        values = []
        for method in methods:
            sub = neighbor_summary[
                (neighbor_summary["method"] == method)
                & (neighbor_summary["label_group"] == group)
                & (neighbor_summary["k"] == 10)
            ]
            values.append(float(sub["mean_purity"].iloc[0]) if not sub.empty else 0.0)
        ax_purity.bar(x + offset, values, width=width, color=colors[group], label=group)
        for xpos, value in zip(x + offset, values):
            ax_purity.text(xpos, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=5.8)
    ax_purity.set_xticks(x)
    ax_purity.set_xticklabels(methods)
    ax_purity.set_ylim(0, 0.9)
    ax_purity.set_ylabel("Warm-neighbor purity@10")
    ax_purity.set_title("d  Representation neighborhood")
    ax_purity.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax_purity.legend(loc="upper left", ncols=1, handlelength=1.0)

    for ax in (ax_delta, ax_wtl, ax_ci, ax_purity):
        ax.tick_params(labelsize=6.3)
    fig.tight_layout(w_pad=1.1, h_pad=1.0)
    save_pub_py(fig, out_base)
    plt.close(fig)


def plot_tsne_panels(
    embeddings: dict[str, np.ndarray],
    query_ids: list[int],
    category_labels: list[set[str]],
    out_base: Path,
    random_state: int = 2025,
) -> None:
    assigned, legend_order = primary_labels(category_labels, query_ids)
    color_values = _label_colors(legend_order)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    for ax, (method, emb), panel in zip(axes, embeddings.items(), ["a", "b", "c"]):
        x = emb[np.asarray(query_ids, dtype=np.int64)]
        perplexity = max(5, min(30, (len(query_ids) - 1) // 3))
        coords = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
        ).fit_transform(x)
        for legend_label in legend_order:
            mask = np.array([v == legend_label for v in assigned])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=7.5,
                color=color_values[legend_label],
                alpha=0.72,
                linewidth=0,
                label=_short_label(legend_label),
            )
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")
        panel_label(ax, panel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=4,
        bbox_to_anchor=(0.5, 0.02),
        handletextpad=0.4,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.02, right=0.995, top=0.88, bottom=0.25, wspace=0.05)
    save_pub_py(fig, out_base)
    plt.close(fig)


def _label_colors(labels: list[str]) -> dict[str, str]:
    base = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#B79F00", "#8A9099"]
    return {label: base[i % len(base)] for i, label in enumerate(labels)}


def _short_label(label: str) -> str:
    return label.replace("COCO_CATEGORY:", "").replace("COCO_CONCEPT:", "")[:24]


def run_coco_probe(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("processed_data_coco")
    run_template = "outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}"
    ours_template = run_template + "/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
    cgrc_template = run_template + "/main_table_compare/per_item_full_cold_cgrc_paper_static.csv"

    ours = load_per_item_for_seeds(ours_template, SEEDS)
    cgrc = load_per_item_for_seeds(cgrc_template, SEEDS)
    paired = merge_per_item_pair(ours, cgrc, ours_name="CKG-RL", baseline_name="CGRC")
    paired_path = out_dir / "coco_paired_cold_item_delta.csv"
    paired.to_csv(paired_path, index=False)
    risk_stats = plot_risk_distribution(paired, out_dir / "coco_cold_exposure_risk")

    seed = 2025
    seed_dir = Path(run_template.format(seed=seed))
    manifest = json.loads((seed_dir / "static_protocol_manifest.json").read_text(encoding="utf-8"))
    content = _torch_load(data_dir / "content_emb.pt")
    item_map = load_item_map(data_dir / "_item_id_map.csv")
    category_sets = load_relation_label_sets(
        data_dir / "relations" / "course-concept.json",
        item_map,
        n_items=int(content.shape[0]),
        prefix="COCO_CATEGORY:",
    )
    concept_sets = load_relation_label_sets(
        data_dir / "relations" / "course-concept.json",
        item_map,
        n_items=int(content.shape[0]),
        prefix="COCO_CONCEPT:",
    )
    query_ids = cold_item_ids_from_per_item(seed_dir / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv")
    candidate_ids = warm_item_ids_from_split(seed_dir / "static_split_assignments.csv")

    ckg_state = load_model_state(Path(f"checkpoints/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}/finished.pt"))
    cgrc_state = load_model_state(Path(f"checkpoints/coco/single_seed_triage/cgrc_paper/strict_item_cold_balanced_thr1_seed_{seed}/best.pt"))
    embeddings = {
        "Content": l2_normalize(content.detach().cpu().numpy()),
        "CGRC": cgrc_item_embeddings(cgrc_state, content),
        "CKG-RL": ckg_cold_item_embeddings(ckg_state, content, manifest),
    }
    label_groups = {"category": category_sets, "concept": concept_sets}
    neighbor_summary = build_neighbor_summary(embeddings, label_groups, query_ids, candidate_ids)
    neighbor_detail = build_neighbor_detail(embeddings, label_groups, query_ids, candidate_ids)
    neighbor_path = out_dir / "coco_neighbor_purity.csv"
    neighbor_detail_path = out_dir / "coco_neighbor_purity_detail.csv"
    neighbor_summary.to_csv(neighbor_path, index=False)
    neighbor_detail.to_csv(neighbor_detail_path, index=False)
    plot_neighbor_summary(neighbor_summary, out_dir / "coco_neighbor_purity", detail=neighbor_detail)
    plot_topconf_composite(paired, neighbor_summary, out_dir / "coco_topconf_probe")
    plot_tsne_panels(embeddings, query_ids, category_sets, out_dir / "coco_cold_course_tsne")

    summary = {
        "figure_contract": {
            "core_conclusion": "CKG-RL should reduce cold-course exposure failures and preserve educationally coherent cold-to-warm neighborhoods under strict item-cold evaluation.",
            "evidence_chain": [
                "Paired per-cold-item N@10 deltas compare CKG-RL against CGRC over three seeds.",
                "Warm-neighbor category/concept purity checks whether cold-course embeddings retrieve educationally related warm bridge courses.",
                "Cold-course t-SNE panels provide a qualitative view of category organization for content, CGRC, and CKG-RL representations.",
            ],
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "export_formats": ["svg", "pdf", "tiff", "png", "csv", "json"],
        },
        "dataset": "COCO",
        "seed_for_representation": seed,
        "cold_query_items": len(query_ids),
        "warm_candidate_items": len(candidate_ids),
        "risk_stats": risk_stats,
        "neighbor_summary_csv": str(neighbor_path),
        "neighbor_detail_csv": str(neighbor_detail_path),
        "paired_delta_csv": str(paired_path),
    }
    summary_path = out_dir / "coco_representation_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = out_dir / "coco_representation_probe_report.md"
    report_path.write_text(_format_coco_report(summary, neighbor_summary), encoding="utf-8")
    return summary


def _format_coco_report(summary: dict, neighbor_summary: pd.DataFrame) -> str:
    risk = summary["risk_stats"]
    lines = [
        "# COCO Representation Probe",
        "",
        "## Figure Contract",
        "",
        f"- Core conclusion: {summary['figure_contract']['core_conclusion']}",
        f"- Dataset: {summary['dataset']}",
        f"- Representation seed: {summary['seed_for_representation']}",
        f"- Cold query items: {summary['cold_query_items']}",
        f"- Warm candidate items: {summary['warm_candidate_items']}",
        "",
        "## Cold-Course Exposure Risk",
        "",
        f"- Paired seed-course units: {risk['paired_units']}",
        f"- Mean CKG-RL - CGRC N@10 delta: {risk['mean_delta_n10']:.6f}",
        f"- Bootstrap 95% CI for mean delta: [{risk['mean_delta_n10_ci_low']:.6f}, {risk['mean_delta_n10_ci_high']:.6f}]",
        f"- Median CKG-RL - CGRC N@10 delta: {risk['median_delta_n10']:.6f}",
        f"- Win / tie / loss counts: {risk['wins']} / {risk['ties']} / {risk['losses']}",
        f"- Win / tie / loss ratios: {risk['gain_ratio']:.3f} / {risk['tie_ratio']:.3f} / {risk['loss_ratio']:.3f}",
        "",
        "Interpretation note: the paired distribution and bootstrap interval are the primary evidence. The t-SNE panel is a qualitative appendix-style sanity check, not a standalone proof.",
        "",
        "Suggested paper use: use `coco_cold_exposure_risk` and `coco_neighbor_purity` as separate analysis figures. Keep `coco_cold_course_tsne` in the appendix as a qualitative visualization only.",
        "",
        "Caption draft for `coco_cold_exposure_risk`: Per-cold-course exposure-risk analysis on COCO. (a) Ranked paired NDCG@10 deltas between CKG-RL and CGRC over seed-course units; the inset reports the bootstrap 95% confidence interval of the mean delta. (b) Empirical cumulative distribution of paired deltas, with the zero-effect reference and mean delta marked. (c) Hexbin density of paired per-course NDCG@10 scores; the diagonal marks equal performance.",
        "",
        "Caption draft for `coco_neighbor_purity`: Warm-neighbor label purity of cold-course representations on COCO. For each cold course, the nearest warm courses are retrieved in the learned representation space and evaluated by shared category or concept labels. Violins and boxes show the per-course distribution; points with error bars show mean +/- bootstrap 95% CI. CKG-RL improves concept-level neighborhood purity while preserving coarse category structure.",
        "",
        "## Warm-Neighbor Purity",
        "",
        "| Method | Label group | k | Mean purity | Evaluated queries | Empty-label queries |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in neighbor_summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.label_group} | {row.k} | "
            f"{row.mean_purity:.4f} | {row.n_evaluated} | {row.n_empty_label_queries} |"
        )
    lines.extend(
        [
            "",
            "Plotting note: the optimized neighbor-purity figure shows a deterministic point subsample for legibility; box/violin summaries and bootstrap 95% confidence intervals are computed from all evaluated cold courses.",
            "",
        "## Generated Files",
        "",
            "- `coco_topconf_probe.svg/pdf/tiff/png`",
            "- `coco_cold_exposure_risk.svg/pdf/tiff/png`",
            "- `coco_neighbor_purity.svg/pdf/tiff/png`",
            "- `coco_cold_course_tsne.svg/pdf/tiff/png`",
            "- `coco_paired_cold_item_delta.csv`",
            "- `coco_neighbor_purity.csv`",
            "- `coco_neighbor_purity_detail.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe cold-course exposure and representation structure.")
    parser.add_argument("--dataset", default="coco", choices=["coco"])
    parser.add_argument("--out-dir", default="output/figures/representation_probe")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.dataset == "coco":
        summary = run_coco_probe(out_dir)
    else:
        raise ValueError(args.dataset)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
