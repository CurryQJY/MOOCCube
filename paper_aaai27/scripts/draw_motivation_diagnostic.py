from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_aaai27"
FIG_DIR = PAPER / "figures"
BASE = FIG_DIR / "mooccube_motivation_diagnostic"
SEEDS = (2025, 2026, 2027)

STAGES = [
    ("sampled_micro", "Sampled\nmicro", "sample_cold"),
    ("full_micro", "Full\nmicro", "full_cold"),
    ("full_item_macro", "Full\nitem-macro", "full_cold_item_macro"),
]

PANEL_A_SOURCES = {
    "BPR": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/bpr_static_result.json",
    "LightGCN": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/lightgcn_static_result.json",
    "DropoutNet": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_dropoutnet_official_teacher80_student120_v1/dropoutnet_official_static_result.json",
    "ALDI": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/aldi_official_static_result.json",
}

PANEL_B_JSON_SOURCES = {
    "BPR": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/bpr_static_result.json",
    "LightGCN": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/lightgcn_static_result.json",
    "DropoutNet": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_dropoutnet_official_teacher80_student120_v1/dropoutnet_official_static_result.json",
    "CCFCRec": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/ccfcrec_static_result.json",
    "ALDI": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/aldi_official_static_result.json",
    "CGRC": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_cgrc_paper_v1/cgrc_paper_static_result.json",
    "USIM": lambda seed: ROOT
    / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/main_table_balanced_itemmacro_v1/usim_official_static_result.json",
}

PANEL_B_ORDER = ["BPR", "LightGCN", "DropoutNet", "USIM", "CCFCRec", "ALDI", "CGRC", "CKG-RL"]

PALETTE = {
    "BPR": "#6C6C6C",
    "LightGCN": "#1F77B4",
    "DropoutNet": "#E69F00",
    "ALDI": "#009E73",
    "CCFCRec": "#8E5EA2",
    "CGRC": "#D55E00",
    "USIM": "#4C566A",
    "CKG-RL": "#C9403A",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.75,
            "font.size": 7.4,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.7,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.4,
        }
    )


def read_record(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[0] if isinstance(payload, list) else payload


def collect_protocol_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, make_path in PANEL_A_SOURCES.items():
        for seed in SEEDS:
            path = make_path(seed)
            record = read_record(path)
            for stage_key, stage_label, json_key in STAGES:
                metrics = record.get(json_key, {})
                if not metrics or "N@10" not in metrics:
                    raise ValueError(f"Missing {json_key}.N@10 in {path}")
                rows.append(
                    {
                        "panel": "protocol",
                        "method": method,
                        "seed": seed,
                        "stage": stage_key,
                        "stage_label": stage_label.replace("\n", " "),
                        "n10": float(metrics["N@10"]),
                        "r10": float(metrics["R@10"]),
                        "source": str(path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(rows)


def collect_aggregation_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, make_path in PANEL_B_JSON_SOURCES.items():
        for seed in SEEDS:
            path = make_path(seed)
            record = read_record(path)
            for metric_key, label in [("full_cold", "Full micro"), ("full_cold_item_macro", "Full item-macro")]:
                metrics = record.get(metric_key, {})
                if not metrics or "N@10" not in metrics:
                    raise ValueError(f"Missing {metric_key}.N@10 in {path}")
                rows.append(
                    {
                        "panel": "aggregation",
                        "method": method,
                        "seed": seed,
                        "eval": label,
                        "n10": float(metrics["N@10"]),
                        "r10": float(metrics["R@10"]),
                        "source": str(path.relative_to(ROOT)),
                    }
                )

    ckg_root = ROOT / "outputs" / "significance_per_item_exports" / "mooccube" / "ckg_rl_full"
    for seed in SEEDS:
        path = ckg_root / f"strict_item_cold_balanced_thr1_seed_{seed}" / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Empty final metric file: {path}")
        record = frame.iloc[0]
        rows.extend(
            [
                {
                    "panel": "aggregation",
                    "method": "CKG-RL",
                    "seed": seed,
                    "eval": "Full micro",
                    "n10": float(record["full_cold_n10"]),
                    "r10": float(record["full_cold_r10"]),
                    "source": str(path.relative_to(ROOT)),
                },
                {
                    "panel": "aggregation",
                    "method": "CKG-RL",
                    "seed": seed,
                    "eval": "Full item-macro",
                    "n10": float(record["full_cold_item_macro_n10"]),
                    "r10": float(record["full_cold_item_macro_r10"]),
                    "source": str(path.relative_to(ROOT)),
                },
            ]
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, as_index=False)
        .agg(
            mean_n10=("n10", "mean"),
            std_n10=("n10", "std"),
            mean_r10=("r10", "mean"),
            std_r10=("r10", "std"),
            n=("seed", "nunique"),
            seeds=("seed", lambda values: ",".join(str(int(x)) for x in sorted(set(values)))),
        )
        .fillna({"std_n10": 0.0, "std_r10": 0.0})
    )


def draw(protocol: pd.DataFrame, aggregation: pd.DataFrame) -> None:
    configure_style()
    protocol_summary = summarize(protocol, ["method", "stage", "stage_label"])
    aggregation_summary = summarize(aggregation, ["method", "eval"])

    fig, (ax_a, ax_b) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(3.35, 4.35),
        gridspec_kw={"height_ratios": [1.1, 1.28]},
        constrained_layout=False,
    )

    x = np.arange(len(STAGES))
    stage_order = [stage for stage, _, _ in STAGES]
    stage_labels = [label for _, label, _ in STAGES]
    markers = {"BPR": "o", "LightGCN": "s", "DropoutNet": "^", "ALDI": "D"}
    linestyles = {"BPR": "-", "LightGCN": "--", "DropoutNet": "-.", "ALDI": ":"}

    for method in PANEL_A_SOURCES:
        data = (
            protocol_summary[protocol_summary["method"] == method]
            .set_index("stage")
            .reindex(stage_order)
            .reset_index()
        )
        ax_a.errorbar(
            x,
            data["mean_n10"],
            yerr=data["std_n10"],
            color=PALETTE[method],
            linestyle=linestyles[method],
            marker=markers[method],
            markersize=3.6,
            linewidth=1.15,
            elinewidth=0.7,
            capsize=1.7,
            label=method,
            zorder=3,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(stage_labels)
    ax_a.set_ylabel("Cold NDCG@10")
    ax_a.set_ylim(-0.015, 0.56)
    ax_a.grid(axis="y", color="#D8D8D8", lw=0.5, linestyle="--", zorder=0)
    ax_a.legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.5,
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
    )
    ax_a.text(-0.12, 1.08, "(a)", transform=ax_a.transAxes, fontsize=8.0, fontweight="bold", va="top")

    methods = PANEL_B_ORDER
    y_pos = np.arange(len(methods))
    full_micro = aggregation_summary[aggregation_summary["eval"] == "Full micro"].set_index("method")
    item_macro = aggregation_summary[aggregation_summary["eval"] == "Full item-macro"].set_index("method")

    for y, method in enumerate(methods):
        micro_mean = float(full_micro.loc[method, "mean_n10"])
        macro_mean = float(item_macro.loc[method, "mean_n10"])
        ax_b.plot([micro_mean, macro_mean], [y, y], color="#B8B8B8", lw=0.9, zorder=1)
        ax_b.errorbar(
            micro_mean,
            y,
            xerr=float(full_micro.loc[method, "std_n10"]),
            fmt="o",
            color="#7A7A7A",
            ecolor="#A8A8A8",
            markersize=3.4,
            elinewidth=0.7,
            capsize=1.5,
            label="Full micro" if y == 0 else None,
            zorder=3,
        )
        ax_b.errorbar(
            macro_mean,
            y,
            xerr=float(item_macro.loc[method, "std_n10"]),
            fmt="s",
            color="#2F6FA3",
            ecolor="#7AA0C4",
            markersize=3.5,
            elinewidth=0.7,
            capsize=1.5,
            label="Full item-macro" if y == 0 else None,
            zorder=4,
        )

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(methods)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Cold NDCG@10")
    ax_b.set_xlim(-0.008, 0.235)
    ax_b.grid(axis="x", color="#D8D8D8", lw=0.5, linestyle="--", zorder=0)
    ax_b.legend(loc="upper right", frameon=False, handlelength=1.2)
    ax_b.text(-0.12, 1.07, "(b)", transform=ax_b.transAxes, fontsize=8.0, fontweight="bold", va="top")

    for ax in (ax_a, ax_b):
        ax.tick_params(length=2.4, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(0.01, 0.012, "Points are three-seed means; bars denote SD.", fontsize=6.2, color="#555555")
    fig.subplots_adjust(left=0.22, right=0.985, top=0.97, bottom=0.095, hspace=0.46)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(BASE.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    protocol.to_csv(FIG_DIR / "mooccube_motivation_protocol_seed_data.csv", index=False)
    aggregation.to_csv(FIG_DIR / "mooccube_motivation_aggregation_seed_data.csv", index=False)
    pd.concat(
        [
            protocol_summary.assign(panel="protocol"),
            aggregation_summary.assign(panel="aggregation"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(FIG_DIR / "mooccube_motivation_diagnostic_summary.csv", index=False)


def main() -> None:
    protocol = collect_protocol_data()
    aggregation = collect_aggregation_data()
    draw(protocol, aggregation)
    print(f"Wrote {BASE.with_suffix('.pdf')}")
    print(f"Wrote {BASE.with_suffix('.png')}")
    print(f"Wrote {BASE.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
