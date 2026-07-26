from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EVIDENCE_SPECS = (
    {
        "evidence_group": "Course-structure limitation",
        "source": "comparison",
        "baseline": "pcgnn",
        "contrast": "CKG-RL vs. PCGNN",
        "metric": "prerequisite_gap",
        "signal": "Prerequisite gap $\\downarrow$",
        "orientation": "lower",
    },
    {
        "evidence_group": "Course-structure limitation",
        "source": "comparison",
        "baseline": "pcgnn",
        "contrast": "CKG-RL vs. PCGNN",
        "metric": "difficulty_gap",
        "signal": "Difficulty gap $\\downarrow$",
        "orientation": "lower",
    },
    {
        "evidence_group": "Course-reward mechanism",
        "source": "mechanism",
        "baseline": "ckg_rl_wo_course_reward",
        "contrast": "Full vs. w/o course reward",
        "metric": "cold_prerequisite_gap",
        "signal": "Cold prerequisite gap $\\downarrow$",
        "orientation": "lower",
    },
    {
        "evidence_group": "Course-reward mechanism",
        "source": "mechanism",
        "baseline": "ckg_rl_wo_course_reward",
        "contrast": "Full vs. w/o course reward",
        "metric": "cold_difficulty_gap",
        "signal": "Cold difficulty gap $\\downarrow$",
        "orientation": "lower",
    },
    {
        "evidence_group": "Course-reward mechanism",
        "source": "mechanism",
        "baseline": "ckg_rl_wo_course_reward",
        "contrast": "Full vs. w/o course reward",
        "metric": "cold_proportion",
        "signal": "Cold-course share $\\uparrow$",
        "orientation": "higher",
    },
)


MAIN_EFFECTIVENESS_ROWS = (
    {
        "evidence_group": "Strict cold-start effectiveness",
        "contrast": "CKG-RL vs. CGRC",
        "signal": "Recall@10 $\\uparrow$",
        "metric": "recall_at_10",
        "ckg_rl_value": 0.2863,
        "reference_value": 0.2589,
        "relative_gain": 0.106,
    },
    {
        "evidence_group": "Strict cold-start effectiveness",
        "contrast": "CKG-RL vs. CGRC",
        "signal": "NDCG@10 $\\uparrow$",
        "metric": "ndcg_at_10",
        "ckg_rl_value": 0.2098,
        "reference_value": 0.1845,
        "relative_gain": 0.137,
    },
)


def _select_one(frame: pd.DataFrame, mask, description: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {description}, found {len(selected)}")
    return selected.iloc[0]


def build_motivation_evidence(
    comparison_paired: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    mechanism_paired: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
) -> pd.DataFrame:
    sources = {
        "comparison": (comparison_paired, comparison_summary),
        "mechanism": (mechanism_paired, mechanism_summary),
    }
    rows = []
    for spec in EVIDENCE_SPECS:
        paired, summary = sources[spec["source"]]
        paired_row = _select_one(
            paired,
            paired["treatment"].eq("ckg_rl")
            & paired["baseline"].eq(spec["baseline"])
            & paired["cutoff"].eq(10)
            & paired["metric"].eq(spec["metric"]),
            f"{spec['contrast']} / {spec['metric']}",
        )
        if int(paired_row["pair_count"]) != 204:
            raise ValueError(
                f"motivation evidence pair coverage mismatch for {spec['contrast']}"
            )

        mean_difference = float(paired_row["mean_difference"])
        raw_low = float(paired_row["bootstrap_ci_low"])
        raw_high = float(paired_row["bootstrap_ci_high"])
        if spec["orientation"] == "lower":
            improvement = -mean_difference
            ci_low, ci_high = -raw_high, -raw_low
            supported = (
                str(paired_row["interpretation"]) == "supports" and raw_high < 0.0
            )
            scale = 1.0
            unit = "risk"
        else:
            improvement = mean_difference
            ci_low, ci_high = raw_low, raw_high
            supported = raw_low > 0.0
            scale = 100.0
            unit = "percent"
        if not supported:
            raise ValueError(
                f"{spec['contrast']} / {spec['metric']} does not support the "
                "motivation claim"
            )

        full_row = _select_one(
            summary,
            summary["model"].eq("ckg_rl") & summary["cutoff"].eq(10),
            f"CKG-RL summary / {spec['metric']}",
        )
        reference_row = _select_one(
            summary,
            summary["model"].eq(spec["baseline"])
            & summary["cutoff"].eq(10),
            f"{spec['baseline']} summary / {spec['metric']}",
        )
        value_column = f"{spec['metric']}_mean"
        full_value = float(full_row[value_column])
        reference_value = float(reference_row[value_column])
        rows.append(
            {
                "evidence_group": spec["evidence_group"],
                "contrast": spec["contrast"],
                "signal": spec["signal"],
                "metric": spec["metric"],
                "ckg_rl_value": full_value * scale,
                "reference_value": reference_value * scale,
                "improvement": improvement * scale,
                "ci_low": ci_low * scale,
                "ci_high": ci_high * scale,
                "unit": unit,
                "pair_count": int(paired_row["pair_count"]),
                "permutation_p_value": float(
                    paired_row["permutation_p_value"]
                ),
                "p_display": None,
                "relative_gain": float("nan"),
            }
        )

    main_rows = []
    for spec in MAIN_EFFECTIVENESS_ROWS:
        main_rows.append(
            {
                **spec,
                "improvement": spec["ckg_rl_value"] - spec["reference_value"],
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "unit": "score",
                "pair_count": 204,
                "permutation_p_value": float("nan"),
                "p_display": "$p_{\\mathrm{Holm}}<.01$",
            }
        )

    structure_rows = [
        row for row in rows if row["evidence_group"] == "Course-structure limitation"
    ]
    mechanism_rows = [
        row for row in rows if row["evidence_group"] == "Course-reward mechanism"
    ]
    return pd.DataFrame(structure_rows + main_rows + mechanism_rows)


def _format_p(row: pd.Series) -> str:
    if pd.notna(row.get("p_display")):
        return str(row["p_display"])
    value = float(row["permutation_p_value"])
    if value < 0.0001:
        return "$<.0001$"
    return f"${value:.4f}$"


def _format_value(row: pd.Series, column: str) -> str:
    value = float(row[column])
    if row["unit"] == "percent":
        return f"{value:.1f}\\%"
    return f"{value:.4f}"


def _format_effect(row: pd.Series) -> str:
    if row["unit"] == "score":
        gain_percent = float(row["relative_gain"]) * 100.0
        return f"+{row['improvement']:.4f} (+{gain_percent:.1f}\\%)"
    if row["unit"] == "percent":
        return (
            f"+{row['improvement']:.1f} pp "
            f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
        )
    return (
        f"{row['improvement']:.4f} "
        f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
    )


def render_motivation_table(evidence: pd.DataFrame) -> str:
    lines = [
        "% Requires \\usepackage{booktabs,threeparttable}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\begin{threeparttable}",
        "\\caption{Role-aligned, empirically supported motivation evidence on held-out MOOCCube test sets.}",
        "\\label{tab:p1-motivation-evidence}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrc}",
        "\\toprule",
        "Contrast & Verified signal & CKG-RL & Reference & Effect / improvement & $p$ \\\\",
        "\\midrule",
    ]
    for group, group_rows in evidence.groupby("evidence_group", sort=False):
        lines.append(
            f"\\multicolumn{{6}}{{l}}{{\\textit{{{group}}}}} " + r"\\"
        )
        for _, row in group_rows.iterrows():
            lines.append(
                " & ".join(
                    [
                        str(row["contrast"]),
                        str(row["signal"]),
                        _format_value(row, "ckg_rl_value"),
                        _format_value(row, "reference_value"),
                        _format_effect(row),
                        _format_p(row),
                    ]
                )
                + " \\\\"
            )
        if group != evidence["evidence_group"].iloc[-1]:
            lines.append("\\addlinespace[1pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\begin{tablenotes}[flushleft]",
            "\\scriptsize",
            "\\item Lower-is-better risk improvements use Reference minus CKG-RL. Their 95\\% bootstrap intervals and permutation tests use 204 paired seed-course units. Main effectiveness values are held-out three-seed means from the strict course-cold table; significance is the corresponding Holm-corrected paired Wilcoxon result over matched seed-course pairs.",
            "\\item Course-reward exposure is descriptive and does not establish pedagogical quality. Only predeclared supported motivation claims are shown; inconclusive or adverse outcomes are excluded from this claim table and remain reported in the complete risk audit.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    comparison_dir: Path,
    mechanism_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    comparison_dir = Path(comparison_dir).resolve()
    mechanism_dir = Path(mechanism_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_motivation_evidence(
        pd.read_csv(comparison_dir / "paired_statistics.csv"),
        pd.read_csv(comparison_dir / "model_summary.csv"),
        pd.read_csv(mechanism_dir / "paired_statistics.csv"),
        pd.read_csv(mechanism_dir / "model_summary.csv"),
    )
    csv_path = output_dir / "mooccube_p1_motivation_evidence.csv"
    latex_path = output_dir / "mooccube_p1_motivation_evidence.tex"
    evidence.to_csv(csv_path, index=False)
    latex_path.write_text(render_motivation_table(evidence), encoding="utf-8")
    return {"csv": csv_path, "latex": latex_path}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=(
            root
            / "paper_aaai27"
            / "figures"
            / "p1_topk_motivation_analysis"
        ),
    )
    parser.add_argument(
        "--mechanism-dir",
        type=Path,
        default=(
            root
            / "paper_aaai27"
            / "figures"
            / "p1_motivation_mechanism_analysis"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_aaai27" / "tables",
    )
    args = parser.parse_args()
    outputs = run(args.comparison_dir, args.mechanism_dir, args.output_dir)
    for kind, path in outputs.items():
        print(f"[P1 motivation table] {kind}: {path}", flush=True)


if __name__ == "__main__":
    main()
