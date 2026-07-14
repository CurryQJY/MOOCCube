from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2025, 2026, 2027)
METRICS = ("R@5", "R@10", "N@5", "N@10")
RECPPO_NAME = "RecPPO (w=0.5, r=0.04)"
CGRC_NAME = "CGRC (exact re-export)"

RECPPO_DIRS = {
    2025: ROOT
    / "outputs/recppo_research_repair/recppo_residual_stage2_w050_seed2025/res004_w050/strict_item_cold_balanced_thr1_seed_2025",
    2026: ROOT
    / "outputs/recppo_research_repair/final_candidate_w050_res004_seeds2026_2027/strict_item_cold_balanced_thr1_seed_2026",
    2027: ROOT
    / "outputs/recppo_research_repair/final_candidate_w050_res004_seeds2026_2027/strict_item_cold_balanced_thr1_seed_2027",
}


def cgrc_dir(seed: int) -> Path:
    return (
        ROOT
        / "outputs/content_delta_pop5/static_item_cold_balanced"
        / f"strict_item_cold_balanced_thr1_seed_{seed}"
        / "significance_cgrc_exact_reexport"
    )


def read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one result object in {path}, found {len(value)}")
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def recppo_metrics(seed: int) -> tuple[dict[str, float], pd.DataFrame]:
    directory = RECPPO_DIRS[seed]
    result = pd.read_csv(
        directory / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
    ).iloc[0]
    metrics = {
        metric: float(result[f"full_cold_item_macro_{metric.replace('@', '').lower()}"])
        for metric in METRICS
    }
    per_item = pd.read_csv(
        directory / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
    )
    return metrics, per_item


def cgrc_metrics(seed: int) -> tuple[dict[str, float], pd.DataFrame]:
    directory = cgrc_dir(seed)
    result = read_json_object(directory / "cgrc_paper_static_result.json")
    metrics = {metric: float(result["full_cold_item_macro"][metric]) for metric in METRICS}
    per_item = pd.read_csv(directory / "per_item_full_cold_cgrc_paper_static.csv")
    return metrics, per_item


def validate_metric_export(
    method: str,
    seed: int,
    metrics: dict[str, float],
    per_item: pd.DataFrame,
    tolerance: float = 1e-8,
) -> None:
    if per_item["item_id"].duplicated().any():
        raise ValueError(f"{method} seed {seed}: duplicate item_id values")
    for metric in METRICS:
        delta = abs(float(per_item[metric].mean()) - metrics[metric])
        if delta > tolerance:
            raise ValueError(
                f"{method} seed {seed} {metric}: aggregate/per-item mismatch {delta:.3e}"
            )


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = pd.Series(differences[differences != 0])
    if nonzero.empty:
        return 0.0
    ranks = nonzero.abs().rank(method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def significance_table(bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260713)
    paired: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}

    for seed in SEEDS:
        ours_metrics, ours = recppo_metrics(seed)
        base_metrics, base = cgrc_metrics(seed)
        validate_metric_export(RECPPO_NAME, seed, ours_metrics, ours)
        validate_metric_export(CGRC_NAME, seed, base_metrics, base)

        ours = ours.set_index("item_id").sort_index()
        base = base.set_index("item_id").sort_index()
        if not ours.index.equals(base.index):
            raise ValueError(f"Seed {seed}: RecPPO and CGRC item_id sets differ")
        for metric in METRICS:
            paired[metric].append(
                ours[metric].to_numpy(dtype=float) - base[metric].to_numpy(dtype=float)
            )

    rows = []
    for metric in METRICS:
        seed_parts = paired[metric]
        differences = np.concatenate(seed_parts)
        statistic, p_raw = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        bootstrap = np.empty(bootstrap_samples, dtype=float)
        for index in range(bootstrap_samples):
            sampled = [
                part[rng.integers(0, len(part), size=len(part))] for part in seed_parts
            ]
            bootstrap[index] = float(np.concatenate(sampled).mean())
        ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(differences),
                "mean_delta_recppo_minus_cgrc": float(differences.mean()),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "wilcoxon_W": float(statistic),
                "p_raw": float(p_raw),
                "p_bonferroni_4": min(1.0, 4.0 * float(p_raw)),
                "p_bonferroni_12": min(1.0, 12.0 * float(p_raw)),
                "rank_biserial": rank_biserial(differences),
            }
        )
    return pd.DataFrame(rows)


def main_table() -> pd.DataFrame:
    rows = []
    for method, loader in ((CGRC_NAME, cgrc_metrics), (RECPPO_NAME, recppo_metrics)):
        seed_values = {metric: [] for metric in METRICS}
        for seed in SEEDS:
            metrics, per_item = loader(seed)
            validate_metric_export(method, seed, metrics, per_item)
            for metric in METRICS:
                seed_values[metric].append(metrics[metric])
        row: dict[str, object] = {"method": method, "seeds": "2025,2026,2027"}
        for metric in METRICS:
            values = np.asarray(seed_values[metric], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def write_latex(table: pd.DataFrame, significance: pd.DataFrame, path: Path) -> None:
    sig = significance.set_index("metric")
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & R@5 & R@10 & N@5 & N@10 \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = []
        for metric in METRICS:
            cell = f"{row[f'{metric}_mean']:.4f} $\\pm$ {row[f'{metric}_std']:.4f}"
            if row["method"] == RECPPO_NAME:
                stars = significance_stars(
                    float(sig.loc[metric, "p_bonferroni_12"])
                )
                if stars:
                    cell += rf"\textsuperscript{{\scriptsize {stars}}}"
            cells.append(cell)
        lines.append(f"{row['method']} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"% Stars compare RecPPO with CGRC using paired Wilcoxon tests.",
            r"% P-values use Bonferroni correction over 12 tests (3 datasets x 4 metrics).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    output_dir = ROOT / "paper_aaai27/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = main_table()
    significance = significance_table(args.bootstrap_samples)
    table_path = output_dir / "recppo_mooccube_exact_main_table.csv"
    significance_path = output_dir / "recppo_mooccube_exact_significance.csv"
    latex_path = ROOT / "paper_aaai27/recppo_mooccube_exact_table.tex"
    table.to_csv(table_path, index=False)
    significance.to_csv(significance_path, index=False)
    write_latex(table, significance, latex_path)
    print(table.to_string(index=False))
    print(significance.to_string(index=False))
    print(f"Wrote {table_path}")
    print(f"Wrote {significance_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
