"""Export the Junyi cold-only table from provenance-checked strictfix results.

This script intentionally fails closed.  A baseline row is accepted only when
its item-macro cold count matches the current strict split's test_cold_items
and the result was produced after the current split artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(r"D:\DeskTop\MOOCCube")
OUT_DIR = ROOT / "output" / "doc" / "junyi_results"
TMP_DIR = ROOT / "tmp"

SEEDS = (2025, 2026, 2027)
METRICS = ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")
CSV_SUFFIX = {
    "R@5": "r5",
    "R@10": "r10",
    "R@20": "r20",
    "N@5": "n5",
    "N@10": "n10",
    "N@20": "n20",
}


@dataclass(frozen=True)
class JsonMethod:
    name: str
    latex_name: str
    subdir: str
    filename: str


JSON_METHODS = (
    JsonMethod("Popularity", "Popularity", "popularity_compare", "popularity_static_result.json"),
    JsonMethod("BPR", r"BPR~\cite{rendle2009bpr}", "bpr_compare", "bpr_static_result.json"),
    JsonMethod("LightGCN", r"LightGCN~\cite{he2020lightgcn}", "lightgcn_compare", "lightgcn_static_result.json"),
    JsonMethod(
        "DropoutNet",
        r"DropoutNet~\cite{volkovs2017dropoutnet}",
        "dropoutnet_compare",
        "dropoutnet_official_static_result.json",
    ),
    JsonMethod("Content-CBF", "Content-CBF", "content_profile_compare", "content_profile_static_result.json"),
    JsonMethod("CCFCRec", r"CCFCRec~\cite{zhou2023ccfcrec}", "ccfcrec_compare", "ccfcrec_static_result.json"),
    JsonMethod("ALDI", r"ALDI~\cite{huang2023aldi}", "aldi_compare", "aldi_static_result.json"),
    JsonMethod("CGRC", r"CGRC~\cite{kim2024cgrc}", "cgrc_paper_compare_strictfix", "cgrc_paper_static_result.json"),
)


def run_dir(seed: int, *, ours: bool = False, cgrc: bool = False) -> Path:
    if seed == 2025:
        if ours or cgrc:
            return ROOT / "outputs/junyi/mask_ablation/mask_tt/strict_item_cold_balanced_thr1_seed_2025"
        return ROOT / "outputs/junyi/official_prereq_seed2025/strict_item_cold_balanced_thr1_seed_2025"
    return ROOT / f"outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_{seed}"


def read_json_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "value" in data:
        return data["value"][0]
    if isinstance(data, list):
        return data[0]
    return data


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_summary(path: Path) -> dict:
    summary_path = path / "static_split_summary.json"
    if not summary_path.exists():
        manifest_path = path / "static_protocol_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["split"]
    return json.loads(summary_path.read_text(encoding="utf-8"))


def assert_current_split_result(seed: int, result_path: Path, row: dict) -> None:
    split_dir = run_dir(seed)
    summary = split_summary(split_dir)
    expected = int(summary["test_cold_items"])
    got = int(row["count_full_cold_item_macro"])
    if got != expected:
        raise ValueError(
            f"{result_path} has count_full_cold_item_macro={got}, expected {expected}"
        )
    if result_path.stat().st_mtime < (split_dir / "static_test.pkl").stat().st_mtime:
        raise ValueError(f"{result_path} is older than current static_test.pkl")


def baseline_result_path(method: JsonMethod, seed: int) -> Path:
    base = run_dir(seed, cgrc=(method.name == "CGRC"))
    if method.name == "CGRC":
        return base / method.subdir / method.filename
    strictfix = base / f"{method.subdir}_strictfix" / method.filename
    if strictfix.exists():
        return strictfix
    if seed == 2025:
        return base / method.subdir / method.filename
    raise FileNotFoundError(
        f"Missing strictfix result for {method.name} seed={seed}: {strictfix}"
    )


def load_json_method(method: JsonMethod) -> dict:
    values = {metric: [] for metric in METRICS}
    sources = []
    for seed in SEEDS:
        path = baseline_result_path(method, seed)
        row = read_json_result(path)
        assert_current_split_result(seed, path, row)
        macro = row["full_cold_item_macro"]
        for metric in METRICS:
            values[metric].append(float(macro[metric]))
        sources.append(str(path))
    return {"method": method.name, "latex_name": method.latex_name, "values": values, "sources": sources}


def load_ours() -> dict:
    values = {metric: [] for metric in METRICS}
    sources = []
    for seed in SEEDS:
        base = run_dir(seed, ours=True)
        per_item = base / "per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
        detail = base / "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        rows = read_csv_rows(per_item)
        expected = int(split_summary(base)["test_cold_items"])
        if len(rows) != expected:
            raise ValueError(f"{per_item} has {len(rows)} item rows, expected {expected}")
        result = read_csv_row(detail)
        for metric in METRICS:
            values[metric].append(float(result[f"full_cold_item_macro_{CSV_SUFFIX[metric]}"]))
        sources.append(str(detail))
    return {"method": "CKG-RL", "latex_name": r"\textbf{CKG-RL}", "values": values, "sources": sources}


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    return sum(values) / len(values), statistics.stdev(values)


def load_stars() -> dict[str, str]:
    path = ROOT / "outputs/junyi/cgrc_strictfix_significance/per_course_ours_vs_cgrc_strictfix_summary.csv"
    if not path.exists():
        return {}
    stars = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["scope"] != "pooled_seed_course":
                continue
            p_value = float(row["randomization_p"])
            if p_value < 0.001:
                stars[row["metric"]] = "***"
            elif p_value < 0.01:
                stars[row["metric"]] = "**"
            elif p_value < 0.05:
                stars[row["metric"]] = "*"
            else:
                stars[row["metric"]] = ""
    return stars


def format_cell(mean: float, std: float, *, bold: bool, star: str = "") -> str:
    head = f"{mean:.4f}"
    if bold:
        head = rf"\textbf{{{head}}}"
    cell = rf"{head}\sd{{{std:.4f}}}"
    if star:
        cell += rf"\sigmark{{{star}}}"
    return cell


def build_rows() -> tuple[list[dict], list[str]]:
    rows = [load_json_method(method) for method in JSON_METHODS]
    rows.append(load_ours())
    stats_rows = []
    for row in rows:
        stats = {"method": row["method"], "latex_name": row["latex_name"], "sources": row["sources"]}
        for metric in METRICS:
            mean, std = mean_std(row["values"][metric])
            stats[f"{metric}_mean"] = mean
            stats[f"{metric}_std"] = std
        stats_rows.append(stats)

    best = {
        metric: max(float(row[f"{metric}_mean"]) for row in stats_rows)
        for metric in METRICS
    }
    stars = load_stars()
    latex_rows = []
    for row in stats_rows:
        cells = []
        for metric in METRICS:
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            star = stars.get(metric, "") if row["method"] == "CKG-RL" else ""
            cells.append(format_cell(mean, std, bold=abs(mean - best[metric]) < 1e-12, star=star))
        latex_rows.append(f"{row['latex_name']} & " + " & ".join(cells) + r" \\")
    return stats_rows, latex_rows


def write_outputs(stats_rows: list[dict], latex_rows: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "junyi_cold_table_strictfix_mean_std.csv"
    latex_path = OUT_DIR / "junyi_cold_table_strictfix_rows.tex"
    tmp_csv = TMP_DIR / "junyi_cold_table_exact_mean_std_strictfix.csv"

    fieldnames = ["method"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    fieldnames.append("sources")

    for path in (csv_path, tmp_csv):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in stats_rows:
                writer.writerow({
                    **{key: row.get(key, "") for key in fieldnames if key != "sources"},
                    "sources": " | ".join(row["sources"]),
                })
    latex_path.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    print(csv_path)
    print(latex_path)
    print(tmp_csv)


def update_tex(latex_rows: list[str], tex_path: Path) -> None:
    text = tex_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\\label\{tab:junyi-cold\}.*?\\midrule\n)(.*?)(\n\\bottomrule)",
        flags=re.DOTALL,
    )
    def replacement(match: re.Match) -> str:
        return match.group(1) + "\n".join(latex_rows) + match.group(3)

    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Could not locate Junyi table body in {tex_path}")
    tex_path.write_text(new_text, encoding="utf-8")
    print(tex_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-tex", action="store_true")
    parser.add_argument("--tex", default=str(ROOT / "paper_wsdm/main.tex"))
    args = parser.parse_args()

    stats_rows, latex_rows = build_rows()
    write_outputs(stats_rows, latex_rows)
    if args.update_tex:
        update_tex(latex_rows, Path(args.tex))


if __name__ == "__main__":
    main()
