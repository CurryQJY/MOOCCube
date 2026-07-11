from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_aaai27"
FIG_DIR = PAPER / "figures"
OUT_TEX = PAPER / "revision_stat_cost_tables.tex"

SEEDS = [2025, 2026, 2027]
METRICS = ["R@10", "N@10"]
SIGNIFICANCE_METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]
T_CRIT = {2: 12.706, 3: 4.303}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def item_macro_block_from_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data[0] if isinstance(data, list) and data else data
    if not isinstance(row, dict):
        return {}
    block = row.get("full_cold_item_macro", {})
    if not isinstance(block, dict):
        return {}
    return {key: float(value) for key, value in block.items() if isinstance(value, (int, float))}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def seed_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean, std = mean_std(values)
    n = len(values)
    if n <= 1:
        return mean, std, mean, mean
    tcrit = T_CRIT.get(n, 1.96)
    half = tcrit * std / math.sqrt(n)
    return mean, std, mean - half, mean + half


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 2027) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def wilcoxon_two_sided_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) == 0:
        return math.nan
    try:
        result = stats.wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
    except TypeError:
        result = stats.wilcoxon(
            nonzero,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
        )
    return float(result.pvalue)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    finite = [(idx, float(p)) for idx, p in enumerate(p_values) if not pd.isna(p)]
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running = 0.0
    for rank, (idx, p_value) in enumerate(finite):
        running = max(running, (m - rank) * p_value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def f4(x: float) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{x:.4f}"


def f1(x: float) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{x:.1f}"


def ci_cell(mean: float, low: float, high: float) -> str:
    return f"{f4(mean)} [{f4(low)}, {f4(high)}]"


def p_cell(x: float) -> str:
    if x is None or pd.isna(x):
        return "--"
    if x < 0.001:
        return f"{x:.1e}"
    return f"{x:.3f}"


def sec_cell(mean: float | None, std: float | None) -> str:
    if mean is None or pd.isna(mean):
        return "--"
    if std is None or pd.isna(std):
        return f"{f1(mean)}"
    return f"{f1(mean)}\\,$\\pm$\\,{f1(std)}"


def load_main_seed_values() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    # MOOCCube: exact True/True CKG-RL e60 run and CGRC main-table aggregate.
    cgrc = read_csv(
        ROOT
        / "outputs/content_delta_pop5/static_item_cold_balanced/main_table_balanced_itemmacro_cgrc_paper_v1/main_table_item_macro_detail.csv"
    )
    for seed in SEEDS:
        ours_path = (
            ROOT
            / f"outputs/significance_per_item_exports/mooccube/ckg_rl_full/strict_item_cold_balanced_thr1_seed_{seed}/final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        )
        ours = read_csv(ours_path).iloc[0]
        base = cgrc[cgrc["seed"].eq(seed)].iloc[0]
        rows.append(
            {
                "dataset": "MOOCCube",
                "seed": seed,
                "baseline": "CGRC",
                "ours_R@10": float(ours["full_cold_item_macro_r10"]),
                "ours_N@10": float(ours["full_cold_item_macro_n10"]),
                "baseline_R@10": float(base["cold_R10"]),
                "baseline_N@10": float(base["cold_N10"]),
                "source": "exact main-table seed aggregates",
            }
        )

    # Junyi: exact main-table three-seed composition. ALDI is the strongest non-CKG @10 baseline.
    junyi_ours = {
        2025: ROOT
        / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2025/final_fullrank_usim_feedback_fast3_content_delta_static.csv",
        2026: ROOT
        / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2026/final_fullrank_usim_feedback_fast3_content_delta_static.csv",
        2027: ROOT
        / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2027/final_fullrank_usim_feedback_fast3_content_delta_static.csv",
    }
    junyi_aldi = {
        2025: ROOT
        / "outputs/junyi/official_prereq_seed2025/strict_item_cold_balanced_thr1_seed_2025/aldi_compare/aldi_static_result.json",
        2026: ROOT
        / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2026/aldi_compare_strictfix/aldi_static_result.json",
        2027: ROOT
        / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2027/aldi_compare_strictfix/aldi_static_result.json",
    }
    for seed in SEEDS:
        ours = read_csv(junyi_ours[seed]).iloc[0]
        base = item_macro_block_from_json(junyi_aldi[seed])
        rows.append(
            {
                "dataset": "Junyi",
                "seed": seed,
                "baseline": "ALDI",
                "ours_R@10": float(ours["full_cold_item_macro_r10"]),
                "ours_N@10": float(ours["full_cold_item_macro_n10"]),
                "baseline_R@10": float(base["R@10"]),
                "baseline_N@10": float(base["N@10"]),
                "source": "exact main-table seed aggregates; ALDI strongest @10 baseline",
            }
        )

    # COCO: exact merged seed detail.
    coco = read_csv(
        ROOT / "outputs/coco/single_seed_triage/main_table_compare/main_table_item_macro_detail_with_ours.csv"
    )
    for seed in SEEDS:
        ours = coco[coco["seed"].eq(seed) & coco["model"].eq("Ours")].iloc[0]
        base = coco[coco["seed"].eq(seed) & coco["model"].eq("CCFCRec")].iloc[0]
        rows.append(
            {
                "dataset": "COCO",
                "seed": seed,
                "baseline": "CCFCRec",
                "ours_R@10": float(ours["cold_R10"]),
                "ours_N@10": float(ours["cold_N10"]),
                "baseline_R@10": float(base["cold_R10"]),
                "baseline_N@10": float(base["cold_N10"]),
                "source": "exact merged main-table seed detail",
            }
        )

    frame = pd.DataFrame(rows)
    for metric in METRICS:
        frame[f"diff_{metric}"] = frame[f"ours_{metric}"] - frame[f"baseline_{metric}"]
    return frame


def summarize_seed_ci(seed_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in seed_values.groupby("dataset", sort=False):
        baseline = str(group["baseline"].iloc[0])
        row: dict[str, object] = {
            "dataset": dataset,
            "baseline": baseline,
            "seeds": ",".join(map(str, sorted(group["seed"].tolist()))),
            "source": str(group["source"].iloc[0]),
        }
        for metric in METRICS:
            for prefix in ["ours", "baseline", "diff"]:
                mean, std, low, high = seed_ci(group[f"{prefix}_{metric}"].astype(float).tolist())
                row[f"{prefix}_{metric}_mean"] = mean
                row[f"{prefix}_{metric}_std"] = std
                row[f"{prefix}_{metric}_ci_low"] = low
                row[f"{prefix}_{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def load_per_course_pairs() -> pd.DataFrame:
    specs = [
        {
            "dataset": "MOOCCube",
            "baseline": "CGRC",
            "source": "main-table per-course diagnostic",
            "ours": {
                seed: ROOT
                / f"outputs/significance_per_item_exports/mooccube/ckg_rl_full/strict_item_cold_balanced_thr1_seed_{seed}/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
                for seed in SEEDS
            },
            "base": {
                seed: ROOT
                / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/significance_cgrc_exact_reexport/per_item_full_cold_cgrc_paper_static.csv"
                for seed in SEEDS
            },
        },
        {
            "dataset": "Junyi",
            "baseline": "ALDI",
            "source": "main-table per-course diagnostic",
            "ours": {
                2025: ROOT
                / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2025/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
                2026: ROOT
                / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2026/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
                2027: ROOT
                / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2027/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv",
            },
            "base": {
                2025: ROOT
                / "outputs/junyi/official_prereq_seed2025/strict_item_cold_balanced_thr1_seed_2025/aldi_compare/per_item_full_cold_aldi_static.csv",
                2026: ROOT
                / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2026/aldi_compare_strictfix/per_item_full_cold_aldi_static.csv",
                2027: ROOT
                / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2027/aldi_compare_strictfix/per_item_full_cold_aldi_static.csv",
            },
        },
        {
            "dataset": "COCO",
            "baseline": "CCFCRec",
            "source": "main-table per-course diagnostic",
            "ours": {
                seed: ROOT
                / f"outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}/per_item_full_cold_usim_feedback_fast3_content_delta_static.csv"
                for seed in SEEDS
            },
            "base": {
                seed: ROOT
                / f"outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}/main_table_compare/per_item_full_cold_ccfcrec_static.csv"
                for seed in SEEDS
            },
        },
    ]

    rows: list[pd.DataFrame] = []
    for spec in specs:
        for seed in SEEDS:
            ours = read_csv(spec["ours"][seed])
            base = read_csv(spec["base"][seed])
            merged = ours.merge(base, on="item_id", suffixes=("_ours", "_baseline"), how="inner")
            merged["dataset"] = spec["dataset"]
            merged["baseline"] = spec["baseline"]
            merged["seed"] = seed
            merged["source"] = spec["source"]
            for metric in SIGNIFICANCE_METRICS:
                merged[f"diff_{metric}"] = merged[f"{metric}_ours"] - merged[f"{metric}_baseline"]
            rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def summarize_per_course(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, baseline, source), group in pairs.groupby(["dataset", "baseline", "source"], sort=False):
        row: dict[str, object] = {
            "dataset": dataset,
            "baseline": baseline,
            "source": source,
            "paired_seed_items": int(len(group)),
        }
        for metric in METRICS:
            diff = group[f"diff_{metric}"].to_numpy(dtype=float)
            low, high = bootstrap_ci(diff)
            row[f"mean_diff_{metric}"] = float(np.mean(diff))
            row[f"ci_low_{metric}"] = low
            row[f"ci_high_{metric}"] = high
            row[f"median_diff_{metric}"] = float(np.median(diff))
            row[f"wins_{metric}"] = int((diff > 1e-12).sum())
            row[f"ties_{metric}"] = int((np.abs(diff) <= 1e-12).sum())
            row[f"losses_{metric}"] = int((diff < -1e-12).sum())
            row[f"loss_rate_{metric}"] = float((diff < -1e-12).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_significance(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, baseline, source), group in pairs.groupby(["dataset", "baseline", "source"], sort=False):
        for metric in SIGNIFICANCE_METRICS:
            diff = group[f"diff_{metric}"].to_numpy(dtype=float)
            raw_p = wilcoxon_two_sided_p(diff)
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "metric": metric,
                    "source": source,
                    "unit": "matched seed-course pairs",
                    "n_pairs": int(len(diff)),
                    "mean_delta": float(np.mean(diff)),
                    "median_delta": float(np.median(diff)),
                    "wins": int((diff > 1e-12).sum()),
                    "ties": int((np.abs(diff) <= 1e-12).sum()),
                    "losses": int((diff < -1e-12).sum()),
                    "test": "Wilcoxon signed-rank",
                    "alternative": "two-sided",
                    "raw_p": raw_p,
                }
            )

    frame = pd.DataFrame(rows)
    main_mask = frame["metric"].isin(["R@5", "R@10", "N@5", "N@10"])
    frame["holm_family"] = "supplement_all_metrics"
    frame["holm_p_all_metrics"] = holm_adjust(frame["raw_p"].astype(float).tolist())
    frame["holm_p_main_at5_at10"] = math.nan
    frame.loc[main_mask, "holm_p_main_at5_at10"] = holm_adjust(
        frame.loc[main_mask, "raw_p"].astype(float).tolist()
    )
    frame["significant_main_at5_at10"] = frame["holm_p_main_at5_at10"].astype(float) < 0.01
    frame["significant_all_metrics"] = frame["holm_p_all_metrics"].astype(float) < 0.05
    return frame


def parse_epoch_times(paths: list[Path]) -> list[float]:
    times: list[float] = []
    pat = re.compile(r"Time:\s*([0-9.]+)s")
    progress_pat = re.compile(
        r"\[CGRC-TRAIN-PROGRESS\] Epoch \d+/\d+ \| "
        r"(\d+)/(\d+) .*?elapsed=([0-9hms]+)"
    )

    def duration_to_seconds(value: str) -> float:
        match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", value)
        if not match:
            return math.nan
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return float(hours * 3600 + minutes * 60 + seconds)

    for path in paths:
        if not path.exists():
            continue
        for line in read_text_auto(path).splitlines():
            if "[STATIC-TRAIN] Epoch" in line or "[CGRC-TRAIN] Epoch" in line:
                match = pat.search(line)
                if match:
                    times.append(float(match.group(1)))
                continue
            if "[CGRC-TRAIN-PROGRESS]" not in line:
                continue
            match = progress_pat.search(line)
            if match and int(match.group(1)) == int(match.group(2)):
                elapsed = duration_to_seconds(match.group(3))
                if math.isfinite(elapsed):
                    times.append(elapsed)
    return times


def load_runtime_profiles() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in (ROOT / "outputs/runtime_profile").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for row in data:
            dataset = row.get("dataset")
            method = row.get("method")
            seed = row.get("seed")
            infer = row.get("final_infer_s", row.get("total_s"))
            if dataset and method and seed and infer is not None:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": int(seed),
                        "infer_s": float(infer),
                        "source_file": path.name,
                    }
                )
    runtime_result_subdirs = {"runtime_cgrc_profile", "cgrc_runtime_profile"}
    for path in ROOT.glob("outputs/**/cgrc_paper_static_result.json"):
        if not runtime_result_subdirs.intersection(path.parts):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        if "content_delta_pop5" in path.parts:
            dataset = "MOOCCube"
        elif "junyi" in path.parts:
            dataset = "Junyi"
        elif "coco" in path.parts:
            dataset = "COCO"
        else:
            continue
        for row in data:
            infer = row.get("final_infer_s", row.get("total_s"))
            seed = row.get("static_seed")
            if seed is None:
                seed = row.get("seed")
            if infer is not None and seed is not None:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": "CGRC",
                        "seed": int(seed),
                        "infer_s": float(infer),
                        "source_file": str(path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(rows)


def cost_ref_diff_ci(seed_values: pd.DataFrame, dataset: str, cost_ref: str) -> tuple[float, float, float]:
    if dataset == "MOOCCube" and cost_ref == "CGRC":
        group = seed_values[seed_values["dataset"].eq(dataset)]
        mean, _std, low, high = seed_ci(group["diff_N@10"].astype(float).tolist())
        return mean, low, high

    if dataset == "Junyi" and cost_ref == "CGRC":
        ours_by_seed = seed_values[seed_values["dataset"].eq("Junyi")].set_index("seed")["ours_N@10"]
        cgrc_per_item = {
            2025: ROOT
            / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2025/cgrc_paper_compare_strictfix/per_item_full_cold_cgrc_paper_static.csv",
            2026: ROOT
            / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2026/cgrc_paper_compare_strictfix/per_item_full_cold_cgrc_paper_static.csv",
            2027: ROOT
            / "outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_2027/cgrc_paper_compare_strictfix/per_item_full_cold_cgrc_paper_static.csv",
        }
        diffs = []
        for seed in SEEDS:
            base_df = read_csv(cgrc_per_item[seed])
            base_n10 = float(base_df["N@10"].mean())
            diffs.append(float(ours_by_seed.loc[seed]) - base_n10)
        mean, _std, low, high = seed_ci(diffs)
        return mean, low, high

    if dataset == "COCO" and cost_ref == "CGRC":
        ours_by_seed = seed_values[seed_values["dataset"].eq("COCO")].set_index("seed")["ours_N@10"]
        diffs = []
        for seed in SEEDS:
            result_path = (
                ROOT
                / f"outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}/main_table_compare/cgrc_paper_static_result.json"
            )
            cgrc_result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            base_n10 = float(cgrc_result["full_cold_item_macro"]["N@10"])
            diffs.append(float(ours_by_seed.loc[seed]) - base_n10)
        mean, _std, low, high = seed_ci(diffs)
        return mean, low, high

    return math.nan, math.nan, math.nan


def summarize_cost(seed_ci_table: pd.DataFrame, seed_values: pd.DataFrame) -> pd.DataFrame:
    train_specs = {
        ("MOOCCube", "CKG-RL"): [
            ROOT
            / f"outputs/significance_per_item_exports/mooccube/ckg_rl_full/strict_item_cold_balanced_thr1_seed_{seed}/run.log"
            for seed in SEEDS
        ],
        ("Junyi", "CKG-RL"): [
            ROOT / f"outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_{seed}/run.log"
            for seed in [2026, 2027]
        ],
        ("COCO", "CKG-RL"): [
            ROOT / f"outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}/run.log"
            for seed in SEEDS
        ],
        ("MOOCCube", "CGRC"): [
            ROOT
            / f"outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_{seed}/runtime_cgrc_profile/run.log"
            for seed in SEEDS
        ],
        ("Junyi", "CGRC"): [
            ROOT
            / f"outputs/junyi/main_table_3seed/strict_item_cold_balanced_thr1_seed_{seed}/cgrc_runtime_profile/run.log"
            for seed in SEEDS
        ],
        ("COCO", "CGRC"): [
            ROOT
            / f"outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}/cgrc_runtime_profile/run.log"
            for seed in SEEDS
        ]
        + [
            path
            for seed in SEEDS
            for path in sorted(
                (
                    ROOT
                    / f"outputs/coco/single_seed_triage/ours_full/strict_item_cold_balanced_thr1_seed_{seed}/main_table_compare"
                ).glob("run_cgrc_paper*.log")
            )
        ],
    }
    train_summary: dict[tuple[str, str], tuple[float, float, int]] = {}
    for key, paths in train_specs.items():
        times = parse_epoch_times(paths)
        mean, std = mean_std(times)
        train_summary[key] = (mean, std, len(times))

    runtime = load_runtime_profiles()
    rows: list[dict[str, object]] = []
    cost_refs = {"MOOCCube": "CGRC", "Junyi": "CGRC", "COCO": "CGRC"}
    infer_alias = {"CGRC": "CGRC", "CKG-RL": "CKG-RL"}
    for dataset in ["MOOCCube", "Junyi", "COCO"]:
        cost_ref = cost_refs[dataset]
        ckg_train = train_summary.get((dataset, "CKG-RL"), (math.nan, math.nan, 0))
        base_train = train_summary.get((dataset, cost_ref), (math.nan, math.nan, 0))
        ckg_infer = runtime[(runtime["dataset"].eq(dataset)) & (runtime["method"].eq(infer_alias["CKG-RL"]))]
        if dataset == "MOOCCube":
            # The MOOCCube inference audit has a smoke and a full-family file; keep the full-family rows.
            ckg_infer = ckg_infer[ckg_infer["source_file"].str.contains("fullfamily", na=False)]
        base_infer = runtime[(runtime["dataset"].eq(dataset)) & (runtime["method"].eq(infer_alias.get(cost_ref, cost_ref)))]
        ckg_i_mean, ckg_i_std = mean_std(ckg_infer["infer_s"].astype(float).tolist())
        base_i_mean, base_i_std = mean_std(base_infer["infer_s"].astype(float).tolist())
        ci_row = seed_ci_table[seed_ci_table["dataset"].eq(dataset)].iloc[0]
        cost_mean, cost_low, cost_high = cost_ref_diff_ci(seed_values, dataset, cost_ref)
        if dataset == "MOOCCube":
            coverage = (
                "CKG train logs; retained full-family CKG inference profile; "
                "matched CGRC MOOCCube checkpoint/profile unavailable"
            )
        elif dataset == "Junyi":
            coverage = "CKG train logs; CGRC eval-only profiles; CGRC epoch timers unavailable"
        else:
            coverage = "CKG train logs; CGRC eval-only profiles; CGRC epoch timers available"

        rows.append(
            {
                "dataset": dataset,
                "cost_ref": cost_ref,
                "ckg_train_epoch_mean_s": ckg_train[0],
                "ckg_train_epoch_std_s": ckg_train[1],
                "ckg_train_epoch_n": ckg_train[2],
                "baseline_train_epoch_mean_s": base_train[0],
                "baseline_train_epoch_std_s": base_train[1],
                "baseline_train_epoch_n": base_train[2],
                "ckg_infer_mean_s": ckg_i_mean,
                "ckg_infer_std_s": ckg_i_std,
                "baseline_infer_mean_s": base_i_mean,
                "baseline_infer_std_s": base_i_std,
                "cost_ref_diff_N@10_mean": cost_mean,
                "cost_ref_diff_N@10_ci_low": cost_low,
                "cost_ref_diff_N@10_ci_high": cost_high,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def write_latex(
    seed_ci_table: pd.DataFrame,
    per_course: pd.DataFrame,
    significance: pd.DataFrame,
    cost: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append(r"\section{Stability, Per-Course Gain, and Cost Diagnostics}")
    lines.append("")
    lines.append(
        "Table~\\ref{tab:supp-seed-ci} reports seed-level confidence intervals for the main @10 metrics. "
        "Because only three seeds are available, these intervals are descriptive and should be read together "
        "with the paired course diagnostics in Table~\\ref{tab:supp-per-course-gain}."
    )
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Seed-level 95\% confidence intervals for strict course-cold @10 metrics. "
        r"Intervals use three seed-level course-macro scores; \(\Delta\) is CKG-RL minus the baseline.}"
    )
    lines.append(r"\label{tab:supp-seed-ci}")
    lines.append(r"\tablecaptiongap")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.8pt}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Metric & Baseline & CKG-RL [CI] & Baseline [CI] & \(\Delta\) [CI] \\"
    )
    lines.append(r"\midrule")
    for _, row in seed_ci_table.iterrows():
        for metric in METRICS:
            lines.append(
                f"{row['dataset']} & {metric} & {row['baseline']} & "
                f"{ci_cell(row[f'ours_{metric}_mean'], row[f'ours_{metric}_ci_low'], row[f'ours_{metric}_ci_high'])} & "
                f"{ci_cell(row[f'baseline_{metric}_mean'], row[f'baseline_{metric}_ci_low'], row[f'baseline_{metric}_ci_high'])} & "
                f"{ci_cell(row[f'diff_{metric}_mean'], row[f'diff_{metric}_ci_low'], row[f'diff_{metric}_ci_high'])} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")
    lines.append("")

    lines.append(
        "Table~\\ref{tab:supp-per-course-gain} summarizes the per-course gain distribution. "
        "MOOCCube, Junyi, and COCO use the per-course exports corresponding to the reported CKG-RL configuration."
    )
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-course gain analysis on N@10. Units are seed-course pairs; bootstrap CIs resample pairs; "
        r"W/T/L counts wins/ties/losses.}"
    )
    lines.append(r"\label{tab:supp-per-course-gain}")
    lines.append(r"\tablecaptiongap")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.8pt}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Baseline & Pairs & Mean \(\Delta\)N@10 [boot CI] & Median \(\Delta\)N@10 & W/T/L \\"
    )
    lines.append(r"\midrule")
    for _, row in per_course.iterrows():
        wtl = f"{int(row['wins_N@10'])}/{int(row['ties_N@10'])}/{int(row['losses_N@10'])}"
        lines.append(
            f"{row['dataset']} & {row['baseline']} & {int(row['paired_seed_items'])} & "
            f"{ci_cell(row['mean_diff_N@10'], row['ci_low_N@10'], row['ci_high_N@10'])} & "
            f"{f4(row['median_diff_N@10'])} & {wtl} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")
    lines.append("")

    lines.append(
        "Table~\\ref{tab:supp-significance-tests} gives the exact paired tests behind the significance markers. "
        "The primary correction family is the twelve @5/@10 comparisons in the main table. The full-metric Holm "
        "column is more conservative and covers all six cold metrics for the same reference baselines."
    )
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Paired significance tests for strict course-cold metrics. Raw \(p\)-values use two-sided "
        r"Wilcoxon tests on matched seed-course differences; Holm columns correct the @5/@10 main-table family "
        r"and all cold metrics.}"
    )
    lines.append(r"\label{tab:supp-significance-tests}")
    lines.append(r"\tablecaptiongap")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.4pt}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Metric & Baseline & Pairs & Mean \(\Delta\) & W/T/L & \(p_{\mathrm{Holm,@5/@10}}\) & \(p_{\mathrm{Holm,all}}\) \\"
    )
    lines.append(r"\midrule")
    for _, row in significance.iterrows():
        wtl = f"{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}"
        lines.append(
            f"{row['dataset']} & {row['metric']} & {row['baseline']} & {int(row['n_pairs'])} & "
            f"{f4(row['mean_delta'])} & {wtl} & {p_cell(row['holm_p_main_at5_at10'])} & "
            f"{p_cell(row['holm_p_all_metrics'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")
    lines.append("")

    lines.append(
        "Table~\\ref{tab:supp-cost-tradeoff} separates training-loop cost from final full-ranking inference. "
        "The cost reference is CGRC, the closest graph-reconstruction baseline, while strongest-baseline "
        "accuracy comparisons remain in Table~\\ref{tab:supp-seed-ci} and Table~\\ref{tab:supp-significance-tests}. "
        "Missing entries are left as unavailable rather than extrapolated."
    )
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cost and gain analysis against CGRC. Times are mean\(\pm\)std seconds; CGRC is the cost reference. "
        r"``--'' indicates that no matched checkpoint or separable timer was retained.}"
    )
    lines.append(r"\label{tab:supp-cost-tradeoff}")
    lines.append(r"\tablecaptiongap")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{2.4pt}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Cost ref. & CKG train/epoch & CGRC train/epoch & CKG infer. & CGRC infer. & \(\Delta\)N@10 vs CGRC \\"
    )
    lines.append(r"\midrule")
    for _, row in cost.iterrows():
        lines.append(
            f"{row['dataset']} & {row['cost_ref']} & "
            f"{sec_cell(row['ckg_train_epoch_mean_s'], row['ckg_train_epoch_std_s'])} & "
            f"{sec_cell(row['baseline_train_epoch_mean_s'], row['baseline_train_epoch_std_s'])} & "
            f"{sec_cell(row['ckg_infer_mean_s'], row['ckg_infer_std_s'])} & "
            f"{sec_cell(row['baseline_infer_mean_s'], row['baseline_infer_std_s'])} & "
            f"{ci_cell(row['cost_ref_diff_N@10_mean'], row['cost_ref_diff_N@10_ci_low'], row['cost_ref_diff_N@10_ci_high'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")
    lines.append("")
    lines.append(
        r"\noindent\footnotesize "
        r"MOOCCube CGRC latency is not inferred from accuracy logs because CGRC inference includes "
        r"test-time graph reconstruction and propagation. Junyi and COCO CGRC inference use retained "
        r"checkpoints and eval-only profiling; CGRC training-loop timing is complete only for COCO. "
        r"\normalsize"
    )
    lines.append("")

    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seed_values = load_main_seed_values()
    seed_ci_table = summarize_seed_ci(seed_values)
    per_course_pairs = load_per_course_pairs()
    per_course_summary = summarize_per_course(per_course_pairs)
    significance = summarize_significance(per_course_pairs)
    cost = summarize_cost(seed_ci_table, seed_values)

    seed_values.to_csv(FIG_DIR / "revision_seed_level_values.csv", index=False)
    seed_ci_table.to_csv(FIG_DIR / "revision_seed_ci_summary.csv", index=False)
    per_course_pairs.to_csv(FIG_DIR / "revision_per_course_gain_pairs.csv", index=False)
    per_course_summary.to_csv(FIG_DIR / "revision_per_course_gain_summary.csv", index=False)
    significance.to_csv(FIG_DIR / "significance_tests.csv", index=False)
    cost.to_csv(FIG_DIR / "revision_cost_tradeoff_summary.csv", index=False)
    write_latex(seed_ci_table, per_course_summary, significance, cost)

    print(f"Wrote {FIG_DIR / 'revision_seed_ci_summary.csv'}")
    print(f"Wrote {FIG_DIR / 'revision_per_course_gain_summary.csv'}")
    print(f"Wrote {FIG_DIR / 'significance_tests.csv'}")
    print(f"Wrote {FIG_DIR / 'revision_cost_tradeoff_summary.csv'}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
