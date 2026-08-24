from __future__ import annotations

import csv
import re
import statistics
import subprocess
from pathlib import Path

try:
    from .build_revision_tables import PAPER, ROOT, read_text_auto
except ImportError:
    from build_revision_tables import PAPER, ROOT, read_text_auto  # type: ignore[no-redef]


SUMMARY_TIMER_PATTERN = re.compile(
    r"\[(?:STATIC|CGRC)-TRAIN\] Epoch (\d+)(?:/\d+)? .*?Time:\s*([0-9.]+)s"
)
PROGRESS_TIMER_PATTERN = re.compile(
    r"\[CGRC-TRAIN-PROGRESS\] Epoch (\d+)/\d+ \| "
    r"(\d+)/(\d+) .*?elapsed=([0-9hms]+)"
)
DURATION_PATTERN = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")
WINDOW_START = 5
WINDOW_END = 15
LATEX_LINE_BREAK = chr(92) * 2


def duration_to_seconds(value: str) -> float:
    match = DURATION_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Unsupported duration: {value}")
    return float(
        int(match.group(1) or 0) * 3600
        + int(match.group(2) or 0) * 60
        + int(match.group(3) or 0)
    )


def parse_indexed_epoch_times(paths: list[Path]) -> dict[int, float]:
    """Extract one timer per epoch, preferring the precise summary timer."""
    epoch_times: dict[int, float] = {}
    for path in paths:
        for line in read_text_auto(path).splitlines():
            summary = SUMMARY_TIMER_PATTERN.search(line)
            if summary:
                epoch_times[int(summary.group(1))] = float(summary.group(2))
                continue
            progress = PROGRESS_TIMER_PATTERN.search(line)
            if progress and int(progress.group(2)) == int(progress.group(3)):
                epoch_times.setdefault(
                    int(progress.group(1)), duration_to_seconds(progress.group(4))
                )
    return epoch_times


def summarize_seed_windows(
    seed_paths: dict[int, list[Path]],
    window_start: int = WINDOW_START,
    window_end: int = WINDOW_END,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if window_start > window_end:
        raise ValueError("window_start must be no greater than window_end")

    window_epochs = list(range(window_start, window_end + 1))
    details: list[dict[str, object]] = []
    seed_means: list[float] = []
    for seed, paths in sorted(seed_paths.items()):
        combined: dict[int, float] = {}
        for path in paths:
            for epoch, value in parse_indexed_epoch_times([path]).items():
                if epoch in combined:
                    raise ValueError(f"Duplicate epoch {epoch} for seed {seed}: {path}")
                combined[epoch] = value
        missing = [epoch for epoch in window_epochs if epoch not in combined]
        if missing:
            raise ValueError(f"Missing epochs {missing} for seed {seed}")
        window_times = [combined[epoch] for epoch in window_epochs]
        seed_mean = statistics.mean(window_times)
        seed_means.append(seed_mean)
        details.append(
            {
                "seed": seed,
                "window_start_epoch": window_start,
                "window_end_epoch": window_end,
                "window_epoch_count": len(window_epochs),
                "mean_train_time_s_per_epoch": seed_mean,
                "std_train_time_s_per_epoch_within_window": statistics.stdev(
                    window_times
                )
                if len(window_times) > 1
                else 0.0,
                "source_files": ";".join(str(path) for path in paths),
            }
        )
    if not seed_means:
        raise ValueError("At least one retained seed is required")
    return details, {
        "window_start_epoch": window_start,
        "window_end_epoch": window_end,
        "seed_count": len(seed_means),
        "mean_train_time_s_per_epoch": statistics.mean(seed_means),
        "std_train_time_s_per_epoch_across_seed_means": statistics.stdev(seed_means)
        if len(seed_means) > 1
        else 0.0,
    }


def relative(path: str) -> Path:
    return ROOT / path


def measured_sources() -> list[tuple[str, str, dict[int, list[Path]]]]:
    return [
        (
            "MOOCCube",
            "CKG-RL",
            {
                2025: [
                    relative(
                        "outputs/significance_per_item_exports/mooccube/ckg_rl_full/"
                        "strict_item_cold_balanced_thr1_seed_2025/run.log"
                    )
                ],
                2026: [
                    relative(
                        "outputs/significance_per_item_exports/mooccube/ckg_rl_full/"
                        "strict_item_cold_balanced_thr1_seed_2026/run.log"
                    )
                ],
                2027: [
                    relative(
                        "outputs/significance_per_item_exports/mooccube/ckg_rl_full/"
                        "strict_item_cold_balanced_thr1_seed_2027/run.log"
                    )
                ],
            },
        ),
        (
            "MOOCCube",
            "CGRC",
            {
                2026: [
                    relative(
                        "background_logs/p1_cgrc_main_table_reproduction/"
                        "seed_2026.stdout.log"
                    ),
                    relative(
                        "background_logs/p1_cgrc_main_table_reproduction/"
                        "seed_2026_resume_epoch11.stdout.log"
                    ),
                    relative(
                        "background_logs/p1_cgrc_main_table_reproduction/"
                        "resumed_after_recommended_queue/seed_2026.log"
                    ),
                ],
                2027: [
                    relative(
                        "background_logs/p1_cgrc_main_table_reproduction/"
                        "resumed_after_recommended_queue/seed_2027.log"
                    )
                ],
            },
        ),
        (
            "Junyi",
            "CKG-RL",
            {
                2026: [
                    relative(
                        "outputs/junyi/main_table_3seed/"
                        "strict_item_cold_balanced_thr1_seed_2026/run.log"
                    )
                ],
                2027: [
                    relative(
                        "outputs/junyi/main_table_3seed/"
                        "strict_item_cold_balanced_thr1_seed_2027/run.log"
                    )
                ],
            },
        ),
        (
            "COCO",
            "CKG-RL",
            {
                2025: [
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2025/run.log"
                    )
                ],
                2026: [
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2026/run.log"
                    )
                ],
                2027: [
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2027/run.log"
                    )
                ],
            },
        ),
        (
            "COCO",
            "CGRC",
            {
                2025: [
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2025/main_table_compare/"
                        "run_cgrc_paper_before_resume_20260614_204215.log"
                    ),
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2025/main_table_compare/"
                        "run_cgrc_paper_before_resume_20260614_213856.log"
                    ),
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2025/main_table_compare/"
                        "run_cgrc_paper_before_resume_20260615_003749.log"
                    ),
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2025/main_table_compare/"
                        "run_cgrc_paper.log"
                    ),
                ],
                2026: [
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2026/main_table_compare/"
                        "run_cgrc_paper.log"
                    )
                ],
                2027: [
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2027/main_table_compare/"
                        "run_cgrc_paper_before_resume_20260616_012901.log"
                    ),
                    relative(
                        "outputs/coco/single_seed_triage/ours_full/"
                        "strict_item_cold_balanced_thr1_seed_2027/main_table_compare/"
                        "run_cgrc_paper.log"
                    ),
                ],
            },
        ),
    ]


def collect_export_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    details: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for dataset, method, sources in measured_sources():
        method_details, method_summary = summarize_seed_windows(sources)
        for row in method_details:
            row["dataset"] = dataset
            row["method"] = method
            row["source_files"] = ";".join(
                str(Path(path).relative_to(ROOT)).replace("\\", "/")
                for path in str(row["source_files"]).split(";")
            )
        method_summary.update(
            {"dataset": dataset, "method": method, "status": "measured"}
        )
        details.extend(method_details)
        summary.append(method_summary)
    summary.append(
        {
            "dataset": "Junyi",
            "method": "CGRC",
            "window_start_epoch": WINDOW_START,
            "window_end_epoch": WINDOW_END,
            "seed_count": 0,
            "mean_train_time_s_per_epoch": None,
            "std_train_time_s_per_epoch_across_seed_means": None,
            "status": "unavailable: no completed CGRC training profile",
        }
    )
    dataset_order = {"MOOCCube": 0, "Junyi": 1, "COCO": 2}
    method_order = {"CKG-RL": 0, "CGRC": 1}
    details.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order[str(row["method"])],
            int(row["seed"]),
        )
    )
    summary.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order[str(row["method"])],
        )
    )
    return details, summary


def time_cell(row: dict[str, object]) -> str:
    mean = row["mean_train_time_s_per_epoch"]
    std = row["std_train_time_s_per_epoch_across_seed_means"]
    if mean is None or std is None:
        return "--"
    return f"{float(mean):.1f}\\,$\\pm$\\,{float(std):.1f}"


def seed_count_cell(row: dict[str, object]) -> str:
    seed_count = int(row["seed_count"])
    return str(seed_count) if seed_count else "--"


def table_rows(summary: list[dict[str, object]]) -> list[str]:
    return [
        f"{row['dataset']} & {row['method']} & 5--15 & "
        f"{seed_count_cell(row)} & {time_cell(row)} {LATEX_LINE_BREAK}"
        for row in summary
    ]


def table_note() -> str:
    return (
        r"Each seed contributes its mean training-loop time over epochs 5--15; "
        r"values are mean\(\pm\) sample std across retained seed means. "
        r"This is a fixed post-warm-up diagnostic, not total training time. "
        r"Post-epoch validation is excluded. Incomplete CGRC profiles are excluded; "
        r"Junyi CGRC has no completed training profile and is not projected."
    )


def render_fragment(summary: list[dict[str, object]]) -> str:
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Fixed-window training time per epoch.}",
            r"\label{tab:stable-epoch-timing}",
            r"\tablecaptiongap",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc}",
            r"\toprule",
            r"Dataset & Method & Epoch window & Seeds & Train/epoch (s) "
            + LATEX_LINE_BREAK,
            r"\midrule",
            *table_rows(summary),
            r"\bottomrule",
            r"\end{tabular*}",
            r"\vspace{2pt}",
            r"\parbox{\textwidth}{\footnotesize " + table_note() + r"}",
            r"\end{table*}",
            "",
        ]
    )


def render_standalone(summary: list[dict[str, object]]) -> str:
    return "\n".join(
        [
            r"\documentclass[varwidth=15cm,border=8pt]{standalone}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{booktabs}",
            r"\usepackage{amsmath}",
            r"\usepackage{caption}",
            r"\captionsetup{justification=centering,singlelinecheck=false}",
            r"\begin{document}",
            r"\begin{minipage}{14cm}",
            r"\centering",
            r"{",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccc}",
            r"\toprule",
            r"Dataset & Method & Epoch window & Seeds & Train/epoch (s) "
            + LATEX_LINE_BREAK,
            r"\midrule",
            *table_rows(summary),
            r"\bottomrule",
            r"\end{tabular*}",
            r"}",
            r"\captionof{table}{Fixed-window training time per epoch.}",
            r"\label{tab:stable-epoch-timing-preview}",
            r"\vspace{2pt}",
            r"\parbox{\linewidth}{\footnotesize " + table_note() + r"}",
            r"\end{minipage}",
            r"\end{document}",
            "",
        ]
    )


def write_exports(output_dir: Path = PAPER) -> tuple[Path, Path, Path, Path]:
    details, summary = collect_export_rows()
    summary_path = output_dir / "stable_epoch_timing_5_15_summary.csv"
    detail_path = output_dir / "stable_epoch_timing_5_15_seed_detail.csv"
    fragment_path = output_dir / "stable_epoch_timing_5_15.tex"
    standalone_path = output_dir / "stable_epoch_timing_5_15_standalone.tex"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    fragment_path.write_text(render_fragment(summary), encoding="utf-8")
    standalone_path.write_text(render_standalone(summary), encoding="utf-8")
    return summary_path, detail_path, fragment_path, standalone_path


def latexmk_command(standalone_path: Path) -> list[str]:
    return [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        standalone_path.name,
    ]


def cleanup_latex_intermediates(standalone_path: Path) -> None:
    for suffix in (".aux", ".fls", ".fdb_latexmk"):
        standalone_path.with_suffix(suffix).unlink(missing_ok=True)


def main() -> None:
    summary_path, detail_path, fragment_path, standalone_path = write_exports()
    subprocess.run(latexmk_command(standalone_path), cwd=PAPER, check=True)
    output_pdf = PAPER / "stable_epoch_timing_5_15.pdf"
    standalone_path.with_suffix(".pdf").replace(output_pdf)
    cleanup_latex_intermediates(standalone_path)
    for path in (summary_path, detail_path, fragment_path, standalone_path, output_pdf):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
