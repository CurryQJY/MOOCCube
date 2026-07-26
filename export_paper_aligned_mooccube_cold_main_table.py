from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_TEX = ROOT / "paper_wsdm" / "main.tex"
OUT_DIR = (
    ROOT
    / "outputs"
    / "content_delta_pop5"
    / "course_ablation_e60_3seed"
    / "full"
    / "paper_aligned_main_table_export"
)

LABEL = r"\label{tab:mooccube-cold}"
METRICS = ["R@5", "R@10", "R@20", "N@5", "N@10", "N@20"]


def extract_table_block(text: str) -> str:
    label_pos = text.find(LABEL)
    if label_pos < 0:
        raise ValueError(f"Missing {LABEL} in {MAIN_TEX}")
    start = text.rfind(r"\begin{table*}", 0, label_pos)
    end = text.find(r"\end{table*}", label_pos)
    if start < 0 or end < 0:
        raise ValueError("Could not locate table* boundaries around tab:mooccube-cold")
    end += len(r"\end{table*}")
    return text[start:end]


def plain_method(cell: str) -> str:
    cell = cell.strip()
    cell = cell.replace(r"\textbf{CKG-RL}", "CKG-RL")
    cell = re.sub(r"\\sigmark\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", cell)
    cell = re.sub(r"~?\\cite\{[^{}]+\}", "", cell)
    cell = cell.replace(r"\textbf{", "").replace("}", "")
    return cell.strip()


def values_from_cell(cell: str) -> tuple[str, str, str]:
    numbers = re.findall(r"\d+\.\d+", cell)
    if len(numbers) < 2:
        raise ValueError(f"Could not parse mean/std from cell: {cell}")
    mean, std = numbers[0], numbers[1]
    return mean, std, f"{mean} +/- {std}"


def extract_rows(table_block: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    row_block = table_block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    formatted_rows: list[dict[str, str]] = []
    numeric_rows: list[dict[str, str]] = []
    for raw_line in row_block.splitlines():
        line = raw_line.strip()
        if not line or not line.endswith(r"\\"):
            continue
        line = line[:-2].strip()
        cells = [part.strip() for part in line.split("&")]
        if len(cells) != 7:
            raise ValueError(f"Expected 7 cells, got {len(cells)}: {raw_line}")
        method = plain_method(cells[0])
        formatted: dict[str, str] = {"Method": method}
        numeric: dict[str, str] = {"Method": method}
        for metric, cell in zip(METRICS, cells[1:]):
            mean, std, combined = values_from_cell(cell)
            formatted[metric] = combined
            numeric[f"{metric}_mean"] = mean
            numeric[f"{metric}_std"] = std
        formatted_rows.append(formatted)
        numeric_rows.append(numeric)
    return formatted_rows, numeric_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standalone_tex(source_tex_name: str) -> str:
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage{{array}}
\usepackage{{amsmath}}
\pagestyle{{empty}}
\newcommand{{\sd}}[1]{{\scriptsize{{$\pm$#1}}}}
\newcommand{{\sigmark}}[1]{{\textsuperscript{{#1}}}}
\renewcommand{{\cite}}[1]{{}}
\begin{{document}}
\input{{{source_tex_name}}}
\end{{document}}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_block = extract_table_block(MAIN_TEX.read_text(encoding="utf-8"))
    formatted_rows, numeric_rows = extract_rows(table_block)

    source_tex = OUT_DIR / "paper_aligned_mooccube_cold_main_table_source.tex"
    standalone = OUT_DIR / "paper_aligned_mooccube_cold_main_table_standalone.tex"
    formatted_csv = OUT_DIR / "paper_aligned_mooccube_cold_main_table.csv"
    numeric_csv = OUT_DIR / "paper_aligned_mooccube_cold_main_table_numeric.csv"

    source_tex.write_text(table_block + "\n", encoding="utf-8")
    standalone.write_text(standalone_tex(source_tex.name), encoding="utf-8")
    write_csv(formatted_csv, formatted_rows)
    write_csv(numeric_csv, numeric_rows)

    print(f"Wrote {formatted_csv}")
    print(f"Wrote {numeric_csv}")
    print(f"Wrote {source_tex}")
    print(f"Wrote {standalone}")


if __name__ == "__main__":
    main()
