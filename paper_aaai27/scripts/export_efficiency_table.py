from __future__ import annotations

import subprocess
from pathlib import Path

try:
    from .build_revision_tables import (
        PAPER,
        load_main_seed_values,
        summarize_cost,
        summarize_seed_ci,
        write_efficiency_tex_exports,
    )
except ImportError:
    from build_revision_tables import (  # type: ignore[no-redef]
        PAPER,
        load_main_seed_values,
        summarize_cost,
        summarize_seed_ci,
        write_efficiency_tex_exports,
    )


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
    seed_values = load_main_seed_values()
    seed_ci_table = summarize_seed_ci(seed_values)
    cost = summarize_cost(seed_ci_table, seed_values)
    fragment_path, standalone_path = write_efficiency_tex_exports(cost, PAPER)

    subprocess.run(latexmk_command(standalone_path), cwd=PAPER, check=True)
    compiled_pdf = standalone_path.with_suffix(".pdf")
    output_pdf = PAPER / "efficiency_table_aaai.pdf"
    compiled_pdf.replace(output_pdf)
    cleanup_latex_intermediates(standalone_path)

    print(f"Wrote {fragment_path}")
    print(f"Wrote {standalone_path}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
