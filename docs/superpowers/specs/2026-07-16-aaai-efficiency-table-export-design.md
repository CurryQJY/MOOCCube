# AAAI Efficiency Table Export Design

## Goal

Export the current CKG-RL versus CGRC efficiency analysis as one independent publication table and one standalone PDF preview without changing the underlying measurements.

## Outputs

- `paper_aaai27/efficiency_table_aaai.tex`: a LaTeX `table*` fragment that can be included with `\input{efficiency_table_aaai}`.
- `paper_aaai27/efficiency_table_aaai_standalone.tex`: a generated standalone wrapper used only to build the preview.
- `paper_aaai27/efficiency_table_aaai.pdf`: a tightly cropped PDF containing the single table.

## Architecture

The existing `build_revision_tables.py` remains the source of truth for loading logs and calculating the cost summary. A pure renderer will turn that summary DataFrame into either the reusable table fragment or the standalone preview document. A dedicated `export_efficiency_table.py` entry point will calculate only the data needed for the efficiency table, write both TeX files, and invoke `latexmk` to build the PDF.

## Data And Presentation Rules

- Preserve all current measured values and confidence intervals exactly.
- Render unavailable CGRC measurements as `--`.
- Keep CGRC as the cost reference.
- Include only the efficiency table, caption, label, and coverage note; do not include the other revision tables.
- Fail clearly if LaTeX compilation fails instead of leaving a newly reported successful export.

## Verification

- Unit-test the renderer with a small in-memory DataFrame.
- Verify the generated fragment contains exactly one efficiency table and no unrelated revision sections.
- Run the dedicated exporter and confirm the TeX and PDF outputs exist.
- Scan the LaTeX log for errors and confirm the PDF has one page.
