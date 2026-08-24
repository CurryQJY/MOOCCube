# TDInit Results Table Design

## Goal

Export publication-ready comparisons between CKG-RL and CKG-RL+TDInit, where TDInit denotes the current CBI experiment.

## Data and metric contract

- Baseline: `outputs/significance_per_item_exports/mooccube/ckg_rl_full/fast3_static_runs_detail.csv`, the three-seed source matching the latest MOOCCube CKG-RL row in `paper_aaai27/main.tex`.
- TDInit: `outputs/cbi_anchor_sim_3seed_serial/fast3_static_runs_detail.csv`.
- Report the three-seed mean of item-macro R@5, R@10, R@20, N@5, N@10, and N@20.
- Cold and Hot use the stored item-macro metrics directly.
- Overall is computed within each seed by weighting Cold and Hot item-macro metrics by their evaluated course counts, then averaging the three seed-level Overall values.
- Improvement is `(TDInit / CKG-RL - 1) * 100%` and retains negative signs.

## Layout and outputs

- Match the supplied reference: Method column, six metric columns, two method rows, and an italic improvement row.
- Produce one vertically stacked three-panel table plus separate Cold, Hot, and Overall tables.
- Bold the better value in each metric column rather than always bolding TDInit.
- Export PNG, PDF, LaTeX, and CSV under `paper_aaai27/figures/ckg_rl_tdinit_3seed_latest/`, kept separate from earlier single-seed or obsolete comparisons.
- Preserve reproducibility with a standalone generator and automated tests; do not modify existing main-table code.
