# Overall Baseline Comparison Export Design

## Goal

Export a reproducible comparison of all currently eligible main-table baselines under the same overall course-macro definition. Include CKG-RL as the final comparison row, retain three-seed detail, and expose coverage gaps without estimating missing results.

## Scope

The export covers MOOCCube, Junyi, and COCO with seeds 2025, 2026, and 2027. It includes the ten baselines with complete cold and hot target evidence:

- Popularity
- BPR
- DropoutNet
- LightGCN
- CCFCRec
- ALDI
- KGRec
- CGRC
- USIM
- SEMCo

CKG-RL is included as the final row. PCGNN is excluded from numeric comparison because its retained main-table reports contain cold-target metrics but no warm-target metrics or warm-target course count. PCGNN remains visible in the coverage sheet as unavailable.

## Metric Definition

The comparison uses Overall Item-Macro metrics, consistent with the current main table's course-macro evaluation unit. For each seed and metric:

\[
M_{\mathrm{overall}}=
\frac{n_{\mathrm{cold}}M_{\mathrm{cold}}+n_{\mathrm{hot}}M_{\mathrm{hot}}}
{n_{\mathrm{cold}}+n_{\mathrm{hot}}},
\]

where the counts are numbers of evaluated target courses, not interaction rows. When a source already provides `full_all_item_macro`, the exporter uses it after verifying consistency with the cold/hot weighted reconstruction when both forms are available. Cold and hot means are never averaged without their course counts.

The exported metrics are Recall and NDCG at K in `{5, 10, 20}`.

## Source Selection

The exporter reuses the current main-table artifact mappings rather than searching for the newest file by timestamp.

- Core baselines and CKG-RL reuse the source candidates defined by `paper_aaai27/scripts/audit_significance_inputs.py` and `paper_aaai27/scripts/export_warm_target_table.py`.
- SEMCo uses the official-adaptation directories already selected by the warm-target exporter.
- KGRec uses the report paths locked in `paper_aaai27/baseline_sources/_kgrec_strict/_remaining_main_table_queue/main_table_summary.json` and reads `test.full_all_item_macro` plus test counts.
- PCGNN coverage is read from the retained strict adapter reports and must remain unavailable when `count_full_hot_item_macro` is zero or `full_hot_item_macro` is empty.

Every exported seed row records its cold source, hot source, aggregation route, cold count, hot count, and status.

## Outputs

The exporter writes the following files under `paper_aaai27/figures/overall_baseline_comparison/`:

1. `overall_baseline_comparison.xlsx`
   - `Summary`: dataset-method three-seed means and sample standard deviations, ranks, strongest-baseline values, and CKG-RL relative improvements.
   - `Seed_Detail`: one row per dataset, method, and seed with all six metrics and provenance fields.
   - `Coverage`: all expected dataset-method-seed combinations, including unavailable PCGNN rows and explicit reasons.
2. `overall_baseline_summary.csv`: long-form three-seed summary.
3. `overall_baseline_seed_detail.csv`: long-form seed-level values.
4. `overall_baseline_wide.csv`: paper-friendly wide table for R@5, R@10, N@5, and N@10 across the three datasets.
5. `overall_baseline_coverage.csv`: coverage and provenance audit.

The workbook uses a professional font, fixed four-decimal metric display, frozen headers, filters, and no formulas. CSV files retain full numerical precision.

## Comparison Rules

- Rank methods within each dataset and metric in descending order.
- The strongest baseline excludes CKG-RL and unavailable methods.
- CKG-RL relative improvement is `(CKG-RL - strongest baseline) / strongest baseline` when the reference is positive.
- No significance markers are generated because the export is descriptive and does not add new paired statistical tests.
- PCGNN cells are blank and marked `unavailable_missing_warm_targets`; no zero or projected value is inserted.

## Error Handling

The export fails with a clear error when:

- an eligible method lacks any of the three required seeds;
- a required metric or course count is missing or non-finite;
- cold and hot target counts are not positive;
- a direct overall value disagrees with the weighted reconstruction beyond tolerance;
- a source resolves to a non-main-table artifact;
- duplicate dataset-method-seed rows are found.

PCGNN's known warm-target absence is a coverage result, not a fatal error.

## Testing

Tests will be written before implementation and will cover:

1. exact count-weighted overall reconstruction;
2. rejection of an unweighted cold/hot average;
3. direct-overall versus reconstructed-overall consistency checks;
4. complete three-seed aggregation with sample standard deviation;
5. ranking and strongest-baseline relative improvement;
6. PCGNN missing-warm coverage without fabricated metrics;
7. required workbook sheets and CSV schemas;
8. end-to-end export against the current artifact tree.

## Acceptance Criteria

- Ten baselines plus CKG-RL have complete overall values for all three datasets and all three seeds.
- PCGNN is the only unavailable main-table baseline and is explicitly documented.
- All six metrics are present in the long-form outputs; the wide comparison contains the four main-table metrics.
- Workbook and CSV summaries agree numerically.
- Recomputed cold/hot weighted values match any available direct overall values within tolerance.
- Focused tests and the end-to-end exporter finish with exit code zero.
