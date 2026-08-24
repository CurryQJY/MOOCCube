# MSEC-Rec Adaptability Audit

Status: NO-GO for the strict item-cold main table without a material model
rewrite.

This audit concerns *Meta-path Sampling-Enhanced Course Recommendation in
Heterogeneous Networks* (MSEC-Rec), Information Processing & Management,
article 104482, 2026. DOI: https://doi.org/10.1016/j.ipm.2025.104482.

## Previous Work

The official repository was already staged at
`paper_aaai27/baseline_sources/MSEC-Rec`, but no strict data adapter,
feasibility run, report, or table entry had been created before this audit.

## Positive Fit

MSEC-Rec directly recommends courses from user-course interactions. Its source
heterograph uses U-C, U-V, C-K, C-V, and V-K relations, which is a close match
to the available MOOCCube relation schema.

## Strict Item-Cold Blocker

The released HAN does not produce course embeddings. All configured meta-paths
start and end at `user`, so the HAN outputs only user representations. The
course candidates used by the BPR scorer are a separate `nn.Embedding` lookup
table. A cold course has no training positive and receives no message from the
course-video or course-concept graph into its candidate representation.

Thus, the release cannot score a strict cold course through side information.
Replacing the lookup table with a graph-derived course encoder changes the
central scorer and is a material model redesign, not an adapter.

## Additional Protocol and Runtime Problems

- `utils.py` samples BPR negatives uniformly from every course ID, including
  strict cold courses. This can be repaired with warm-only sampling, but is not
  sufficient to solve the cold-item representation problem.
- The supplied HR/NDCG/MRR evaluator ranks each positive against 99 random
  negatives. It needs an external full-catalog, train-history-masked,
  item-macro evaluator.
- `HANLayer` has `rw_num_traces = 2000` and repeats every user before random
  walks. At 199,199 strict MOOCCube users this is 398,398,000 walks per
  meta-path; the release configures four meta-paths. The sampled endpoint lists
  are built as Python lists, making the released path infeasible on the current
  12 GB GPU.
- The release hard-codes all four U-C-U, U-V-U, U-C-K-C-U, and U-V-K-V-U paths.
  Junyi and COCO lack user-video, course-video, and video-concept relations, so
  they cannot run the published path set.
- The active `zw` environment has CUDA PyTorch but no `dgl` installation.

## Decision

Do not install DGL or start a MSEC-Rec GPU smoke under the released design.
It is a directly relevant course-recommendation paper, but it cannot meet the
strict cold-course and three-dataset requirements without replacing its item
encoder, path sampler, and evaluation pipeline. Such a result would need a new
method-level justification rather than the `MSEC-Rec (adapted)` label.
