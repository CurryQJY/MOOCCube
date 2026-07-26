# KnowPath Adaptability Audit

Status: CONDITIONAL GO for MOOCCube only; NO-GO for a three-dataset main-table
run until source code and data provenance are verified.

This audit concerns *KnowPath: An LLM-Supported Knowledge Graph Construction
and Path Finding Framework to Explainable MOOC Recommendations*, ACM
Transactions on Information Systems, 2026. DOI: https://doi.org/10.1145/3779436.

## Previous Work and Reproducibility

No KnowPath repository, preprint, adapter, or experiment was found in the
workspace. Public metadata describes XueTang and COCO experiments, but public
code and a reproducible full-text source were not verified during this audit.

### Official-Code Search Update (2026-07-13)

- Exact-title, `KnowPath` + MOOC, and DOI (`10.1145/3779436`) searches found
  no matching public repository or code reference on GitHub, Gitee, or GitLab.
- The Crossref record provides only the ACM publisher PDF link; it has no
  code, dataset, supplementary-material, or repository relation.
- The GitHub account of coauthor Zhangze Chen that identifies Zhejiang Normal
  University, `https://github.com/chenzhangze-web`, has no KnowPath repository.
  The unrelated `tize-72/KnowPath-arXiv` project must not be cited or adapted.

Therefore, KnowPath remains paper-only. Do not begin an implementation or GPU
run unless the authors release an implementation or explicitly approve one.

## Potential Fit

KnowPath is a direct MOOC course recommender. Its reported LLM-built knowledge
graph over learners, instructors, and educational resources could provide
metadata-side paths to cold courses, unlike an ID-only collaborative filter.
MOOCCube has course descriptions, concepts, videos, teachers, schools, and
time-stamped user activities, so it is the only current dataset with a close
enough schema for a constrained feasibility study.

## Strict Protocol Risks

- Every LLM-generated KG edge must be built once from permitted static metadata
  and stored with model, prompt, date, and input hashes. Dynamic web retrieval,
  user review text, or validation/test behavior would be prohibited.
- Reinforcement learning must use strict train interactions and warm-only
  negative/candidate construction. The final evaluation must score the full
  course catalog with train-history masking and item-macro aggregation.
- The current processed COCO provides course metadata, concepts/categories,
  language, level, and audience, but does not preserve the full
  instructor/resource schema implied by KnowPath. It is not yet established
  that it is the identical COCO preprocessing used by the paper.
- Junyi lacks the instructor, video, and resource layers needed for the
  published KG. Removing these components would materially change the method.
- The source scorer must be inspected to establish that it produces scores for
  cold courses from KG paths rather than only trained course IDs.

## Decision

Do not start a KnowPath GPU run. A MOOCCube-only audit may be reconsidered if
official code is released and its LLM/KG construction can be frozen to the
allowed metadata. It is not currently defensible as a shared three-dataset
baseline.
