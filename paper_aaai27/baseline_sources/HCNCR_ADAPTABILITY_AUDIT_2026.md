# HCNCR Adaptability Audit

Status: CONDITIONAL GO pending full text and official code acquisition.

This audit concerns *Hypergraph Convolutional Networks for Course
Recommendation in MOOCs* (HCNCR), IEEE Transactions on Knowledge and Data
Engineering, 2025. DOI: https://doi.org/10.1109/TKDE.2025.3568709.

## Previous Work and Reproducibility

No HCNCR repository, preprint, data adapter, or experiment was found in the
workspace. Public metadata exposes the abstract but not a public full-text
location, and no official code was verified during this audit.

### Official-Code Search Update (2026-07-13)

- Exact-title, `HCNCR`, and DOI (`10.1109/TKDE.2025.3568709`) searches found
  no matching public repository or code reference on GitHub, Gitee, or GitLab.
- The Crossref record provides only the publisher PDF link; it has no code,
  dataset, supplementary-material, or repository relation.
- The verified GitHub account of coauthor Zhonghua Yan (CCNU),
  `https://github.com/ccnuyan`, has no HCNCR or course-recommendation
  repository. Other same-name accounts were not attributed without an
  affiliation match.

Therefore, HCNCR remains paper-only. Do not treat similarly named hypergraph
repositories as an official implementation.

## Why It Is the Best Remaining Lead

The abstract describes a direct course recommender that builds a course
hypergraph from course attributes and a learner hypergraph from learner
similarities, then integrates both with a learner-course bipartite graph. It
states that the framework learns representations for both learners and courses.
This is the required high-level property for strict cold-course ranking: a cold
course can potentially be represented through static course attributes instead
of an interaction-ID lookup table.

## Current Data Fit

- MOOCCube provides course name/about metadata, course-concept, course-video,
  teacher-course, and school-course relations.
- Junyi provides course name/about metadata, content embeddings, and
  course-concept relations.
- COCO provides course metadata/content embeddings and course-concept/category
  information.

Thus all three datasets can support an attribute-side course hypergraph. The
learner hypergraph must be built only from strict training interactions; no
validation/test interaction may contribute to learner similarity.

## Required Gates Before a Run

1. Confirm from the full method or source that the final scorer uses the
   hypergraph-derived course representation for every candidate, including cold
   courses, rather than an ID-only BPR/classification row.
2. Build course hyperedges only from permitted static metadata and verify every
   strict cold course has a nonempty attribute/concept hyperedge.
3. Construct learner similarity from train interactions only and sample it
   sparsely enough for the 12 GB GPU.
4. Replace any sampled evaluator with external full-catalog ranking, train
   history masking, and item-macro Recall/NDCG.
5. Verify that any teacher-dependent component has a documented fallback on
   Junyi and COCO, whose current processed schemas do not preserve teacher
   nodes.

## Decision

HCNCR is the highest-priority future feasibility target, but it is not yet a
runnable baseline. Do not implement from the abstract or start a GPU run.
Obtain official code or an author-approved reproducible implementation first.
