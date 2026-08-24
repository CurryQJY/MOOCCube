# Recent MOOC Course-Recommendation SOTA Discovery

Search date: 2026-07-13.

This follow-up search covers 2023--2026 MOOC/course recommendation work after
the TOME, BEICL, and MSEC-Rec audits. A paper is not treated as runnable until
both its released code and its strict item-cold compatibility are verified.

## Highest-Priority New Leads

| Priority | Model | Venue | Why it is relevant | Current gate |
|---:|---|---|---|---|
| 1 | HCNCR: *Hypergraph Convolutional Networks for Course Recommendation in MOOCs* | IEEE TKDE 2025 | Direct course recommendation; constructs course hypergraphs from course attributes and learner hypergraphs from learner similarities, then integrates them with the learner-course graph. This is the strongest newly found cold-course-compatible design on paper. | Full text/preprint and official code not found in the workspace or indexed public locations. Request author code or obtain a reproducible manuscript before a feasibility audit. |
| 2 | KnowPath: *An LLM-Supported Knowledge Graph Construction and Path Finding Framework to Explainable MOOC Recommendations* | ACM TOIS 2026 | Direct MOOC course recommendation; reports XueTang and COCO experiments. Its LLM-built KG could in principle provide metadata-side representations for cold courses. | Code not verified. Need to establish whether its reported COCO is the same source as the current processed COCO, freeze all LLM/KG inputs to permitted metadata, and inspect its scorer before running. |

## Supporting Evidence

### HCNCR

- DOI: https://doi.org/10.1109/TKDE.2025.3568709.
- The abstract states that it models higher-order learner-course-teacher
  interactions, builds course hypergraphs from course attributes and learner
  hypergraphs from learner similarities, then combines hypergraph and
  learner-course-bipartite representations.
- IEEE TKDE is a CCF A journal. The paper is recent and its target is exactly
  course recommendation, unlike concept-recommendation models.
- Strict audit focus: verify that a cold course remains in an attribute-only
  course hypergraph, that all learner similarity edges use train evidence only,
  and that the final scorer is a full-catalog course scorer rather than a
  sampled-candidate classifier.

### KnowPath

- DOI: https://doi.org/10.1145/3779436.
- The abstract describes an LLM-supported KG and path-finding recommender for
  MOOC courses, evaluated on XueTang and COCO.
- ACM TOIS is a CCF A journal. The method is more recent than the available
  conference candidates, but reproducibility and temporal/metadata leakage are
  substantial concerns.
- The current processed COCO has course metadata and concept/category KGs but
  does not preserve the instructor/resource structure required by the title's
  full entity set. Dataset identity and preprocessing must be verified before
  adaptation.

## Found but Not Runnable

| Model | Venue | Reason not to prioritize now |
|---|---|---|
| EduGraph: *Learning Path-Based Hypergraph Neural Networks for MOOC Course Recommendation* | IEEE Transactions on Big Data 2024 | Direct course target, but earlier repository searches found no official code. Its interaction-derived learner hyperedges also leave strict cold courses without training incidence unless an unverified attribute channel exists. |
| Course-fairness multi-graph contrastive recommendation | IP&M 2024 | Direct course topic, but non-A venue and no verified code. |
| LE-DLCM | KBS 2025 | Recent LLM course recommendation, but earlier screening found no verified official implementation. |
| TOME | AAAI 2025 | Audited NO-GO: fixed course classification head cannot score strict cold courses and Junyi/COCO lack the required video layer. |
| BEICL / CL-KCRec | WWW 2024 | Audited NO-GO: target is knowledge concepts, not courses. |
| MSEC-Rec | IP&M 2026 | Audited NO-GO: released item scorer is ID-only and its random-walk implementation is infeasible at current user scale. |

## Practical Recommendation

Do not begin another GPU run from this list yet. The highest-value next action
is an HCNCR code/full-text acquisition audit. If the authors provide a runnable
implementation and its course hypergraph encodes cold courses from permitted
attributes, HCNCR should become the next one-seed strict feasibility target.
KnowPath is the second lead, but only after its data provenance, frozen LLM/KG
construction, and source code are available.
