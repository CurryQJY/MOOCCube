# Recent A-Tier Course-Recommendation Baseline Screening

Screening date: 2026-07-12

## Scope

The screening targets papers from 2023--2026 in AAAI, SIGIR, KDD, WWW, and
NeurIPS. A candidate must be directly related to course, MOOC, learning-resource,
or learning-path recommendation. Main-table suitability additionally requires a
runnable implementation and a defensible adaptation to strict item-cold splits,
full-catalog ranking, train-history masking, and item-macro Recall/NDCG.

## Shortlist

| Priority | Model | Venue | Direct educational recommendation | Code | Fit to current protocol | Decision |
|---:|---|---|---|---|---|---|
| 1 | EdGCL | AAAI 2026 | Yes; student-resource recommendation on MOOCCubeX and MOOPer | Official: <https://github.com/DaSESmartEdu/EdGCL> | Failed feasibility gate | Keep as related work; see `EDGCL_MOOCCUBE_SEED2025_FEASIBILITY_AUDIT.md` |
| 2 | TOME | AAAI 2025 | Yes; hierarchical course, concept, and video recommendation on XuetangX data | No verified public repository | Medium on MOOCCube, low on Junyi/COCO | Wait for code or author response; paper-only reimplementation is high risk |
| 3 | SRC | AAAI 2023 | Yes; concept-aware learning-path recommendation | Official MindSpore implementation | Low | Related-work or supplementary candidate, not a main-table item ranker |
| 4 | MRMLREC | AAAI 2024 student abstract | Yes; MOOC video recommendation | No verified public repository | Low | Exclude from the main table |

## Candidate Evidence

### EdGCL

Paper: *EdGCL: Disentangling Social and Cognitive Homophily in Graph-Based
Educational Recommender Systems*.

- The paper reports MOOCCubeX and MOOPer experiments with Recall@5/10,
  NDCG@5/10, MRR, and an all-ranking protocol.
- The released implementation uses user-resource dot-product scores and a BPR
  recommendation loss, so the final scorer is compatible with an external
  full-catalog evaluator.
- The core method requires both a social graph and a heterogeneous cognitive
  learning graph. The current MOOCCube files contain user-course, user-video,
  course-video, course-concept, and video-concept relations, but no direct
  forum, reply, friendship, or discussion relation. Junyi and COCO expose even
  less comparable social behavior.
- Replacing social behavior with co-enrolment or train-history similarity would
  materially change the meaning of EdGCL's social view. Such a run must be
  labelled adapted and justified, not presented as an official reproduction.
- The implementation uses dense graph tensors. The paper reports an A800 80 GB
  GPU, while the current machine has an RTX 5070 with 12 GB. A sparse or batched
  graph backend is likely required before formal runs.
- Repository audit found only MOOPer entries in `data.zip`, despite the README
  describing MOOCCubeX. Some released filenames also differ from those expected
  by `run_edgcl.py`, so the release needs a loader audit.

Go/no-go gates for a one-seed MOOCCube audit:

1. Construct every social/cognitive edge from training evidence only, with no
   validation/test cold-course interaction leakage.
2. Preserve cold-course representations through course-side relations rather
   than train interaction IDs.
3. Replace dense graph tensors sufficiently to fit the 12 GB GPU without
   changing the model objective.
4. Produce full-catalog scores with padding and train-history masking.
5. Obtain nonempty strict test `full_cold_item_macro` metrics from a fixed
   checkpoint.

Failure at any of the first three gates should stop the formal three-seed run.

### TOME

Paper: *Multi-Type MOOCs Recommendation: Leveraging Deep Multi-Relational
Representation and Hierarchical Reasoning* (AAAI 2025), DOI
<https://doi.org/10.1609/aaai.v39i12.33453>.

- TOME jointly models course, knowledge-concept, and video relations and uses
  hierarchical beam search for multi-type recommendation.
- Its XuetangX Computer and MOOC3 setup is highly relevant to MOOCCube's
  course-video-concept relations.
- No public GitHub/Gitee implementation was verified from the paper, title,
  acronym, DOI, or author-name searches.
- Junyi and COCO do not expose the same three-level course-concept-video
  structure, so a cross-dataset adaptation would require collapsing core model
  levels. This weakens its value as a matched three-dataset baseline.

TOME is the strongest paper-only backup, but it should not be implemented before
EdGCL's feasibility gate or an official-code release.

### SRC

Paper: *Set-to-Sequence Ranking-Based Concept-Aware Learning Path
Recommendation* (AAAI 2023), DOI
<https://doi.org/10.1609/aaai.v37i4.25630>.

- Official code is available at
  <https://github.com/mindspore-ai/models/tree/master/research/recommend/SRC>.
- The released datasets include Junyi, but the output is an entire learning
  path optimized by policy-gradient reward and a knowledge-tracing simulator.
- Converting SRC into single-course full-catalog ranking would change the task,
  target, and evaluator rather than only replacing the data loader.

SRC is relevant related work but is not a defensible next main-table baseline.

## Exclusions

- MRMLREC is an AAAI 2024 student abstract focused on MOOC video recommendation;
  no verified full implementation was found.
- DisCo (AAAI 2025) is a generic cold-start cross-domain recommender rather than
  a course-specific method. The current table already contains multiple generic
  cold-start baselines.
- MOOC quality evaluation and educational-video engagement papers do not solve
  the ranking task and are not recommendation baselines.
- The structured venue search did not identify another 2023--2026 direct
  course/MOOC recommender in SIGIR, KDD, WWW, or NeurIPS. This is an indexed
  metadata result, not a claim that no such paper exists.

## Recommendation

Do not start UPGPR as the next main-table baseline because LAK has no CCF A/B/C
rating. The EdGCL MOOCCube seed-2025 audit failed the social-evidence and
dense-memory gates: the source does not provide a true social relation for the
current protocol, and the release constructs global user-by-user tensors that
require at least 147.82 GiB per float32 matrix for 199,199 users. Retain EdGCL
and TOME as related-work citations rather than reporting a heavily degraded
adaptation.

## Practical Follow-Up Candidates

The EdGCL audit shows that venue quality alone is insufficient. The following
models were additionally screened for code availability, sparse scalability,
and compatibility with the current course-side relations.

| Priority rule | Model | Venue | Code and scalability | Strict adaptation work | Decision |
|---|---|---|---|---|---|
| A-tier first | KGRec | KDD 2023 | Official PyTorch code: <https://github.com/HKUDS/KGRec>; sparse user-item and KG edge indices; includes an all-item evaluator | Export strict CF train/validation/test plus course KG; exclude cold courses from CF positives and negatives while retaining their KG edges; use external item-macro evaluator | Best next A-tier feasibility audit |
| Course-specific first | MSEC-Rec | IP&M 2025 | Official code: <https://github.com/mmx124/MSEC-Rec>; directly consumes user-course, user-video, course-video, course-knowledge, and video-concept relations | Install/isolated DGL environment; replace random validation and sampled metrics; filter cold-course video behavior; use external full-catalog item-macro evaluator | Best next course-specific feasibility audit, but not an A-tier conference paper |
| Course-specific fallback | KGAN | ESWA 2023 | Official code: <https://github.com/StZHY/KGAN>; released MOOCCube course data | Replace hard-coded random split/evaluator and port TensorFlow 1.10 implementation to a supported stack | Do not prioritize: old framework and non-A venue |
| Explainable path fallback | UPGPR | LAK 2024 | Official code already staged locally | Replace random split and path-only top-10 scoring with full-catalog scores | Supplement-only candidate because LAK has no CCF A/B/C rating |

### KGRec Audit Rationale

KGRec is not a course-specific paper, but it is the most defensible next
A-tier baseline under the existing protocol. Its KDD 2023 implementation keeps
user-item and knowledge-graph relations sparse, computes item scores by dot
product, and already contains full-item evaluation infrastructure. MOOCCube
course-video, course-concept, course-teacher, course-school, and prerequisite
relations can become KG edges. Strict cold courses would have no CF training
positives, yet can still receive KG messages through their course-side edges.

The current environment lacks `torch_scatter` and the repository pins a much
older PyTorch/PyG stack. A feasibility audit should first establish whether a
compatible binary environment can be isolated or whether the two scatter
reductions can be replaced with native PyTorch operations without changing the
model objective. This dependency gate is materially smaller than EdGCL's global
dense-graph redesign.
