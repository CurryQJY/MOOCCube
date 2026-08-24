# BEICL / CL-KCRec Adaptability Audit

Status: NO-GO for the strict item-cold course-recommendation main table.

This audit concerns CL-KCRec, published as *Modeling Balanced Explicit and
Implicit Relations with Contrastive Learning for Knowledge Concept
Recommendation in MOOCs* at The Web Conference (WWW) 2024. DOI:
https://doi.org/10.1145/3589334.3645559.

## Previous Work

The workspace had no BEICL/CL-KCRec repository clone, adapter, screening note,
or experiment result before this audit.

## Original Task

CL-KCRec recommends knowledge concepts, not courses. Its heterogeneous
information network contains users, knowledge concepts, courses, videos, and
teachers, with user-click-concept (U-K) interactions as the supervised target.
The final score is a dot product between a user representation and a knowledge
concept representation. BPR is trained with positive and negative knowledge
concepts.

The original MOOCCube1819 protocol splits by a fixed calendar date and evaluates
each positive knowledge-concept interaction against 99 sampled negatives. It
does not use full-catalog item-macro course Recall/NDCG or strict item-cold
course splits.

## Data Compatibility

MOOCCube contains several compatible side relations: U-C, U-V, V-K, C-K, C-V,
and C-T. It does not contain the required U-K / user-click-concept relation.
The current strict task exposes U-C interactions as labels; neither the train,
validation, nor test split has knowledge-concept targets.

Junyi and COCO are even less compatible. Their processed relations contain
course-concept links (and Junyi prerequisite links), but no user-video,
video-concept, course-video, or teacher relation needed for the published HIN.

## Strict Protocol Blockers

- Retaining the original target would evaluate knowledge concepts, which is a
  different task from the paper's strict cold-course main table.
- Replacing knowledge concepts with courses changes the BPR target, all positive
  and negative sampling, the user/concept prototype graph, and the final scorer.
  It is a material redesign rather than a data-loader adaptation.
- The original evaluation uses 1-positive-plus-99-random-negatives. It would
  still need an external full-catalog, train-history-masked item-macro evaluator.
- The implicit-relation module composes and multiplies higher-order adjacency
  matrices. The paper reports 2,204 users, while the current strict MOOCCube
  split has 199,199 users. Without released sparse implementation details, its
  memory and runtime behavior at this scale are high risk.

## Reproducibility Gate

No official public repository was found locally or through exact-title,
acronym, and dataset-name GitHub repository searches on 2026-07-13. The paper
alone is insufficient for a defensible from-scratch reproduction under the
current protocol.

## Decision

Do not start a BEICL GPU feasibility run and do not include it in the strict
course-recommendation main table. It is relevant WWW 2024 educational
recommendation related work, but it is not an adaptable course-ranking baseline
for this study.
