# Source-Available MOOC Baseline Shortlist

Search date: 2026-07-13.

## Inclusion Rule

This screen requires a released implementation, a verified paper-to-repository
link, and a course, MOOC, educational-resource, concept, or learning-path
recommendation task. A source-available paper is not eligible for the main
table until it passes the existing strict item-cold gates: train-only evidence,
no cold-course CF positives or negatives in training, full-catalog ranking,
train-history masking, and item-macro Recall/NDCG.

## Closest Runnable Course Baseline

| Model | Paper and venue | Official source | Data overlap | Strict item-cold status | Decision |
|---|---|---|---|---|---|
| UPGPR | *Finding Paths for Explainable MOOC Recommendation: A Learner Perspective*, LAK 2024, DOI `10.1145/3636555.3636898` | <https://github.com/epfl-ml4ed/courserec> | The release directly supports MOOCCube/Xuetang and COCO, and exposes a custom entity/relation interface. | Conditional GO. The staged reader accepts strict MOOCCube exports, but the official `make_dataset.py` makes random per-user splits and the released evaluator is path/top-k based. Both must be replaced externally. | Best remaining course-specific source-available candidate. Run a one-seed strict feasibility audit before any formal run. |

LAK is a leading learning-analytics venue but has no CCF A/B/C rating. UPGPR
must therefore be labelled `UPGPR (adapted)` and, if reported, positioned as a
course-specific complement rather than an A-tier conference baseline.

## A-Tier Source Releases That Do Not Fit the Main Table

| Model | Paper and venue | Official source | Why it is excluded now |
|---|---|---|---|
| EdGCL | *Disentangling Social and Cognitive Homophily in Graph-Based Educational Recommender Systems*, AAAI 2026 | <https://github.com/DaSESmartEdu/EdGCL> | It ranks educational resources, not specifically courses, requires a real social graph absent from all current strict inputs, and materializes global dense user-by-user tensors. The verified 12 GB feasibility audit is NO-GO. |
| SRC | *Set-to-Sequence Ranking-Based Concept-Aware Learning Path Recommendation*, AAAI 2023, DOI `10.1609/aaai.v37i4.25630` | <https://github.com/mindspore-ai/models/tree/master/research/recommend/SRC> | Its target is a concept-aware learning path rather than a single course, so adapting it would change task, candidate space, and evaluation. |
| ACKRec | *Attentional Graph Convolutional Networks for Knowledge Concept Recommendation in MOOCs in a Heterogeneous View*, SIGIR 2020, DOI `10.1145/3397271.3401057` | <https://github.com/JockWang/ACKRec> | Its target is a knowledge concept, not a course. It is useful related work but cannot populate the course-ranking main table. |
| HRL | *Hierarchical Reinforcement Learning for Course Recommendation in MOOCs*, AAAI 2019 | <https://github.com/jerryhao66/HRL> | This is a direct course recommender, but it is substantially older, depends on legacy TensorFlow, and learns course-ID recommendation rows from interaction data. Strict cold courses would retain no trained interaction-side representation. |

## Recent Paper-Only Models

The exact-title, acronym, DOI, author-profile, GitHub, Gitee, and GitLab
searches did not find verified official code for the recent direct models:

- TOME, AAAI 2025.
- HCNCR, IEEE TKDE 2025.
- KnowPath, ACM TOIS 2026.

Their audits remain paper-only; they should not be reimplemented from an
abstract for a main-table comparison.

## Search Conclusion

The 2023--2026 intersection of A-tier venue, direct MOOC course ranking,
released official code, and current strict item-cold compatibility is empty.
For a new course-specific baseline with released source, UPGPR is the only
defensible next audit despite its LAK venue. The already completed PCGNN and
KGRec entries remain the A-tier adapted baselines in the main table.
