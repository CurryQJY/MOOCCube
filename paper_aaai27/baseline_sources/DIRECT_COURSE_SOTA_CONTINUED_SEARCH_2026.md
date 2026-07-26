# Continued Search for Direct Course-Recommendation SOTA Baselines

Search and source-audit date: 2026-07-14.

## Scope

This pass searched 2020--2026 work using course recommendation, MOOC course
recommendation, cold-start course recommendation, online/training course
recommendation, course recommender, knowledge-graph course recommendation, and
hypergraph course recommendation. Results were cross-checked through OpenAlex,
Semantic Scholar, DOI/publisher pages, Google, GitHub repository search, and
author publication pages.

A candidate is executable under the current protocol only if it can consume the
existing external strict item-cold split, construct every interaction-derived
input from training data, represent courses with zero training interactions,
rank the full course catalog, mask training history, select checkpoints on
validation data, and report course-macro Recall/NDCG.

## New Direct-Course Candidates

| Priority | Model | Venue | Source evidence | Strict cold-course assessment | Decision |
|---:|---|---|---|---|---|
| 1 | **IDRMI**, *An Explainable Graph-Based Course Recommendation Model Based on Multiple Interest Factors* | Expert Systems with Applications 2025, DOI `10.1016/j.eswa.2024.125889` | The paper-indexed source is <https://github.com/miaomiao924/IDRMI>; its `master` codeload archive is already staged under `baseline_sources/IDRMI` and contains the PyTorch model, NGCF/KGCN branches, interest-factor modules, loader, and training entry point. The README says the dataset is too large to upload and must be requested. | The pair scorer can evaluate arbitrary user-course pairs and its KGCN branch can encode KG-connected cold courses. However, the released loader leaves `self.R` empty because the only population block is commented out, hard-codes 199,199 users and 698 courses, samples train batches with replacement, and evaluates balanced-label classification rather than full-catalog ranking. | **Newest verified direct-course source lead, but repair-heavy.** Run a minimal source-repair feasibility gate before any formal seed. |
| 2 | EduGraph: *Learning Path-Based Hypergraph Neural Networks for MOOC Course Recommendation* | IEEE Transactions on Big Data 2024, DOI `10.1109/TBDATA.2024.3453757` | No official training repository was found. The corresponding author's publication page links only to IEEE Xplore, whereas papers with released code on the same page carry an explicit `code` link. | Potentially representable through course hyperedges, but the learning-path hypergraph must be reconstructed from train-only evidence. Any path containing validation/test interactions would leak held-out courses. The learner-course graph still needs an external full-catalog cold scorer and course-macro evaluator. | **Best newly identified paper-only lead.** Request source; do not reimplement or add to the table yet. |
| 3 | MHRR: *MOOCs Recommender Service With Meta Hierarchical Reinforced Ranking* | IEEE Transactions on Services Computing 2023, DOI `10.1109/TSC.2023.3325302` | No author repository was found by title, acronym, DOI, or author search. | Directly ranks MOOC courses and reports HR/NDCG, but its meta embedding generator revises noisy/sparse observed course profiles rather than demonstrating zero-interaction strict cold courses. The main dataset and seven-day online traffic are proprietary. | Strong direct-course related work, but **not reproducible or cold-ready**. |
| 4 | LE-DLCM: *Decoupled Learner and Course Modeling with Large Language Models for Enhanced Course Recommendation* | Knowledge-Based Systems 2025/2026, DOI `10.1016/j.knosys.2025.115135` | No official repository or open reproducible package was found. | LLM-based course modeling may encode unseen courses from text, but the exact LLM, prompts, course corpus, candidate construction, and leakage boundaries must be frozen before a strict-cold claim. Full-catalog scoring cost also needs an audit. | Recent direct-course watchlist item; wait for code and full inputs. |
| 5 | Prerequisite-Enhanced Category-Aware GNN | ACM TKDD 2024, DOI `10.1145/3643644` | No official repository was found. | Prerequisite/category metadata are promising for unseen courses, but an interaction-trained course-ID branch can still leave zero-degree courses untrained. Prerequisite edges must be static and available for all held-out courses. | Good architectural fit on paper; source request only. |
| 6 | Multi-graph contrastive course fairness model | Information Processing & Management 2024, DOI `10.1016/j.ipm.2024.103750` | No official repository was found. | Optimizes fairness with learner knowledge-background graphs, not strict new-course representation. It also requires learner-background inputs that are not consistently available across MOOCCube, Junyi, and COCO. | Not a practical three-dataset baseline. |
| 7 | H-BERT4Rec | IEEE Access 2024, DOI `10.1109/ACCESS.2024.3462830` | No official repository was found. | Sequential masked-item prediction over a fixed course vocabulary. A held-out course with no training occurrence has no learned output-head signal, so it is not strict item-cold compatible without changing the model. | Exclude from strict-cold experiments. |
| 8 | PLAN-BERT: *Degree Planning with PLAN-BERT* | AAAI 2021, DOI `10.1609/aaai.v35i17.17751` | No verified official training repository was found in this pass. | Recommends multi-semester degree plans and future known courses. Its output and evaluation target are course-plan generation rather than static full-catalog user-to-course ranking. | Top-conference course work, but task-incompatible. |

## Previously Known Candidates Reconfirmed

| Model | Status after this pass |
|---|---|
| HCNCR, IEEE TKDE 2025 | Still the strongest recent direct-course architecture on paper. No verified official source/full reproducible package. |
| C3Rec, RecSys 2025 | Still the strongest recent direct-course conference lead. No verified trainable source; Co-MAC is not a substitute for the C3Rec model. |
| HHCoR, IJCAI 2024 | Still the most plausible direct-course adaptation lead, but no official source. A local style implementation must remain explicitly labeled as adapted/style. |
| MSEC-Rec, IP&M 2026 | Official PyTorch/DGL source and MOOCCube data link are available. The released graph produces user embeddings only; candidate courses are ID embeddings, so strict zero-interaction courses require a method-level item-encoder redesign. |
| UPGPR, LAK 2024 | Official source and direct MOOC task, but the completed strict seed-2025 audit showed weak path reachability. |
| TOME, AAAI 2025 | Fixed course classification head still cannot represent strict held-out course IDs. |
| BEICL/CL-KCRec, WWW 2024 | Recommends knowledge concepts, not courses. |

## Source-Availability Result

IDRMI newly satisfies requirements 1--3 below, but its released loader and
evaluator fail requirement 4 without source repairs. No 2024--2026 model
satisfies all four requirements out of the box:

1. the original prediction target is a course or MOOC;
2. the venue is a top conference or strong journal;
3. an official trainable implementation is publicly available; and
4. the released model can represent a course with zero training interactions
   under full-catalog ranking.

The negative source finding is based on exact-title, acronym, DOI, and author
searches. It should be treated as "no verified public source found as of the
audit date," not as proof that private or unindexed code does not exist.

## Recommended Next Action

1. Run an **IDRMI source-repair feasibility gate**: populate the NGCF adjacency
   from positive training rows, remove hard-coded cardinalities/CUDA calls,
   preserve its three interest factors, and verify strict full-catalog scores.
2. If IDRMI cannot pass that gate without changing its method, use **KGAN** as
   the lower-risk official-source single-seed candidate.
3. Contact the authors of **HCNCR** and **EduGraph** for source and preprocessing
   scripts. These two models have the best metadata/graph path to strict cold
   courses.
4. Track **C3Rec** for an official release because it is the strongest recent
   conference paper whose original target is course recommendation.
5. Do not insert IDRMI, MHRR, LE-DLCM, EduGraph, or PLAN-BERT into the main table until
   source and protocol gates pass.

## Evidence Links

- EduGraph DOI: https://doi.org/10.1109/TBDATA.2024.3453757
- EduGraph author publication page: https://mingli-ai.github.io/publications.html
- MHRR DOI: https://doi.org/10.1109/TSC.2023.3325302
- LE-DLCM DOI: https://doi.org/10.1016/j.knosys.2025.115135
- Prerequisite-Enhanced Category-Aware GNN DOI: https://doi.org/10.1145/3643644
- Course fairness multi-graph DOI: https://doi.org/10.1016/j.ipm.2024.103750
- Explainable multiple-interest graph model DOI: https://doi.org/10.1016/j.eswa.2024.125889
- IDRMI source: https://github.com/miaomiao924/IDRMI
- H-BERT4Rec DOI: https://doi.org/10.1109/ACCESS.2024.3462830
- PLAN-BERT DOI: https://doi.org/10.1609/aaai.v35i17.17751
- HCNCR DOI: https://doi.org/10.1109/TKDE.2025.3568709
- C3Rec DOI: https://doi.org/10.1145/3705328.3748083
- HHCoR IJCAI page: https://www.ijcai.org/proceedings/2024/232
