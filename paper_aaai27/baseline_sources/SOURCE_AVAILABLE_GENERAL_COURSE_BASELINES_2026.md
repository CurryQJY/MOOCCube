# Source-Available General Course-Recommendation Baselines

Search and source-audit date: 2026-07-14.

## Revised Inclusion Rule

This shortlist applies the revised rule that a model may target general
university, enterprise-training, or MOOC course recommendation. A paper-specific
trainable source release is required. Native strict course-cold support is no
longer required for discovery, but the adaptation burden is reported explicitly.

Toy Streamlit/Coursera similarity projects, paper lists, and repositories that
only contain prompts or result files are excluded.

## Ranked Shortlist

The previously provisional ESWA 2025 lead is now identified as **IDRMI** and
matched to the paper-indexed repository <https://github.com/miaomiao924/IDRMI>.
Its local `master` codeload archive has been audited. The code is paper-specific,
but the public release omits the large dataset and contains a concrete empty-
adjacency loader defect, so source availability must not be confused with
protocol readiness.

| Rank | Model | Venue | Source status | Current-protocol fit | Decision |
|---:|---|---|---|---|---|
| 1 | **IDRMI**, *An Explainable Graph-Based Course Recommendation Model Based on Multiple Interest Factors* | Expert Systems with Applications (2025), DOI `10.1016/j.eswa.2024.125889` | Paper-indexed PyTorch source: <https://github.com/miaomiao924/IDRMI>. Local codeload archive audited; public data are omitted. | Newest direct-course source, with arbitrary pair scoring and a KG course branch. The released NGCF adjacency is empty, cardinalities and CUDA are hard-coded, train batches are resampled with replacement, and evaluation is label classification. | **First source-repair feasibility gate; not yet a formal baseline.** |
| 2 | **KGAN**, *Knowledge Grouping Aggregation Network for Course Recommendation in MOOCs* | Expert Systems with Applications 211 (2023), DOI `10.1016/j.eswa.2022.118344` | Paper-specific repository with TensorFlow source and processed MOOCCube course data: <https://github.com/StZHY/KGAN> | **Best lower-risk fit.** The release already scores every non-history course and reports Recall/NDCG. User representations are built from train histories and a course KG. Replace the random row split, balanced labels, user-macro evaluator, and test-every-epoch loop with the external strict split and course-macro evaluator. | **Next single-seed candidate if IDRMI's source gate fails.** |
| 3 | **MSEC-Rec**, *Meta-path Sampling-Enhanced Course Recommendation in Heterogeneous Networks* | Information Processing & Management 63 (2026), article 104482, DOI `10.1016/j.ipm.2025.104482` | Paper-specific PyTorch/DGL source and a MOOCCube data link: <https://github.com/mmx124/MSEC-Rec> | Direct and recent, but all released meta-paths produce user representations while candidate courses remain ID embeddings. A zero-interaction course gets no graph-derived candidate representation. The configured random walks are also infeasible at 199,199 users. | **Source-valid recent model, but strict-cold NO-GO without method redesign.** |
| 4 | **HRL**, *Hierarchical Reinforcement Learning for Course Recommendation in MOOCs* | AAAI 2019, DOI `10.1609/aaai.v33i01.3301435` | Authors' basic implementation: <https://github.com/jerryhao66/HRL> | Direct course task and trainable TensorFlow code, but the release uses leave-one-out sampled negatives and learned course IDs. Strict cold courses have no trained embedding. A content/KG course encoder would materially change the original model. | Keep as a top-conference warm/general-course baseline; lower strict-cold priority than KGAN. |
| 5 | **CKGE**, *Contextualized Knowledge Graph Embedding for Explainable Talent Training Course Recommendation* | ACM TOIS 2023, DOI `10.1145/3597022` | Paper-specific PyTorch source: <https://github.com/njustkmg/CKGE> | Scores training courses through employee-course KG paths, but the release requires proprietary-style `e2c_*`, shortest-distance, and path files and contains no ready course dataset. Exporting equivalent inputs for all three datasets is expensive. | Source-valid, but only audit after KGAN. |
| 6 | **UPGPR**, *Finding Paths for Explainable MOOC Recommendation: A Learner Perspective* | LAK 2024, DOI `10.1145/3636555.3636898` | Official source: <https://github.com/epfl-ml4ed/courserec> | The strict adapter is already implemented and protocol-compliant, but seed 2025 showed weak path/policy reachability under full-catalog ranking. | Retain as completed feasibility evidence; do not spend three-seed GPU time unless the candidate policy is redesigned. |
| 7 | **Goal-based Course Recommendation** | LAK 2019, DOI `10.1145/3303772.3303814` | Authors' PyTorch code: <https://github.com/CAHLR/goal-based-recommendation> | Recommends personalized prerequisites for a specified goal course using semester histories, majors, and grades. Only synthetic example data are public. This is not unconditional user-to-course catalog ranking. | Source-valid but task-incompatible with the main table. |
| 8 | **Serendipitous Course Recommendation** | LAK 2020, DOI `10.1145/3375462.3375524` | Authors' PyTorch code: <https://github.com/CAHLR/Serendipitous-Course-Recommendation> | Learns course/subject/instructor embeddings with a fixed course output layer and evaluates analogy/serendipity behavior. Cold courses cannot receive a learned output weight without changing the model. | Useful related work, not a strict-cold main-table baseline. |

## IDRMI Source Audit

The audited `master` codeload archive contains the train entry point, NGCF and
KGCN branches, and all three published interest-factor modules: course match,
user choice, and course preference. Its scorer accepts arbitrary user-course
pairs, and KG-connected courses can receive KGCN representations without CF
training interactions.

Mandatory source repairs and protocol replacements:

1. Populate `Data.R` from positive `train_set.txt` rows. In the release, the
   only `self.R[uid, item] = 1` block is commented out, so every saved NGCF
   adjacency is empty.
2. Derive user, course, entity, and relation cardinalities from the strict
   export instead of hard-coding 199,199 users and 698 courses.
3. Remove hard-coded `.cuda()` calls and preserve gradients only through the
   learned NGCF/KGCN score; the three rule-based interest factors intentionally
   operate outside autograd in the released implementation.
4. Build `course` and `User` histories from train positives only. Cold
   validation/test interactions must not enter the course-match, user-choice,
   or course-preference factors.
5. Replace random-with-replacement epoch batches and balanced-label
   precision/recall/AUC evaluation with deterministic train batches,
   validation checkpointing, full-catalog masking, and course-macro
   Recall/NDCG.

This is a repair of a paper-specific source release, not a paper-only
reimplementation. It is nevertheless a larger and riskier adaptation than
KGAN because the released training path is not operational as written.

## KGAN Source Audit

Audited repository commit:
`041f80099ad5232fbb1b04fa0fcad47de4edc407`.

The bundled MOOCCube-derived course data contain:

- 64,172 labeled user-course rows;
- 7,157 users and 219 courses;
- 32,086 positives and 32,086 sampled negatives;
- 17,893 KG triples, 7 relations, and 2,072 entities.

Useful source properties:

1. `train.py::topk_eval` builds the candidate set from all available items,
   removes each user's training history, and scores every remaining course.
2. The scorer accepts arbitrary user-course pairs and therefore does not need a
   sampled-negative output head at inference.
3. The user's multi-hop memories are generated from positive training history
   and KG neighbors, which maps naturally to the existing train-only history
   rule.
4. Course IDs are KG entity IDs, so a held-out course can at least be represented
   when it has permitted KG edges. This is substantially better than a pure
   course-ID classifier.

Mandatory repairs:

1. Replace `dataset_split`, which randomly splits already balanced positive and
   negative rows, with the existing external strict item-cold split.
2. Generate training negatives only from warm training courses; never use
   validation/test cold courses as negative labels.
3. Build all user memories from `static_train.pkl` only. Use static course-KG
   edges for held-out courses only when those edges are allowed by the protocol.
4. Add a validation split and select checkpoints by validation cold
   course-macro `N@10`; the release currently prints test metrics every epoch.
5. Replace user-macro Recall/NDCG with the established course-macro evaluator,
   while retaining all warm courses as full-catalog competitors.
6. Port TensorFlow 1.10/Keras 2.2 code to the current runtime while preserving
   the published TransD grouping/aggregation equations and hyperparameters.

## HRL Source Audit

Audited repository commit:
`a6867e315f00d6c3c865091a0cc962084dcfbfc7`.

The source is complete enough to inspect and contains the recommender, high/low
policies, environment, data generator, training loop, and evaluation. However:

- the dataset is not included;
- evaluation is one positive plus sampled negatives;
- user state is a sequence of learned course-ID embeddings;
- the candidate course is also a learned ID embedding;
- training repeatedly evaluates the test set.

Therefore, a strict cold-course HRL result would require a new side-information
course encoder. That is a larger method change than the KGAN adapter and should
not be the next experiment.

## Other Audited Source Commits

| Repository | Commit |
|---|---|
| CKGE | `4fdc1021c7f7a9a46222219868235773481f6902` |
| Goal-based Course Recommendation | `58d0d417e34ef9506a3d81cda83c1b2152de553b` |
| Serendipitous Course Recommendation | `20ec73c9ee15bb718fd644a3292c00e0d875eaa9` |

## Recommendation

Under the revised source-first rule, **IDRMI is the newest verified direct-course
source**, while **KGAN remains the lower-risk experiment**. The defensible order
is a small IDRMI source-repair feasibility gate first; if its scorer cannot
produce nonempty strict full-catalog outputs without method changes, stop and
implement one strict KGAN seed. Neither model belongs in the main table before
that gate and the usual protocol checks pass.
