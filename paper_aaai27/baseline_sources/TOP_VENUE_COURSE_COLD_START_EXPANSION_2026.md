# Top-Venue Course and Cold-Start Baseline Expansion

Search and source-audit date: 2026-07-13.

## Scope and Protocol Gate

This screen broadens the search beyond papers whose titles explicitly contain
"course recommendation." It covers MOOC/course recommendation, strict
item/course cold-start recommendation, sequential new-item recommendation, and
learning-path recommendation in AAAI, IJCAI, KDD, WWW, SIGIR, CIKM, WSDM,
RecSys, ACM MM, ICDE, and top journals.

A model is not considered main-table ready merely because its source runs. A
valid adaptation must use the existing external strict course-cold splits,
train-only interaction/graph evidence, full-catalog user-to-course ranking,
train-history masking, validation-only checkpoint selection, and course-macro
Recall/NDCG. Warm courses must remain in the candidate catalog when cold targets
are evaluated.

## Deduplication Against the Current Main Table

The current table already includes the following relevant families: DropoutNet,
CCFCRec, ALDI, SEMCo, CGRC, KGRec, PCGNN, and USIM. Consequently, ALDI, CGRC,
CCFCRec, and USIM are evidence that the generic item-cold literature has already
been represented; they are not new expansion candidates.

The strongest genuinely new choices are:

1. **GAR** for the fastest low-risk single-seed feasibility run.
2. **TDRO** for the highest incremental scientific value, because it adds
   temporal and distributional robustness rather than another plain
   content-to-CF mapper.
3. **ColdGPT** for the closest match between a strict item-attribute graph and
   the current course-concept graph.

## New Source-Available Item-Cold Candidates

| Priority | Model | Venue | Source status | Fit and required protocol repair | Decision |
|---:|---|---|---|---|---|
| 1 | GAR, *Generative Adversarial Framework for Cold-Start Item Recommendation* | SIGIR 2022, DOI `10.1145/3477495.3531897` | Official TensorFlow 1.15 source: <https://github.com/zfnWong/GAR>; modern implementation also exists in ColdRec | Direct new-item content-to-collaborative transfer with one content vector and pretrained warm embeddings. The official cold evaluator masks every warm item and is user-macro, so its loader/evaluator cannot be reused. | **Best next one-seed engineering target.** Low adapter burden, but method-family novelty is limited because ALDI and CCFCRec are already in the table. |
| 2 | TDRO, *Temporally and Distributionally Robust Optimization for Cold-start Recommendation* | AAAI 2024, DOI `10.1609/aaai.v38i8.28721` | Official PyTorch source: <https://github.com/Linxyhaha/TDRO> | Supports item features, cold-item sets, temporal environments, and an all-item ranking path. MOOCCube train rows contain timestamps. Official inference builds exclusion state from train+validation+test, cold evaluation removes warm items, and metrics are user-macro. Its dense user-item environment state also needs a memory audit. | **Highest-value new paper candidate**, after a one-seed memory/leakage audit. |
| 3 | ColdGPT, *Multi-task Item-attribute Graph Pre-training for Strict Cold-start Item Recommendation* | RecSys 2023, DOI `10.1145/3604915.3608806` | Official source: <https://github.com/YuweiCao-UIC/ColdGPT> | Course-concept relations map naturally to its item-attribute graph. The release ranks within cold items and constructs user profiles with held-out warm interactions. Replace both with train-only profiles and an external full-catalog scorer. Requires PyG, Transformers, and sentence-transformers. | **Strong graph-aware feasibility audit**, but heavier than GAR/TDRO. |
| 4 | GoRec, *A Generative Cold-start Recommendation Framework* | ACM MM 2023, DOI `10.1145/3581783.3612238` | Official source: <https://github.com/HaoyueBai98/GoRec>; ColdRec implementation available | Uses side information, pretrained warm embeddings, KMeans labels, and a VAE generator. The release defaults to a cold-only evaluation range and monitors test metrics during training; both must be replaced. | Good backup after GAR/TDRO; overlaps the existing content-transfer family. |
| 5 | PROMO, *Prompt Tuning for Item Cold-start Recommendation* | RecSys 2024, DOI `10.1145/3640457.3688126` | Official source: <https://github.com/PROMOREC/PROMO> | Uses user/item features and sequential history. The release evaluates one positive against 100 sampled negatives and computes test metrics every epoch. A full-catalog pair scorer is possible but expensive, and strict histories must be rebuilt externally. | Recent and official, but substantially less protocol-ready than GAR. |
| 6 | MI4Rec, *Pretrained Language Model based Cold-Start Recommendation with Meta-Item Embeddings* | CIKM 2025, DOI `10.1145/3746252.3761313` | Source: <https://github.com/zhengzaiyi/MI4Rec> | Produces all-item logits and masks train interactions, which is useful. Its cold evaluator still masks warm items and reports user-macro results. The pipeline expects item text and user-item review text and launches eight processes by default. | Attractive recent text model, but poor immediate fit to the single-12-GB-GPU, three-dataset setting. |
| 7 | Heater, *Recommendation for New Users and New Items via Randomized Training and Mixture-of-Experts Transformation* | SIGIR 2020, DOI `10.1145/3397271.3401178` | Official source: <https://github.com/Zziwei/Heater--Cold-Start-Recommendation>; ColdRec implementation available | Direct new-item feature-to-embedding transformation. The official release is legacy TensorFlow and its evaluator must be replaced. | Defensible older backup, but adds less than GAR or TDRO. |
| 8 | CLCRec, *Contrastive Learning for Cold-Start Recommendation* | ACM MM 2021, DOI `10.1145/3474085.3475665` | Official source: <https://github.com/weiyinwei/CLCRec>; ColdRec implementation available | The official code has a full-ranking path and a hybrid all-item evaluation, but cold validation masks warm items. It assumes old PyG packages and often multiple modalities. | Runnable through a modern reimplementation, but older and redundant with current content-transfer baselines. |

### Source-Available but Not Top-N Protocol Ready

| Model | Venue/source | Mismatch |
|---|---|---|
| CVAR, model-agnostic conditional VAE | SIGIR 2022; <https://github.com/Pillars-Creation/Conditional-Variational-Autoencoder-Recommendation> | Official task is CTR-style classification with AUC/F1 across warm-up stages, not full-catalog top-N ranking. |
| CREU | CIKM 2025; <https://github.com/EsiksonX/CREU> | Five-page paper and source use CTR backbones and AUC/F1, not user-to-item full ranking. |
| CCFCRec official release | WWW 2023; <https://github.com/zzhin/CCFCRec> | Official test retrieves users for each new item. The existing main-table wrapper is therefore already a labeled task adaptation. |
| FS-GNN | AAAI 2025; <https://github.com/leisongyuan/FS-GNN> | Official experiments are rating prediction with MAE/RMSE. The ColdRec PPR state is also too dense at the present user scale without redesign. |
| M2VAE | AAAI 2026; <https://github.com/hchchchchchchc/M2VAE> | Official evaluation is new-item-to-user retrieval and requires categorical plus image modalities. A user-to-course result would be a substantial task conversion. |

## Recent Paper-Only Watchlist

These papers are relevant but did not yield a verified author training
repository in the current search. They should remain related-work or
author-code-request entries rather than paper-only main-table reimplementations.

| Model/paper | Venue | Main reason to wait |
|---|---|---|
| Preference Aware Dual Contrastive Learning for Item Cold-Start Recommendation | AAAI 2024, DOI `10.1609/aaai.v38i8.28763` | No verified source. |
| Firzen: Firing Strict Cold-Start Items with Frozen Heterogeneous and Homogeneous Graphs | ICDE 2024, DOI `10.1109/ICDE60146.2024.00354` | No verified source; multimodal heterogeneous graph burden is high. |
| Preference Aware Item Cold-Start Recommendation With Hierarchical Item Alignment | IEEE TKDE 2025, DOI `10.1109/TKDE.2025.3613263` | No verified source. |
| Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning | KDD 2025, DOI `10.1145/3690624.3709336` | No verified source; online arrival protocol differs from the fixed strict split. |
| LLM Reasoning for Cold-Start Item Recommendation | WWW 2026, DOI `10.1145/3774904.3792872` | Very recent; no verified source in the current audit. |
| Let It Go? Not Quite: Content-Based Initialization for Sequential Item Cold Start | RecSys 2025, DOI `10.1145/3705328.3748038` | No verified source; sequential next-item protocol must be converted to static full-catalog course ranking. |

The WWW 2024 paper *Large Language Models as Data Augmenters for Cold-Start
Item Recommendation* (DOI `10.1145/3589335.3651532`) appears in the companion
proceedings, not the main WWW research track, and no verified source was found.

## Direct Course/MOOC Models

| Candidate | Venue/source | Status under the current protocol |
|---|---|---|
| HHCoR | IJCAI 2024, <https://www.ijcai.org/proceedings/2024/232> | Best direct course-paper adaptation lead. No verified official source; the local HHCoR-style implementation must be labeled adapted and moved to the strict external split. |
| HCNCR | IEEE TKDE 2025, DOI `10.1109/TKDE.2025.3568709` | Course-attribute hypergraphs are promising for cold courses, but no verified source/full reproducible package was found. |
| C3Rec | RecSys 2025, DOI `10.1145/3705328.3748083` | Direct cross-behavior course recommendation, but no verified trainable model source. The Co-MAC demo is not C3Rec. |
| TOME | AAAI 2025, DOI `10.1609/aaai.v39i12.33453` | Multi-type MOOC recommendation, but its fixed course classification head cannot natively score held-out cold-course IDs; no verified source. |
| DCBVN | WWW 2020, DOI `10.1145/3366423.3380236` | Full WWW research paper with career-development-aware employee training recommendation. No source and enterprise skill/career inputs do not match the three datasets. |
| UPGPR | LAK 2024, <https://github.com/epfl-ml4ed/courserec> | Source-available and course-specific, but not an A-tier venue; the completed one-seed audit showed weak path reachability under strict full-catalog scoring. |

## Learning-Path Models

Learning-path work is useful related work and a source of educational graph or
policy components, but it cannot be inserted into the course-ranking table
without changing the prediction target and metrics.

| Model | Venue/source | Why it is not a course-ranking baseline |
|---|---|---|
| SRC | AAAI 2023; MindSpore source at <https://github.com/mindspore-ai/models/tree/master/research/recommend/SRC> | Generates a concept-aware path and uses a knowledge-tracing/reward evaluator. |
| GEHRL | CIKM 2023, DOI `10.1145/3583780.3614897` | Hierarchical-RL learning-path model; an implementation is available through the UNO repository, but the target is a problem sequence. |
| PKSD | KDD 2024, DOI `10.1145/3637528.3671872` | Educational path policy with privileged knowledge-state distillation; no verified source. |
| Item-Difficulty-Aware Learning Path Recommendation | KDD 2024, DOI `10.1145/3637528.3671947` | Optimizes path difficulty/learning behavior, not catalog-level course Recall/NDCG; no verified source. |
| LIGHT | SIGIR 2025, DOI `10.1145/3726302.3730022` | Topology-aware path sequence optimization; no verified source. |
| UNO | AAAI 2026; <https://github.com/PengLinzhi/UNO-LPR> | Official source directly supports Junyi and ASSIST09, but predicts problem-level paths and evaluates learning gain. |

## Three-Dataset Adaptability

The workspace already uses 768-dimensional frozen course-content features for
MOOCCube/MOOCCubeX, Junyi, and COCO. This makes single-content-vector methods
portable, although it does not remove the need for protocol-safe training and
evaluation.

| Candidate | MOOCCube | Junyi | COCO | Main extra input |
|---|---|---|---|---|
| GAR | High | High | High | One content vector plus warm CF embeddings |
| TDRO | High, timestamps confirmed | Needs timestamp audit | Needs timestamp audit | Item content, first-appearance/period grouping, warm backbone |
| ColdGPT | High | Medium | Medium | Item-attribute graph and text encoder |
| GoRec | High | High | High | Content vector, warm embeddings, cluster labels |
| PROMO | Medium | Low-medium | Low-medium | User/item features and ordered histories |
| MI4Rec | Medium | Low | Low | Item text plus user-item text/reviews; large PLM runtime |
| Heater | High | High | High | Content vector plus warm embeddings |

## Mandatory Adapter Rules

No newly found official evaluator is directly compliant. A common adapter must:

1. Load the existing strict split instead of each repository's random or
   temporal splitter.
2. Train the warm backbone and every auxiliary graph/profile only from
   `static_train.pkl`.
3. Exclude strict validation/test cold courses from recommendation positives and
   negative sampling during training, while retaining permitted static metadata.
4. Score every catalog course, including warm competitors; mask only the user's
   training history and padding IDs.
5. Select checkpoints using validation cold course-macro `N@10`, never test
   metrics.
6. Export per-course Recall/NDCG and the same protocol manifest fields used by
   the current table.

## Recommended Execution Order

1. **GAR seed 2025**: fastest proof because the model only needs the already
   available content matrix and warm embeddings. Use the ColdRec implementation
   but replace its loader, negative sampling, checkpoint rule, and evaluator.
2. **TDRO seed 2025**: retain its robust objective, use the MOOCCube train
   timestamps for environments, remove train+validation+test history leakage,
   and replace dense environment state if the 12 GB GPU/RAM gate fails.
3. **ColdGPT feasibility audit**: verify that all course-attribute edges are
   static/train-safe and that a train-only learner profile can drive an external
   full-catalog scorer.
4. Use **GoRec** or **Heater** only as backups. Do not spend GPU time on FS-GNN,
   M2VAE, CVAR, or CREU for the main table without explicitly changing and
   relabeling their tasks.

The practical distinction is important: GAR is the best next implementation,
while TDRO is the better potential paper addition because the main table already
contains several content-transfer baselines.

## Local Source Evidence

The audited repositories are staged under `tmp/candidate_repos` and were not
used to overwrite experiment or paper source files. Key snapshots include:

| Repository | Audited commit |
|---|---|
| ColdRec | `18efd24ec79b0ac2b5b7b10ebc8703274fc117d1` |
| GAR | `280bfc1ec32c5746ce61c76c85a2bb796b6d8002` |
| TDRO | `1ee2a91647653575ba00e4be8240a99807b67e95` |
| ColdGPT | `d6a69f065a9e3268597845116077108890484be6` |
| GoRec | `20836eebc2ca6b6e580998b31523948d5734ce5f` |
| Heater | `355b9b85352fc5bfe904280a591d05d7d507467c` |
| PROMO | `1bfb2bb6c077512476483cda5747687fb6de75d7` |
| MI4Rec | `4be82860bd6575de67c0dd7866ab5c29fadbbe1f` |
| UNO-LPR | `8f9c4bc2c78089539f8415cf8d491b7bf21a1289` |

