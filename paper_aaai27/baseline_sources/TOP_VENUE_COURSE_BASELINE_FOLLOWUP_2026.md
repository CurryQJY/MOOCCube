# Top-Venue Course-Recommendation Follow-Up

Search date: 2026-07-13.

This follow-up continues the post-UPGPR search for course-recommendation
baselines that could be aligned with the current strict course-cold protocol:
full-catalog ranking, train-history masking, train-only graph construction, and
course-macro Recall/NDCG on MOOCCube, Junyi, and COCO where possible.

## Bottom Line

The strongest new executable lead is **HHCoR**, but only as a clearly labelled
strict-protocol adaptation/reimplementation. It is an IJCAI 2024 direct course
recommendation paper and the local workspace already contains an HHCoR-style HIN
implementation. The missing piece is a strict split loader; the current script
still uses an older static/random split.

The strongest new top-venue paper lead is **C3Rec** from RecSys 2025, but no
verified trainable implementation was found. The public Co-MAC repository is an
LLM multi-agent sampled-candidate demo, not the C3Rec model.

The strongest source-available top-journal lead is **CKGE** from ACM TOIS, but
it targets talent-training course recommendation rather than MOOC course
recommendation and requires a custom enterprise-style KG/path export. It is less
matched to the current datasets than HHCoR.

## Ranked Candidates

| Priority | Candidate | Venue | Source status | Fit to current protocol | Decision |
|---:|---|---|---|---|---|
| 1 | HHCoR: *Hierarchical Reinforcement Learning on Multi-Channel Hypergraph Neural Network for Course Recommendation* | IJCAI 2024 | No verified official code found; local `hhcor_static_hin.py` is an HHCoR-style implementation | High for MOOCCube if adapted: course-side concept/prereq/co-occurrence channels can keep cold courses represented; current local evaluator supports full ranking and item-macro, but the entry point still uses `static_split_df` | Best next one-seed feasibility target, labelled `HHCoR-style/adapted` unless official code appears |
| 2 | C3Rec: *Breaking Knowledge Boundaries: Cognitive Distillation-enhanced Cross-Behavior Course Recommendation Model* | RecSys 2025 | Paper verified; no official model code found. Co-MAC repo links the publication but is not the trainable C3Rec implementation | Potentially good for MOOCCubeX because it is cross-behavior course recommendation, but likely depends on exercise/problem behavior and sampled candidate construction | Track author code; do not implement from paper only for the main table |
| 3 | CKGE: *Contextualized Knowledge Graph Embedding for Explainable Talent Training Course Recommendation* | ACM TOIS 2023/2024 | Official GitHub found and cloned to `tmp/candidate_repos/CKGE` | Medium-low: it can score over a course set, but its data pipeline expects `e2c_*` employee-course KG/path files and is not MOOC/MOOCCube-native | Optional top-journal feasibility audit only after HHCoR; high data-export burden |
| 4 | HCNCR: *Hypergraph Convolutional Networks for Course Recommendation in MOOCs* | IEEE TKDE 2025 | No verified public code/full reproducible package found | High on paper: course attribute hypergraphs are naturally cold-course compatible | Keep as author-code request / related work until source is available |
| 5 | KnowPath: *An LLM-Supported Knowledge Graph Construction and Path Finding Framework to Explainable MOOC Recommendations* | ACM TOIS 2026 | No verified public code found | Medium: reports XueTang and COCO, but LLM KG construction creates provenance/leakage risks | Track only; needs frozen KG inputs and source before any run |
| 6 | MAECR: *Towards Better Course Recommendations: Integrating Multi-Perspective Meta-Paths and Knowledge Graphs* | LAK 2025 | Paper found; no verified code | Medium for MOOCCube, weak for Junyi/COCO unless comparable meta-paths exist | Related-work / author-code follow-up, not next GPU target |
| 7 | Co-MAC | Public GitHub; related to RecSys 2025 C3Rec line | Source available, but it is an LLM agent/demo workflow | Low: OpenAI API dependency, sampled top-k candidate lists, no trainable full-catalog scorer | Exclude from main table |
| 8 | KnowCR | Applied Soft Computing 2026 | GitHub found but incomplete: `train.py` imports missing `models/KnowCR` | Low-medium if completed: MOOCCubeX and full user-course masking are present, but exercise graph is required and venue is not top conference | Do not run unless repository is completed |
| 9 | DRG | Neural Networks 2026 | Local source is prompt/data only | Low: generation over candidate text, not a runnable recommender | Exclude from main table |

## HHCoR Adaptation Notes

Local files inspected:

- `hhcor_static_hin.py`
- `hhcor_full_hin.py`
- `hin_eval_common.py`

Useful existing properties:

- Item vectors are content-plus-graph embeddings, so cold courses can still have
  representations when their train interactions are removed.
- The graph channels are concept similarity, prerequisite/co-order signal, and
  co-occurrence.
- `evaluate_embedding_ranker` already supports full-catalog ranking,
  train-history masking via `user_seen_items`, and `average_mode="item_macro"`.

Required changes before any main-table claim:

1. Add a strict split loader that reads
   `outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_<seed>`.
2. Build concept graph from permitted static course metadata only.
3. Build prerequisite/co-occurrence channels from `static_train.pkl` only.
4. Build validation histories from train history only; build test histories from
   train plus validation histories only if that matches the established baseline
   convention.
5. Select checkpoint by validation `full_cold_item_macro.N@10`.
6. Export JSON with protocol gates: `candidate_mode=full_catalog`,
   `item_macro_metrics=true`, `train_history_masking=true`, strict split root,
   seed, and cold item counts.

Recommended first run:

```powershell
.\py.bat paper_aaai27\scripts\hhcor_strict_adapter.py `
  --split-root outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2025 `
  --output-dir paper_aaai27\baseline_sources\_hhcor_strict\mooccube_seed2025_single_gpu `
  --epochs 8 `
  --device cuda
```

Only expand to three seeds if the seed-2025 run produces nonempty strict cold
course-macro metrics and the validation-selected checkpoint is not a pure
near-zero/reachability failure.

## Source Evidence

- HHCoR IJCAI page/PDF: https://www.ijcai.org/proceedings/2024/232
- C3Rec ACM DOI page: https://doi.org/10.1145/3705328.3748083
- Co-MAC repository: https://github.com/bianyh/Co-MAC
- CKGE DOI page: https://doi.org/10.1145/3597022
- CKGE repository: https://github.com/njustkmg/CKGE
- HCNCR DOI page: https://doi.org/10.1109/TKDE.2025.3568709
- KnowPath DOI page: https://doi.org/10.1145/3779436
- MAECR DOI page: https://doi.org/10.1145/3706468.3706486
- KnowCR repository: https://github.com/KasISET/KnowCR

## Recommendation

Do **not** start another UPGPR three-seed run. Its strict adapter is valid, but
the seed-2025 path/policy reachability result was too weak for a formal run.

If the goal is another course-specific baseline under the current protocol, the
next practical action is a **single-seed HHCoR strict adapter**. If the goal is a
strictly official top-venue source baseline, there is no newly found candidate
that is both source-available and protocol-ready beyond the baselines already in
the table.
