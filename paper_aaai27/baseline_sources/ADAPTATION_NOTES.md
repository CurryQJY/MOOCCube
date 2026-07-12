# Course Baseline Source Pull and Adaptation Notes

Date: 2026-07-04

## Pulled Sources

| Candidate | Source | Local path | Pull result | Adaptation status |
|---|---|---|---|---|
| IDRMI | https://github.com/miaomiao924/IDRMI | `paper_aaai27/baseline_sources/IDRMI` | downloaded via GitHub codeload, branch `master` | Data files can be generated, but official graph loader builds an empty user-course adjacency unless source code is fixed. |
| KGAN | https://github.com/StZHY/KGAN | `paper_aaai27/baseline_sources/KGAN` | downloaded via GitHub codeload, branch `main` | Best current candidate for a course-specific baseline. Data loader accepts adapted MOOCCube format in a smoke test. Training needs a TensorFlow 1.x/Keras environment or porting. |
| DRG | https://github.com/WHCK1102/DRG | `paper_aaai27/baseline_sources/DRG` | downloaded via GitHub codeload, branch `main` | Repository contains prompt/data examples, not a directly runnable recommender training pipeline. Better treated as LLM-course-recommendation discussion or future baseline. |
| MSEC-Rec | https://github.com/mmx124/MSEC-Rec | `paper_aaai27/baseline_sources/MSEC-Rec` | downloaded via GitHub codeload, branch `main` | Recent course-specific IP&M 2025 candidate. Key files compile, but the official evaluator uses sampled ranking and random validation positives; needs strict split, full-catalog evaluator, and DGL dependency. |
| UPGPR / courserec | https://github.com/epfl-ml4ed/courserec | `paper_aaai27/baseline_sources/UPGPR-courserec` | downloaded via GitHub codeload, branch `main` | LAK 2024 explainable MOOC recommendation code. Custom-dataset interface is usable, but default split is user-level random and evaluation is top-10 path ranking; needs strict item-cold data generation and evaluator replacement. |
| PCGNN | https://github.com/lcwy220/PCGNN plus Google Drive code package from README | `paper_aaai27/baseline_sources/PCGNN` and `paper_aaai27/baseline_sources/PCGNN_recbole_drive` | GitHub README downloaded via codeload; 103 MB Drive package downloaded with `gdown` and extracted | Strong recent course-specific TKDD 2024 candidate. Drive package contains a modified RecBole project with `kg_model`, `course` and `xuetangx` atomic files. This is the best new baseline candidate if we can export our strict item-cold split into its RecBole atomic format. |

`git clone` to `github.com:443` failed in this environment, so the sources were downloaded from GitHub `codeload` zip archives. The original zip files are under `paper_aaai27/baseline_sources/_zips`.

Additional source note: the PCGNN GitHub repository itself only contains a README. The actual code is hosted through the README's Google Drive link as `RecBole-based PCGNN.zip`.

## Generated Adapter Data

Adapter script:

`paper_aaai27/scripts/prepare_course_baseline_adapters.py`

Generated strict MOOCCube seed-2025 adapter data:

`paper_aaai27/baseline_sources/_prepared/mooccube_seed2025`

Smoke-test adapter data:

`paper_aaai27/baseline_sources/_prepared_smoke/mooccube_seed2025`

Priority-order adaptability smoke script:

`paper_aaai27/scripts/course_baseline_adaptability.py`

Priority-order concrete experiment script:

`paper_aaai27/scripts/priority_baseline_experiments.py`

Generated priority-order smoke artifacts:

`paper_aaai27/baseline_sources/_adaptability/mooccube_seed2025_smoke`

Generated priority-order concrete experiment report:

`paper_aaai27/baseline_sources/_priority_experiments/mooccube_seed2025/priority_baseline_experiments.md`

The priority smoke uses the same seed-2025 strict split but caps the quick experiment at 2,000 train positives, 500 validation positives, and 500 test positives. It keeps the full 698-course item space.

The adapter exports:

| Target | Files |
|---|---|
| KGAN | `ratings_final.txt`, `kg_final.txt`, `strict_train.txt`, `strict_val_cold_pos.txt`, `strict_test_cold_pos.txt` |
| IDRMI | `train_set.txt`, `eval_set.txt`, `test_set.txt`, `rating_index.tsv`, `kg_index.tsv`, `user_items.txt`, `item_users.txt` |

Seed-2025 generated-data summary:

| Field | Value |
|---|---:|
| users | 199199 |
| courses/items | 698 |
| exported train positives | 50000 |
| exported train negatives | 50000 |
| validation cold positives | 32464 |
| test cold positives | 65605 |
| KG triples | 340652 |
| KG entities | 27743 |
| relation width | 25 |

## Smoke Test Results

### KGAN

The official `data_loader.load_data` accepts the smoke adapter data.

Observed smoke output:

| Check | Result |
|---|---:|
| train rows after official random split | 2906 |
| test rows after official random split | 326 |
| KG entities | 27743 |
| KG relations | 25 |
| aggregate users | 1608 |

Fresh priority-order smoke, after copying the adapted files into KGAN's hard-coded `KGAN-master/data/course_strict_seed2025` path:

| Check | Result |
|---|---:|
| train rows after official random split | 2884 |
| test rows after official random split | 328 |
| KG entities | 27743 |
| KG relations | 25 |
| aggregate users | 1598 |

Important caveats:

- The official KGAN loader randomly splits `ratings_final.txt`, so it does not preserve our strict item-cold train/val/test protocol.
- For fair paper use, KGAN needs a custom strict loader/evaluator that uses our existing cold split and full-catalog item-macro Recall/NDCG.
- Current `py.bat` environment does not have `tensorflow` or `keras`; KGAN requires the older TensorFlow 1.x stack, according to its README.

### IDRMI

The official loader reads the adapter files, but the graph is not usable without a source fix.

Observed smoke output:

| Check | Result |
|---|---:|
| train triples | 4000 |
| eval triples | 32464 |
| test triples | 65605 |
| users with train items | 1992 |
| items with train users | 314 |
| `R.nnz` before adjacency | 0 |
| adjacency nnz | 0 |

Root cause in official code:

`NGCF/utility/load_data.py` reads `train_set.txt`, `user_items.txt`, and `item_users.txt`, but the block that populates `self.R[uid, item] = 1` is commented out. As a result, `get_adj_mat()` creates an empty user-course graph.

### MSEC-Rec

Inspected files:

- `train.py`
- `model.py`
- `graph.py`
- `utils.py`
- `user_feature.py`

Observed status:

| Check | Result |
|---|---|
| key files compile | pass |
| framework | PyTorch + DGL |
| input format | `.npy` matrices and DGL heterograph |
| course-specific signals | user-course, user-video, course-video, course-knowledge, video-concept |
| default evaluation | sampled HR/NDCG/MRR with random negatives |
| strict item-cold readiness | medium; requires generating train/val/test `.npy` matrices and replacing sampled evaluator |

Main caveats:

- `train.py` samples one validation positive per user and evaluates against random negatives, so official metrics are not comparable to the paper's full-catalog item-macro protocol.
- `graph.py` has hard-coded paths under `./data` and saves `./graph/trin_heterograph.bin`, while `train.py` defaults to `./graph/train_heterograph.bin`; this should be normalized before a full run.
- Current `py.bat` environment is missing `dgl`.

Priority-order smoke artifacts were exported under:

`paper_aaai27/baseline_sources/_adaptability/mooccube_seed2025_smoke/msec/data`

Observed matrix smoke:

| Matrix | Shape | Nonzero count |
|---|---:|---:|
| `train_uc.npy` | 2976 x 698 | 2000 |
| `val_uc.npy` | 2976 x 698 | 500 |
| `train_uv.npy` | 2976 x 5180 | 14689 |
| `ck.npy` | 698 x 9712 | 5422 |
| `course_video.npy` | 698 x 5180 | 2094 |
| `video_concept.npy` | 5180 x 9712 | 23310 |

DGL remains missing, so official graph construction was not executed in this environment.

### UPGPR / courserec

Inspected files:

- `README.md`
- `config/UPGPR/mooc.json`
- `src/UPGPR/make_dataset.py`
- `src/UPGPR/data_utils.py`
- `src/UPGPR/test_agent.py`
- `src/baselines/baseline.py`

Observed status:

| Check | Result |
|---|---|
| key files compile | pass |
| framework | PyTorch path-reasoning agent plus RecBole baselines |
| input format | custom `enrolments.txt`, entity files, and relation files |
| course-specific signals | item-concept, item-teacher, item-school; can be extended |
| default split | per-user random train/validation/test split |
| default evaluation | top-10 path recommendations, user-level metrics |
| strict item-cold readiness | medium-high; custom data interface is simple, but split/eval must be overridden |

Main caveats:

- `make_dataset.py` always creates its own per-user random split. For strict item-cold use, we should bypass that splitter and write `train.txt`, `validation.txt`, and `test.txt` from our existing seed splits.
- `test_agent.py` ranks only candidates reached by predicted paths. This is explainable, but not automatically equivalent to full-catalog ranking; if used as a baseline, missing-path courses must receive a defined low score so the evaluator can rank the whole catalog.
- `easydict` and `wandb==0.16.3` were installed for smoke testing. This introduced a `protobuf` version warning against `opentelemetry-proto`, so full experiments should use a separate environment.
- `wandb==0.16.3` is not NumPy-2-clean because it references `np.float_` and `np.complex_`; the smoke test used a temporary alias shim. A clean environment should use NumPy<2 or a newer compatible wandb.

Priority-order smoke artifacts were exported under:

`paper_aaai27/baseline_sources/_adaptability/mooccube_seed2025_smoke/upgpr/processed_files`

With the NumPy alias shim, the official `Dataset` and `KnowledgeGraph` readers accept the generated strict files:

| Check | Result |
|---|---:|
| interactions loaded from `train.txt` | 2000 |
| users | 2976 |
| items | 698 |
| concepts | 1334 |
| teachers | 1206 |
| schools | 151 |
| KG user nodes | 2976 |
| KG item nodes | 698 |

### PCGNN

Inspected files:

- `PCGNN/README.md`
- `PCGNN_recbole_drive/RecBole-master/run_recbole.py`
- `PCGNN_recbole_drive/RecBole-master/recbole/model/sequential_recommender/kg_model.py`
- `PCGNN_recbole_drive/RecBole-master/recbole/properties/model/kg_model.yaml`
- `PCGNN_recbole_drive/RecBole-master/recbole/properties/dataset/course.yaml`
- bundled `dataset/course` and `dataset/xuetangx` atomic files

Observed status:

| Check | Result |
|---|---|
| Drive package size | 103,351,965 bytes |
| key files compile | pass |
| framework | modified RecBole |
| default entry | `python run_recbole.py --dataset=course` with model `kg_model` |
| input format | RecBole atomic files: `.inter`, `.item`, `.kg`, `.link` |
| course-specific signals | first/second-level course category and KG relations |
| default evaluation | RecBole full ranking with chronological leave-out settings |
| strict item-cold readiness | high among new candidates; needs strict atomic-file export and evaluator audit |

Main caveats:

- The provided RecBole project is modified and contains nonstandard model names and comments, so we should treat it as source code to adapt rather than a drop-in pip package.
- The default dataset config uses `eval_setting: TO_LS,full`; this is full ranking, but not our strict item-cold split.
- To use PCGNN fairly, export our train/validation/test folds as RecBole benchmark atomic files and ensure target cold-course interactions do not enter `.inter` training rows or KG-derived target ID features.

Priority-order smoke artifacts were exported into the modified RecBole source tree:

`paper_aaai27/baseline_sources/PCGNN_recbole_drive/RecBole-master/dataset/mooccube_strict_seed2025_smoke`

The generated config is:

`paper_aaai27/baseline_sources/PCGNN_recbole_drive/RecBole-master/recbole_mooccube_strict_seed2025_smoke.yaml`

Observed official loader smoke:

| Check | Result |
|---|---:|
| interactions read | 3000 |
| benchmark file sizes | 2000 / 500 / 500 |
| users after RecBole remap | 2977 |
| items after RecBole remap | 699 |
| entities after RecBole remap | 1554 |
| train dataloader batches | 1 |
| validation dataloader batches | 0 |
| test dataloader batches | 0 |

Interpretation:

- Atomic files and external split files are readable after ASCII-safe token export.
- The stock `kg_modelDataset` inherits the sequential build path and does not faithfully use the external item-cold train/valid/test split for sequence construction.
- The zero validation/test dataloaders show that PCGNN needs a loader/build patch or a custom evaluator before training, even though its data format is the best match among new candidates.

Observed concrete forward/full-sort smoke, using manually built strict train-history examples and forcing RecBole/PCGNN execution to CPU:

| Check | Result |
|---|---:|
| train sequence examples | 32 |
| validation sequence examples | 32 |
| one-step loss | 6.5364 |
| full-sort score shape | 32 x 699 |
| sample Recall@10 | 0.0000 |
| sample NDCG@10 | 0.0000 |
| median target rank | 325.5 |

Interpretation:

- PCGNN can execute its model forward pass, training loss, and full-catalog scoring on strict-history examples after the smoke script explicitly sets `use_gpu=False`, `gpu_id=-1`, and `device=cpu`.
- These numbers are not publishable results. They use one mini-batch, a capped smoke export, and no full training schedule.
- The next real work is to replace the official sequential dataloader/evaluator path with our strict item-cold split, train-history masking, full-catalog ranking, and item-macro aggregation.

Observed strict adapter run:

Script:

`paper_aaai27/scripts/pcgnn_strict_adapter.py`

Full PCGNN atomic dataset:

`paper_aaai27/baseline_sources/PCGNN_recbole_drive/RecBole-master/dataset/mooccube_strict_seed2025_full`

Full PCGNN config:

`paper_aaai27/baseline_sources/PCGNN_recbole_drive/RecBole-master/recbole_mooccube_strict_seed2025_full.yaml`

Full export summary:

| Check | Result |
|---|---:|
| dataset name | `mooccube_strict_seed2025_full` |
| train rows | 464314 |
| validation cold rows | 32464 |
| test cold rows | 65605 |
| item rows | 698 |
| KG rows | 5115 |

Latest report:

`paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2025/pcgnn_strict_adapter_report.md`

Latest full-config smoke report with best validation checkpoint:

`paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2025_full_smoke/pcgnn_strict_adapter_report.md`

Latest KG-joint smoke report:

`paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2025_full_kg_smoke/pcgnn_strict_adapter_report.md`

Latest KG-joint warm-candidate smoke report:

`paper_aaai27/baseline_sources/_pcgnn_strict/mooccube_seed2025_full_kg_warm_fullval_smoke/pcgnn_strict_adapter_report.md`

| Check | Result |
|---|---:|
| train sequence examples | 2048 |
| validation sequence examples | 1024 |
| test sequence examples | 1024 |
| epochs | 1 |
| last training loss | 6.5007 |
| validation cold item-macro items | 33 |
| validation cold item-macro R@10 / N@10 | 0.0000 / 0.0000 |
| test cold item-macro items | 58 |
| test cold item-macro R@10 / N@10 | 0.0069 / 0.0035 |

KG-joint smoke:

| Check | Result |
|---|---:|
| train sequence examples | 512 |
| validation sequence examples | 256 |
| test sequence examples | 256 |
| epochs | 2 |
| KG triples | 5115 |
| KG batch size / weight | 256 / 1.0 |
| last total / RS / KG loss | 8.5130 / 6.5155 / 1.9975 |
| validation cold item-macro R@10 / N@10 | 0.0000 / 0.0000 |
| test cold item-macro R@20 / N@20 | 0.0238 / 0.0060 |

KG-joint warm-candidate full-validation smoke:

| Check | Result |
|---|---:|
| train sequence examples | 2048 |
| validation sequence examples | 32464 |
| test sequence examples | 1024 |
| epochs | 1 |
| RS candidate mode | warm train items only |
| RS candidate items | 596 |
| validation cold item-macro R@10 / N@10 | 0.0006 / 0.0002 |
| validation cold item-macro R@20 / N@20 | 0.0100 / 0.0025 |
| test cold item-macro R@10 / N@10 | 0.0152 / 0.0052 |

Interpretation:

- The external PCGNN adapter now bypasses the broken official sequential valid/test dataloader and constructs train/eval sequences directly from the existing strict item-cold split.
- Evaluation uses PCGNN `full_sort_predict`, masks train-history items and padding token 0, restores the target score, then reports full-catalog item-macro Recall/NDCG.
- The adapter now defaults to the full dataset/config name and includes best validation checkpointing based on validation `full_cold_item_macro.N@10`, plus early stopping.
- The adapter now jointly trains PCGNN's recommendation loss and KG margin-ranking loss. This is required for a fair strict item-cold run because cold validation/test courses do not appear as recommendation positives in the train split, so item/entity embeddings must still receive KG-side updates.
- The adapter now computes the recommendation cross-entropy over train-split items by default (`--rs-candidate-mode warm`). This avoids treating strict validation/test cold courses as negative classes in every warm-item training step. Evaluation remains full-catalog.
- The adapter accepts `--device auto|cpu|cuda` and records the resolved device in each report. On the CUDA path, model weights, recommendation batches, and KG batches use the selected device.
- The local PCGNN source maps category lookup indices to CPU before indexing RecBole's CPU-resident item feature table, then moves the resulting category entity ids back to the model device. This is a device-compatibility fix only; it does not change PCGNN's graph, loss, candidates, or evaluator. A fixed-checkpoint CPU/CUDA check on 64 strict test sequences had identical Top-20 sets and orders, with maximum score difference `2.12e-5` from FP32 arithmetic.
- The full RecBole config no longer loads the all-one `rating` column, so the noisy `All the same value in [rating]` warning is avoided. PCGNN's session adjacency conversion was also changed from `torch.FloatTensor(A)` on a Python list to `torch.from_numpy(np.asarray(A, dtype=np.float32))`, removing the slow tensor-construction warning.
- Existing capped runs should not be copied into the paper main table. They prove the adaptation path works. A paper-number run should use uncapped examples, for example `--max-train-examples -1 --max-val-examples -1 --max-test-examples -1`.
- The earlier no-KG formal directory `mooccube_seed2025_full_formal` should not be used as a final result because it optimized only the recommendation loss and showed validation `full_cold_item_macro.N@10=0.0000` for the first five epochs.
- The earlier KG formal directory `mooccube_seed2025_full_formal_kg` should also not be used as a final result because its full-catalog CE still penalized strict cold courses as negatives during RS training. Its best checkpoint had exact validation N@10 `0.000006730884`, which was rounded to `0.0000` in the old four-decimal log.

Formal single-seed KG-joint command:

```powershell
.\py.bat paper_aaai27\scripts\pcgnn_strict_adapter.py `
  --max-train-examples -1 `
  --max-val-examples -1 `
  --max-test-examples -1 `
  --epochs 20 `
  --early-stop-patience 5 `
  --train-batch-size 32 `
  --eval-batch-size 64 `
  --kg-batch-size 256 `
  --kg-loss-weight 1.0 `
  --rs-candidate-mode warm `
  --device cuda `
  --out-dir paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2025_full_formal_kg_warm `
  --checkpoint-dir paper_aaai27\baseline_sources\_pcgnn_strict\mooccube_seed2025_full_formal_kg_warm\checkpoints
```

## Concrete Priority Experiment Summary

Latest report:

`paper_aaai27/baseline_sources/_priority_experiments/mooccube_seed2025/priority_baseline_experiments.md`

| Priority | Candidate | Concrete experiment | Status | Main finding |
|---:|---|---|---|---|
| 1 | PCGNN | one-step loss + strict-history full-sort smoke | ok | loss=6.5364, 32 eval cases, median target rank=325.5 |
| 2 | UPGPR | strict split path-reachability proxy | ok | validation reachability=0.750 over 4 history-bearing cases; test reachability=0.778 over 9 history-bearing cases |
| 3 | MSEC-Rec | matrix density and dependency gate | blocked_by_dependency | matrices exist, but current environment lacks DGL and the official sampled evaluator still needs replacement |
| 4 | KGAN | official loader on adapted smoke data | ok_loader_only | loader accepts data, but official random split remains protocol-unsafe |

Concrete conclusion:

1. PCGNN is still the highest-priority baseline for a real table because it now has an external strict adapter for training and full-catalog item-macro evaluation. It still needs an uncapped multi-epoch run before results can be reported.
2. UPGPR is the best second candidate for an explainable/path-reasoning baseline, but the reachability proxy shows only a small number of cold validation/test cases have train-history users in this capped smoke. The model needs full-catalog scoring over missing-path items before any Recall/NDCG comparison.
3. MSEC-Rec has the right course-specific matrices but is gated by DGL and evaluator replacement. It should wait until PCGNN/UPGPR are settled.
4. KGAN should remain backup only because its official loader destroys the external split through random train/test splitting.

## Additional Literature Candidates Not Selected as Runnable Baselines Yet

| Paper/model | Evidence found | Code status | Recommendation |
|---|---|---|---|
| EduGraph: Learning Path-Based Hypergraph Neural Networks for MOOC Course Recommendation | IEEE TBD 2024 paper found in literature search | no official code found in GitHub searches | cite as related course-recommendation literature, not a main baseline now |
| HGNN: Hyperedge-based graph neural network for MOOC Course Recommendation | IP&M 2022 paper found in literature search | no official code found in GitHub searches | relevant but older; do not prioritize over PCGNN/MSEC/UPGPR |
| Knowledge-aware sequence modelling with deep learning for online course recommendation | IP&M 2023 paper found in literature search | no official code found in GitHub searches | cite if needed; not a runnable baseline now |
| Bilateral knowledge graph enhanced online course recommendation | Information Systems 2022 paper found in literature search | no official code found in GitHub searches | older and code-unverified; lower priority |
| LE-DLCM: Decoupled learner and course modeling with large language models for enhanced course recommendation | KBS 2025 paper found in literature search | no verified official code found | good recent citation for LLM course recommendation, but too risky as a main baseline without code |
| DRG: dual relational graph / LLM course recommendation direction | GitHub repository pulled | repository lacks complete training/evaluation pipeline | cite/discuss only unless authors release runnable code |

## Recommendation

1. Prioritize PCGNN as the next new course-specific baseline. It is recent, directly about course recommendation, has a sizeable released code package, and now has an external strict adapter that avoids the official dataloader's validation/test collapse. The next step is an uncapped multi-epoch run.
2. Use UPGPR as the second priority if we want an explainable/path-reasoning MOOC baseline. The exported strict files are accepted by the official `Dataset` and `KnowledgeGraph` readers, but dependency isolation and evaluator replacement are required.
3. Use MSEC-Rec as the third priority. It is the newest course-specific paper among pulled candidates and has concise PyTorch/DGL source, but its released evaluation protocol is sampled-ranking and its feature matrices require more preprocessing.
4. Keep KGAN as a backup course-specific baseline. It has a loader smoke pass, but it is older and depends on TensorFlow 1.x/Keras.
5. Do not use IDRMI as a main-table baseline unless we patch and verify the official graph loader.
6. Do not use DRG, LE-DLCM, EduGraph, HGNN, or BKGE as runnable baselines yet unless official code or a reproducible implementation is obtained.

For AAAI main-paper space, the most defensible next experiment is: add PCGNN on MOOCCube first, then decide whether to include UPGPR or MSEC-Rec in the supplement depending on run time. A main-table addition should only be made after the strict split, full-catalog ranking, train-history masking, and item-macro aggregation are verified.
