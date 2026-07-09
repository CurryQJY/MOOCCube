# Course Baseline Adaptability Experiment

Split: `D:\DeskTop\MOOCCube\outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2025`
Output: `D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_adaptability\mooccube_seed2025_smoke`

## Dependency Gate

| Package | Status |
|---|---|
| torch | OK |
| dgl | MISSING |
| recbole | OK |
| easydict | OK |
| wandb | OK |
| numpy | OK |
| pandas | OK |

## Priority Assessment

| Priority | Candidate | Smoke artifact | Loader/dependency status | Protocol fit | Recommendation |
|---:|---|---|---|---|---|
| 1 | PCGNN | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\PCGNN_recbole_drive\RecBole-master\dataset\mooccube_strict_seed2025_smoke | atomic files load; stock sequential dataloader needs patch for external strict valid/test histories | medium-high: data split is preservable, official build/evaluator is not protocol-safe yet | adapt first, but patch loader/build before training |
| 2 | UPGPR | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_adaptability\mooccube_seed2025_smoke\upgpr\processed_files | processed files exported; easydict=OK, wandb=OK; NumPy 2 needs alias shim or older env | medium-high for data; path evaluator must be replaced | adapt second in a clean dependency env |
| 3 | MSEC-Rec | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_adaptability\mooccube_seed2025_smoke\msec\data | matrix export ready; dgl=MISSING | medium; sampled evaluator and DGL graph need changes | adapt after PCGNN/UPGPR |
| 4 | KGAN | D:\DeskTop\MOOCCube\paper_aaai27\baseline_sources\_prepared\mooccube_seed2025\kgan | loader smoke passes on adapted data; tensorflow/keras missing from py.bat env for training | medium; original loader randomly splits ratings | backup only |

## Key Takeaways

- PCGNN remains the best next runnable course-specific baseline because its RecBole atomic format can preserve an external strict split.
- PCGNN's files load in the modified RecBole tree, but the stock sequential build path still needs a protocol patch; otherwise validation/test sequence construction can collapse under external item-cold splits.
- UPGPR is highly relevant and its Dataset/KnowledgeGraph reader accepts the exported files, but requires replacing path-only top-10 evaluation with full-catalog scoring.
- MSEC-Rec is recent and course-specific, but current environment lacks DGL and its released evaluator is sampled-ranking.
- KGAN remains a backup; the existing adapter smoke passes, but TF1/Keras and random official splitting make it less attractive.
