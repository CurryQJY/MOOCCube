# Dataset Processing Protocol

This project uses one FAST3-compatible processed format for both MOOC course
recommendation datasets and educational exercise recommendation datasets.

## Processed Directory Contract

Each dataset processor writes:

| File | Purpose |
| --- | --- |
| `stream_data.pkl` | Interaction stream with `user_id`, `course_id`, `timestamp`, `u_idx`, `i_idx`, and `popularity`. For EdNet/Junyi, `course_id` is the generic item id. |
| `content_emb.pt` | One content vector per encoded item, ordered by `i_idx`. |
| `meta.json` | Dataset statistics and embedding metadata. |
| `_item_id_map.csv` | Mapping from `i_idx` to raw item id. |
| `relations/course-concept.json` | TSV-compatible item-concept pairs consumed by course-aware artifacts. |
| `relations/prerequisite-dependency.json` | TSV-compatible concept-prerequisite pairs when available. |
| `entities/course.json` | JSONL item metadata for compatibility with existing course artifact code. |

The downstream model still expects the generic field name `course_id`, so the
exercise id / question id is intentionally stored there.

## Content Embedding Protocol

For the three main course recommendation datasets, content embeddings should
use BERT `[CLS]` representations with the same encoder:

```text
embedding_backend: bert_cls
embedding_model: bert-base-chinese
embedding_max_length: 256
content_dim: 768
```

The current MOOCCourse processed directory follows this protocol and records it
in `processed_data_mooccourse/meta.json`. The MOOCCube and MOOCCubeX processing
scripts also use BERT `[CLS]` embeddings; their scripts now write the encoder
metadata when rerun.

For EdNet-KT1 and Junyi, `stable_hash` embeddings are acceptable only for
generalization/smoke-test experiments. Do not describe those embeddings as BERT
unless they are regenerated with `--embedding-backend bert_cls`.

## MOOCCubeX

Current FAST3 experiments use the processed directory and relation directory:

```powershell
$env:USIM_DATA_DIR = "processed_data_hin_x"
$env:USIM_RELATION_DIR = "MOOCCubeX\relations"
$env:USIM_PREREQ_GRAPH_SOURCE = "concept"
```

The minimum files needed by the current course-aware pipeline are present:

```text
processed_data_hin_x/stream_data.pkl
processed_data_hin_x/content_emb.pt
processed_data_hin_x/meta.json
MOOCCubeX/entities/course.json
MOOCCubeX/relations/course-concept.json
MOOCCubeX/relations/prerequisite-dependency.json
```

The original MOOCCubeX prerequisite files are JSONL files under:

```text
MOOCCubeX/prerequisites/cs.json
MOOCCubeX/prerequisites/math.json
MOOCCubeX/prerequisites/psy.json
```

They store raw concept names in `c1` and `c2`. Convert them to the local
`K_<concept>_<domain>` concept ids consumed by `build_course_artifacts`:

```powershell
.\py.bat -B convert_mooccubex_prerequisites.py --base-dir MOOCCubeX --output MOOCCubeX\relations\prerequisite-dependency.json
```

The conservative paper protocol keeps only `ground_truth == 1` pairs. On the
current local data, this yields 1,355 concept-level prerequisite pairs:

| File | Ground-truth pairs | Converted pairs |
| --- | ---: | ---: |
| `cs.json` | 725 | 483 |
| `math.json` | 761 | 240 |
| `psy.json` | 831 | 632 |

The converted file has been validated with `prereq_graph_source="concept"`:
1,355 concept pairs become 2,849 raw item-level candidate edges, 782 kept
item-level prerequisite edges, and 161 items with prerequisite evidence.

## MOOCCourse

MOOCCourse is the XuetangX course recommendation dataset used by the AAAI 2019
course recommendation paper. The official MoocData page names it "Course
Recommendation"; recent course recommendation papers often refer to this data
as MOOCCourse.

Raw files are under:

```text
data_raw/MOOCCourse/mooc_data/data.csv
data_raw/MOOCCourse/mooc_data/Data/mooc.train.rating
data_raw/MOOCCourse/mooc_data/Data/mooc.test.rating
data_raw/MOOCCourse/mooc_data/Data/mooc.all.rating
data_raw/MOOCCourse/mooc_data/Data/mooc.test.negative
```

`data.csv` is encoded as GB18030 and contains:

```text
stu_id,time,course_index,name,type,type_id
```

Run:

```powershell
.\py.bat -B data_process_mooccourse.py --raw-dir data_raw\MOOCCourse --output-dir processed_data_mooccourse
```

Current local processed statistics:

| Metric | Value |
| --- | ---: |
| Users | 82,535 |
| Courses | 1,302 |
| Interactions | 458,453 |
| Course category concept edges | 2,934 |
| Content dimension | 768 |
| Content embedding backend | BERT `[CLS]`, `bert-base-chinese` |

MOOCCourse does not provide an official concept-prerequisite graph. Use
behavior-derived prerequisites:

```powershell
$env:USIM_DATA_DIR = "processed_data_mooccourse"
$env:USIM_RELATION_DIR = "processed_data_mooccourse\relations"
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"
```

Validation with `prereq_graph_source="behavior"` and `prereq_min_support=30`
currently yields 213,783 raw behavior prerequisite candidates, 1,852 kept
item-level prerequisite edges, and 453 items with prerequisite evidence.

## COCO

COCO is a semantic-enriched online course dataset. The current local copy uses
the public KG-preprocessed repository:

```text
data_raw/COCO_Educational_Recommendation_Dataset/
```

Required public files:

```text
preprocessed/ratings.txt
preprocessed/i2kg_map.txt
preprocessed/e_map.txt
preprocessed/r_map.txt
preprocessed/kg_final.txt
```

Run:

```powershell
.\py.bat -B data_process_coco.py --raw-dir data_raw\COCO_Educational_Recommendation_Dataset --output-dir processed_data_coco --concept-scope conservative --embedding-backend bert_cls --embedding-model bert-base-uncased --embedding-batch-size 64
```

Current local processed statistics:

| Metric | Value |
| --- | ---: |
| Users | 24,036 |
| Courses | 8,196 |
| Interactions | 378,469 |
| Course metadata concept edges | 27,827 |
| Items with metadata concept | 8,196 |
| Content dimension | 768 |
| Content embedding backend | BERT `[CLS]`, `bert-base-uncased` |

The conservative relation scope exports `belong_to_category` and
`related_to_concept` as `relations/course-concept.json`. Course text for
`content_emb.pt` additionally includes level, language, and target-audience
labels when present. The public KG does not provide official prerequisite
relations, so `relations/prerequisite-dependency.json` is intentionally empty.
Use behavior-derived prerequisites for FAST3:

```powershell
$env:USIM_DATA_DIR = "processed_data_coco"
$env:USIM_RELATION_DIR = "processed_data_coco\relations"
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"
```

Single-seed go/no-go triage:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_coco_single_seed_triage.ps1
```

## EdNet-KT1

Place raw files under:

```powershell
data_raw\ednet_kt1\
```

Supported layouts include:

```text
data_raw/ednet_kt1/KT1/*.csv
data_raw/ednet_kt1/KT1/train_data/*.csv
data_raw/ednet_kt1/contents/questions.csv
```

Run:

```powershell
.\py.bat data_process_ednet_kt1.py --raw-dir data_raw\ednet_kt1 --output-dir processed_data_ednet_kt1
```

For a quick parser smoke test:

```powershell
.\py.bat data_process_ednet_kt1.py --raw-dir data_raw\ednet_kt1 --output-dir processed_data_ednet_kt1_smoke --max-users 100 --max-rows 200000
```

EdNet-KT1 usually has no explicit prerequisite graph. Use behavior-derived
dependencies for FAST3:

```powershell
$env:USIM_DATA_DIR = "processed_data_ednet_kt1"
$env:USIM_RELATION_DIR = "processed_data_ednet_kt1\relations"
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"
```

## Junyi

Place raw files under:

```powershell
data_raw\junyi\
```

Supported interaction file names include:

```text
Log_Problem.csv
junyi_ProblemLog_original.csv
ProblemLog.csv
interactions.csv
```

Supported metadata file names include:

```text
Info_Content.csv
junyi_Exercise_table.csv
exercise_table.csv
contents.csv
```

Run:

```powershell
.\py.bat data_process_junyi.py --raw-dir data_raw\junyi --output-dir processed_data_junyi
```

If the metadata contains prerequisite columns, the processor exports
concept-level prerequisite edges. Otherwise use behavior-derived dependencies:

```powershell
$env:USIM_DATA_DIR = "processed_data_junyi"
$env:USIM_RELATION_DIR = "processed_data_junyi\relations"
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"
```

## Paper Protocol Note

Use MOOCCube/MOOCCubeX as the main course recommendation datasets. Treat
EdNet-KT1 and Junyi as educational item recommendation generalization datasets,
because their items are questions/exercises rather than courses.
