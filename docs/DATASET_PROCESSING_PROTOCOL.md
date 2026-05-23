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
