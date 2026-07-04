# MOOCCourse Dataset Audit

## Source

- Official page: `http://moocdata.cn/data/course-recommendation`
- Downloaded archive: `data_raw/MOOCCourse/course_recommendation.rar`
- Extracted directory: `data_raw/MOOCCourse/mooc_data`
- Reference paper: *Hierarchical Reinforcement Learning for Course Recommendation in MOOCs*, AAAI 2019.

The MoocData page names the dataset "Course Recommendation". Recent course
recommendation papers often refer to this XuetangX course recommendation
dataset as MOOCCourse.

## Files

```text
mooc_data/data.csv
mooc_data/Readme
mooc_data/Data/mooc.all.rating
mooc_data/Data/mooc.train.rating
mooc_data/Data/mooc.test.rating
mooc_data/Data/mooc.test.negative
```

`data.csv` is encoded as GB18030. Fields:

| Field | Meaning |
| --- | --- |
| `stu_id` | Anonymized student id. |
| `time` | First enrollment time for the course. |
| `course_index` | Course id. |
| `name` | Course name. |
| `type` | Course category text. |
| `type_id` | Course category id. |

The `Data/` directory contains processed implicit-feedback files used by the
AAAI 2019 recommendation setup.

## Local Statistics

| Metric | Value |
| --- | ---: |
| Interactions | 458,453 |
| Users | 82,535 |
| Courses | 1,302 |
| Density | 0.4266% |
| Time range | 2016-10-01 to 2018-03-30 |
| Duplicate user-course pairs | 0 |
| Course categories | 23 |

Sequence statistics:

| Metric | Value |
| --- | ---: |
| Min interactions/user | 3 |
| Median interactions/user | 4 |
| Mean interactions/user | 5.555 |
| Max interactions/user | 398 |
| Users with >= 5 interactions | 35,760 |
| Median interactions/course | 24.5 |
| Max interactions/course | 11,058 |
| Courses with >= 10 interactions | 770 |

Processed rating files:

| File | Rows | Users | Items |
| --- | ---: | ---: | ---: |
| `mooc.train.rating` | 411,109 | 82,535 | 1,109 |
| `mooc.test.rating` | 47,344 | 19,067 | 853 |
| `mooc.all.rating` | 458,453 | 82,535 | 1,302 |
| `mooc.test.negative` | 47,344 | - | 100 columns |

## Suitability

MOOCCourse is suitable as the third main course recommendation dataset because:

- It is a real MOOC course-enrollment dataset rather than an exercise dataset.
- It is tied to a top-conference course recommendation paper.
- It includes explicit enrollment timestamps, allowing chronological splits and
  cold-start protocol construction.
- It has course names and coarse course categories, enabling lightweight content
  embeddings and category-as-concept fallback relations.

Limitations:

- It does not provide an official concept graph or prerequisite graph.
- Course metadata is much lighter than MOOCCube/MOOCCubeX.
- The provided train/test split is interaction-level implicit feedback; for
  cold-start experiments, we should build our own item/user cold protocol from
  `data.csv`.

## Recommended Protocol

Use MOOCCourse as the third main dataset with behavior-derived or hybrid
prerequisite edges:

```powershell
$env:USIM_DATA_DIR = "processed_data_mooccourse"
$env:USIM_RELATION_DIR = "processed_data_mooccourse\relations"
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"
```

For FAST3 compatibility, process `data.csv` into:

```text
processed_data_mooccourse/stream_data.pkl
processed_data_mooccourse/content_emb.pt
processed_data_mooccourse/meta.json
processed_data_mooccourse/_item_id_map.csv
processed_data_mooccourse/entities/course.json
processed_data_mooccourse/relations/course-concept.json
processed_data_mooccourse/relations/prerequisite-dependency.json
```

Use course categories as coarse concepts. Content embeddings are generated with
the same BERT `[CLS]` protocol used for the main MOOC course datasets:

```text
embedding_backend = bert_cls
embedding_model = bert-base-chinese
embedding_max_length = 256
content_dim = 768
```

MOOCCourse has no explicit prerequisites, so keep
`relations/prerequisite-dependency.json` empty and derive behavioral
dependencies during artifact construction.

## Main-Table Runner

The main-table model set in `output/doc/final_narrow_topconf/tables_narrow_topconf.docx`
is:

```text
Popularity, BPR, LightGCN, DropoutNet, ContentProfile, CCFCRec, ALDI, CGRC, Ours
```

Run the MOOCCourse version with:

```powershell
.\run_main_table_mooccourse.ps1
```

For a one-seed, one-epoch compatibility run:

```powershell
.\run_main_table_mooccourse.ps1 -Smoke -Seeds @(2025)
```

This runner first creates the MOOCCourse `strict_item_cold_balanced` split via
the FAST3 static runner, then makes all baselines reuse the same
`USIM_STATIC_SPLIT_DIR`. It sets:

```powershell
$env:USIM_DATA_DIR = "processed_data_mooccourse"
$env:USIM_RELATION_DIR = "processed_data_mooccourse\relations"
$env:USIM_PREREQ_GRAPH_SOURCE = "behavior"
```

The split protocol has been smoke-tested locally with seed 2025:

| Metric | Value |
| --- | ---: |
| Eligible cold items | 917 |
| Validation cold items | 46 |
| Test cold items | 92 |
| Train rows | 311,875 |
| Validation rows | 61,792 |
| Test rows | 84,666 |
