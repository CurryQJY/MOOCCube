# LLM-Enhanced Knowledge and Learning Path Understanding for Graph-based Educational Recommendation
This repository contains the official implementation of **KLU4EduRec**, a dual-branch educational recommendation framework that integrates graph-based structural modeling with LLM-enhanced learning path understanding.

## Environments
The code is implemented and tested with the following environment:
* python=3.9.23
* numpy=1.26.4
* pandas=1.5.3
* tqdm=4.67.1
* transformers=4.44.0
* torch=2.4.0+cu121
* torch_scatter=2.1.2+pt24cu121
* torch_sparse=0.6.18+pt24cu121

## Code Structure
```
KLU4EduRec
├── config
│   ├── mooccube.json
│   └── mooccubex.json
├── data
│   ├── mooccube
│   ├── mooccubex
│   │   ├── all.txt
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   ├── test.txt
│   │   ├── resource_info.json
│   │   ├── resource_summary_embeddings_qwen3.pt
│   │   ├── resource_summary_output.json
│   │   ├── resource_summary_prompt.json
│   │   ├── user_behavior_seq_info_train.txt
│   │   ├── user_summary_embeddings_qwen3_knowseg.pt
│   │   ├── user_summary_output_knowseg_train.json
│   └─  └── user_behavior_seq_info_train_segment_75%.json
├── llm_response&emb
│   ├── get_resource_prompt.py
│   ├── get_resource_summary.py
│   ├── get_text_emb_by_qwen3.py
│   ├── get_user_segment_summary.py
│   └── semantic_segment.py
├── model
│   ├── GNNRec.py
│   ├── dataset.py
│   ├── early_stops.py
│   ├── metric.py
│   └── utils.py
├── readme.md
└── main.py
```

## Dataset
We provide processed versions of **MOOCCubeX** and **MOOCCube** to reproduce the experimental results. To facilitate direct evaluation, the released data include:
* Pre-extracted **resource knowledge summaries**
* Pre-extracted **segmented learning path summaries**
* Corresponding **semantic embeddings**

So, you can skip the LLM-based semantic extraction step defined in *Train and Test* and directly run the model.

Dataset statistics are as follows：
| Dataset | MOOCCubeX | MOOCCube |
| - | - | - |
| Students | 9,506 | 9,831 |
| Resources | 7,219 | 11,885 |
| Graph Edges | 659,714 | 935,943 |
| Graph Density | 0.96% | 0.80% |
| Avg. Learning path Length | 69.40 | 95.20 |

## Train and Test
### Step 1: Unzip Data
Please download the processed dataset from Google Drive (https://drive.google.com/file/d/1XP62s2c9L3MFlDfmCmUqG9ymP7Yd3w9X/view?usp=sharing). After downloading, unzip the file in the project root directory:
```
unzip data.zip
```

### Step 2 (Optional): LLM-based Semantic Extraction
If you would like to reproduce the semantic extraction pipeline from scratch, follow the steps below.

**Step 2.1**: Resource knowledge summarization
```
cd llm_response&emb
python get_resource_summary.py
python get_text_emb_by_qwen3.py
```

**Step 2.2**: Learning path segmentation via pattern drift
```
python semantic_segment.py
```

**Step 2.3**: Segment-level searning path summarization
```
python get_user_segment_summary.py
python get_text_emb_by_qwen3.py
```

### Step 3: Run the Model
You can directly run the model using the provided semantic representations:
```
python main.py -d mooccubex
```
Recommended hyperparameters for both datasets are specified in the `config/` directory.


