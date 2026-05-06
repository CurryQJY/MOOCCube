# Project Structure Guide

本文件用于解释当前仓库结构，帮助评审或协作者快速找到最终成果。仓库目前是科研实验工作区形态，保留了多轮模型版本、基线、日志和结果文件；因此不是典型的精简工程目录。

## 最终主线

建议把项目主线理解为：

```text
MOOCCube / MOOCCubeX raw data
        |
        v
data_process_hin.py
        |
        v
processed_data_hin_clean_pop5
        |
        v
usim_feedback_fast3.py / usim_feedback_fast3_standalone.py
        |
        v
outputs/usim_feedback_fast3_course_ablation
        |
        v
ablation_report.md + csv/png results
```

## 目录与文件类别

| 路径或文件 | 类别 | 说明 |
| --- | --- | --- |
| `MOOCCube/`, `MOOCCubeX/` | 原始数据 | MOOCCube 数据集及关系文件，本地运行需要，公开仓库可不完整上传 |
| `processed_data*` | 处理中间数据 | 存放 stream data、content embedding、LLM scores 等训练输入 |
| `usim*.py` | 主模型和实验版本 | USIM 系列模型，包含原始、课程增强、反馈增强和独立重构版本 |
| `*_hin.py` | HIN 版本 | 使用异构信息网络特征的模型或基线 |
| `run_*.ps1` | 实验脚本 | PowerShell 批量实验入口，负责设置环境变量并运行多组实验 |
| `outputs/` | 实验输出 | 消融实验报告、CSV 汇总、结果图等 |
| `checkpoints/` | 训练检查点 | 本地训练恢复文件，体积较大，不建议上传 |
| `*.csv`, `*.json`, `*.png` | 结果文件 | 推荐指标、图表和基线对比结果 |
| `*.log`, `*.pt`, `*.pth` | 日志与权重 | 运行日志、模型权重和 checkpoint，通常不适合直接提交 |

## 核心代码入口

| 文件 | 推荐程度 | 说明 |
| --- | --- | --- |
| `usim_feedback_fast3_standalone.py` | 最推荐 | 最终独立版 FAST3 实验入口，包含模型、数据、评估和课程图逻辑 |
| `usim_feedback_fast3.py` | 推荐 | FAST3 主实验版本，适合查看主要训练逻辑 |
| `data_process_hin.py` | 推荐 | HIN 数据构建入口 |
| `run_usim_feedback_fast3_course_ablation.ps1` | 推荐 | 课程侧奖励消融实验入口 |
| `hin_data_common.py` | 辅助 | HIN 数据公共逻辑 |
| `hin_eval_common.py` | 辅助 | 评估公共逻辑 |

## 旧版本脚本说明

根目录中存在 `fast`, `fast2`, `fast3`, `legacy`, `reconstructed`, `decouple`, `content_delta`, `seq` 等命名。这些文件主要对应不同实验阶段：

- `legacy`：早期或原始方法复现。
- `reconstructed`：为了脱离旧依赖重建的可运行版本。
- `fast`, `fast2`, `fast3`：训练效率、候选采样和奖励稳定性逐步优化的版本。
- `decouple`：将奖励项拆开做独立分析的版本。
- `content_delta`：侧重课程内容差异或语义变化的实验版本。
- `standalone`：用于最终展示和复核的独立版本。

## 建议的后续重构方向

如果后续要把仓库整理成更标准的开源项目，可以逐步调整为：

```text
src/
  data/
  models/
  evaluation/
  experiments/
scripts/
  run_ablation.ps1
  run_baselines.ps1
docs/
outputs/
tests/
```

目前没有直接移动文件，是为了避免破坏已有实验脚本中的相对路径。当前改进重点是通过 README 和 docs 让仓库可读、可评审、可证明。

