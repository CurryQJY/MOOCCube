# MOOCCube MOOC Recommendation

本项目面向 MOOCCube / MOOCCubeX 在线课程数据，构建了一个用于 MOOC 个性化课程推荐与冷启动优化的实验系统。项目重点解决用户行为稀疏、冷启动用户难推荐、热门课程偏置，以及传统推荐模型难以利用课程概念、教师、学校、先修关系等知识结构的问题。

仓库保留了完整的科研实验迭代过程，因此根目录中包含多轮模型版本、基线脚本、日志、结果表和消融实验文件。为了便于评审或复现，建议优先查看本 README 中标出的最终入口和结果文件。

## 核心思路

项目以 USIM 强化学习推荐框架为基础，将课程推荐过程建模为序列决策问题，并在原有用户行为建模之外加入课程侧知识增强信号：

- 课程语义表示：使用 768 维课程内容向量表征课程文本、概念和元信息。
- 异构信息增强：融合课程、教师、学校、概念、用户交互和课程关系等多源信息。
- LLM 语义反馈：引入 LLM 评分作为课程语义适配信号，辅助冷启动推荐。
- 学习路径约束：将先修关系、概念覆盖、难度适配和冗余惩罚转化为奖励函数或重排序策略。
- 冷/热用户分组评估：分别衡量冷启动用户和行为充分用户上的推荐效果。

本项目不采用多 Agent 协作，也不是通用大模型长链推理系统。项目中的“推理”主要体现在将课程学习路径中的先修顺序、知识概念关联、课程难度和内容冗余，转化为可计算的推荐奖励与排序逻辑。

## 主要成果

实验覆盖约 19.9 万用户和 698 门课程，课程内容向量维度为 768。课程侧消融实验中包含 3418 名冷启动用户和 67 万级热用户样本。

| 模块 | 主要效果 |
| --- | --- |
| Concept Reward | 冷启动 Sampled R@5 提升 5.51%，Full Cold R@20 提升 5.98% |
| Prereq Reward | Full Hot R@5 提升 22.93%，Full Hot N@5 提升 23.57% |
| Difficulty Reward | Full Hot R@5 提升 15.27%，Full Hot N@5 提升 14.72% |
| Redundant Penalty | Full Hot R@20 提升 2.17%，对课程内容重复有抑制作用 |
| All Course 组合 | 对冷启动整体指标有提升，但存在冷/热用户权衡 |

主要结论是：课程概念奖励更适合提升冷启动用户推荐效果，先修关系奖励对热用户的学习路径排序收益最明显，说明知识结构增强推荐在 MOOC 场景中具有实际价值。

## 推荐阅读顺序

如果只是评审或快速了解项目，建议按下面顺序查看：

1. `README.md`：项目总览、核心成果和运行入口。
2. `docs/PROJECT_STRUCTURE.md`：仓库结构说明，解释根目录中各类脚本的用途。
3. `docs/SUBMISSION_EVIDENCE.md`：评审材料和截图上传建议。
4. `outputs/usim_feedback_fast3_course_ablation/ablation_report.md`：课程侧消融实验报告。
5. `mooc_metrics_usim_feedback_fast3_summary.csv`：FAST3 主实验冷/热用户指标汇总。
6. `mooc_result_usim_feedback_fast3.png`：主实验可视化结果图。

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `data_process_hin.py` | 基于 MOOCCubeX 构建课程异构信息网络和训练数据 |
| `usim_feedback_fast3.py` | USIM-Feedback FAST3 主实验模型 |
| `usim_feedback_fast3_standalone.py` | 最终独立版实验入口，减少对旧版本脚本的依赖 |
| `run_usim_feedback_fast3_course_ablation.ps1` | 课程侧奖励模块消融实验脚本 |
| `run_usim_feedback_fast3_standalone_redundant_compare.ps1` | 独立版冗余惩罚对比实验脚本 |
| `hin_data_common.py` | HIN 数据加载与通用处理逻辑 |
| `hin_eval_common.py` | 冷/热用户指标评估工具 |
| `llm_rescore_hin_clean.py` | LLM 评分对齐与清洗相关脚本 |
| `outputs/usim_feedback_fast3_course_ablation/ablation_report.md` | 主要消融实验结果报告 |

## 运行方式

本地环境使用 Python / PyTorch / pandas / numpy / scikit-learn / matplotlib / tqdm 等常用机器学习依赖。仓库中提供了 `py.bat`，默认指向本机 Anaconda 环境。

数据处理：

```powershell
.\py.bat data_process_hin.py
```

运行 FAST3 课程侧消融实验：

```powershell
.\run_usim_feedback_fast3_course_ablation.ps1
```

运行最终独立版实验：

```powershell
$env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
.\py.bat usim_feedback_fast3_standalone.py
```

说明：原始 MOOCCube / MOOCCubeX 数据、处理后的特征、模型权重和 checkpoint 体积较大，不建议全部提交到 GitHub。公开仓库中可以保留核心代码、关键结果表、结果图和实验报告。

## 结果文件

| 文件 | 内容 |
| --- | --- |
| `mooc_metrics_usim_feedback_fast3_summary.csv` | FAST3 sampled / full-rank 指标汇总 |
| `final_report_usim_feedback.csv` | USIM-Feedback 冷/热用户详细指标 |
| `final_fullrank_usim_original_reconstructed_standalone.csv` | 原始 USIM 独立重构版全量排序结果 |
| `drop_static_result.json` | DropoutNet 静态基线结果 |
| `mooc_result_usim_feedback_fast3.png` | FAST3 实验曲线图 |
| `outputs/usim_feedback_fast3_course_ablation/ablation_report.md` | 消融实验完整报告 |

## 仓库整理说明

当前仓库刻意保留了多轮实验痕迹，包括 baseline、USIM 原始重构、FAST / FAST2 / FAST3、课程侧奖励、冗余惩罚、LLM 评分对齐等版本。这样便于回溯实验过程，但会让根目录看起来比较密集。

对外展示时，核心版本以 `usim_feedback_fast3.py` 和 `usim_feedback_fast3_standalone.py` 为准；旧版本脚本主要用于对比、消融和结果复核。更详细的结构说明见 `docs/PROJECT_STRUCTURE.md`。

