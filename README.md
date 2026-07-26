# MOOCCube MOOC Recommendation

本项目面向 MOOCCube / MOOCCubeX 在线课程数据，构建了一个用于 MOOC 个性化课程推荐与课程冷启动优化的实验系统。项目核心目标是解决训练阶段几乎没有交互记录的新课程推荐问题，并验证课程概念、先修关系、难度和内容冗余等知识结构能否提升严格课程冷启动场景下的排序效果。

仓库保留了完整的科研实验迭代过程，因此根目录中包含多轮模型版本、基线脚本、日志、结果表和消融实验文件。为了便于评审或复现，建议优先查看本 README 中标出的最终入口和结果文件。

## 核心思路

项目以 USIM 强化学习推荐框架为基础，将课程推荐过程建模为序列决策问题，并在原有用户行为建模之外加入课程侧知识增强信号：

- 课程语义表示：使用 768 维课程内容向量表征课程文本、概念和元信息。
- 异构信息增强：融合课程、教师、学校、概念、用户交互和课程关系等多源信息。
- LLM 语义反馈：引入 LLM 评分作为课程语义适配信号，辅助冷启动推荐。
- 学习路径约束：将先修关系、概念覆盖、难度适配和冗余惩罚转化为奖励函数或重排序策略。
- 冷启动主评估：以 strict item-cold、full-ranking、item-macro 为主口径，热启动指标作为不严重退化的辅助参考。

本项目不采用多 Agent 协作，也不是通用大模型长链推理系统。项目中的“推理”主要体现在将课程学习路径中的先修顺序、知识概念关联、课程难度和内容冗余，转化为可计算的推荐奖励与排序逻辑。

## 主要成果

实验覆盖约 19.9 万用户和 698 门课程，课程内容向量维度为 768。最终目标口径聚焦严格课程冷启动：测试集冷启动课程在训练集中无交互，主指标采用 full-ranking item-macro R@K / N@K，避免采样负例和热门课程交互数量对结论的影响。

| 模块 | 主要效果 |
| --- | --- |
| FAST3 cold-start main | 3 seeds 下 Cold item-macro R@10 / N@10 达到 0.2667 / 0.1962 |
| Strong cold-start baseline | CGRC-paper Cold item-macro R@10 / N@10 为 0.2589 / 0.1845 |
| Concept / course reward | 主要用于提升新课程语义匹配和学习路径合理性 |
| Prereq / difficulty / redundancy | 作为课程结构约束，解释冷启动推荐中的可学习先修顺序、难度适配和内容重复抑制 |
| Hot metrics | 作为辅助稳定性检查，不作为本项目主胜负口径 |

主要结论是：FAST3 在严格课程冷启动 full-ranking item-macro 协议下优于现有强冷启动基线；课程知识结构增强推荐在 MOOC 新课程推荐场景中具有实际价值。热启动指标用于说明模型没有只靠牺牲已知课程排序来换取冷启动提升。

## 推荐阅读顺序

如果只是评审或快速了解项目，建议按下面顺序查看：

1. `README.md`：项目总览、核心成果和运行入口。
2. `docs/PROJECT_STRUCTURE.md`：仓库结构说明，解释根目录中各类脚本的用途。
3. `docs/SUBMISSION_EVIDENCE.md`：评审材料和截图上传建议。
4. `outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_multiseed_summary.csv`：FAST3 严格课程冷启动三种子主结果。
5. `outputs/content_delta_pop5/static_item_cold_balanced/main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120_cgrc_paper/main_table_item_macro_summary.csv`：主要基线与 SOTA 的 item-macro 对比表。
6. `outputs/usim_feedback_fast3_course_ablation/ablation_report.md`：课程侧奖励消融实验报告。

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `data_process_hin.py` | 基于 MOOCCubeX 构建课程异构信息网络和训练数据 |
| `usim_feedback_fast3_content_delta.py` | 当前课程冷启动主实验入口，支持 strict item-cold 静态协议和 full-ranking item-macro 评估 |
| `run_usim_feedback_fast3_content_delta_static.ps1` | FAST3 课程冷启动静态协议批量运行脚本 |
| `usim_feedback_fast3.py` | USIM-Feedback FAST3 流式主实验版本 |
| `usim_feedback_fast3_standalone.py` | 独立版实验入口，保留用于复核旧主线 |
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

运行 FAST3 严格课程冷启动静态实验：

```powershell
.\run_usim_feedback_fast3_content_delta_static.ps1 -Protocol strict_item_cold_balanced -ColdThresholds 1 -Seeds 2025,2026,2027 -Epochs 60 -EarlyStopAverageMode item_macro
```

运行旧版 FAST3 课程侧消融实验：

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
| `outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_multiseed_summary.csv` | FAST3 严格课程冷启动三种子 full-ranking item-macro 主结果 |
| `outputs/content_delta_pop5/static_item_cold_balanced/main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120_cgrc_paper/main_table_item_macro_summary.csv` | FAST3 与主要 baseline / SOTA 的 item-macro 对比 |
| `mooc_metrics_usim_feedback_fast3_summary.csv` | 旧版 FAST3 sampled / full-rank 指标汇总 |
| `final_report_usim_feedback.csv` | USIM-Feedback 冷/热用户详细指标 |
| `final_fullrank_usim_original_reconstructed_standalone.csv` | 原始 USIM 独立重构版全量排序结果 |
| `drop_static_result.json` | DropoutNet 静态基线结果 |
| `mooc_result_usim_feedback_fast3.png` | FAST3 实验曲线图 |
| `outputs/usim_feedback_fast3_course_ablation/ablation_report.md` | 消融实验完整报告 |

## 仓库整理说明

当前仓库刻意保留了多轮实验痕迹，包括 baseline、USIM 原始重构、FAST / FAST2 / FAST3、课程侧奖励、冗余惩罚、LLM 评分对齐等版本。这样便于回溯实验过程，但会让根目录看起来比较密集。

对外展示时，课程冷启动主线以 `usim_feedback_fast3_content_delta.py` 和 `run_usim_feedback_fast3_content_delta_static.ps1` 为准；旧版 FAST3 脚本主要用于流式实验、对比、消融和结果复核。更详细的结构说明见 `docs/PROJECT_STRUCTURE.md`。
