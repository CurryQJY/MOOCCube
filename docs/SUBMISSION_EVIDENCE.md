# Submission Evidence Guide

这个文件用于准备评审系统中“使用证明与影响力证明”一栏的上传材料。建议把材料整理成一张长截图、一个 PDF，或者多个截图组合上传。

## 最推荐上传的材料

1. 项目 GitHub 链接

   `https://github.com/CurryQJY/MOOCCube`

2. 项目首页截图

   截图中最好能看到仓库名称、README、核心成果和主要结果表。

3. 项目目录截图

   建议截到这些文件：

   - `README.md`
   - `usim_feedback_fast3_standalone.py`
   - `data_process_hin.py`
   - `run_usim_feedback_fast3_course_ablation.ps1`
   - `outputs/usim_feedback_fast3_course_ablation/ablation_report.md`

4. 终端运行截图

   推荐展示以下命令之一：

   ```powershell
   .\run_usim_feedback_fast3_course_ablation.ps1
   ```

   或：

   ```powershell
   $env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
   .\py.bat usim_feedback_fast3_standalone.py
   ```

5. 结果指标截图

   推荐截这些文件：

   - `outputs/usim_feedback_fast3_course_ablation/ablation_report.md`
   - `mooc_metrics_usim_feedback_fast3_summary.csv`
   - `final_report_usim_feedback.csv`
   - `mooc_result_usim_feedback_fast3.png`

## 建议展示的指标

评审更容易看懂下面这些结论：

| 结论 | 指标 |
| --- | --- |
| 概念奖励提升冷启动推荐 | Sampled Cold R@5 +5.51%，Full Cold R@20 +5.98% |
| 先修关系提升热用户推荐 | Full Hot R@5 +22.93%，Full Hot N@5 +23.57% |
| 难度奖励对热用户有效 | Full Hot R@5 +15.27%，Full Hot N@5 +14.72% |
| 课程结构信号存在冷/热权衡 | All Course 对冷启动更有利，但可能降低热用户部分指标 |

## 上传时可以写的说明

可以直接粘贴下面这段：

```text
本仓库保留了完整实验迭代过程，核心实现见 usim_feedback_fast3_standalone.py，主要课程侧消融结果见 outputs/usim_feedback_fast3_course_ablation/ablation_report.md。项目基于 MOOCCube/MOOCCubeX 数据构建 MOOC 个性化课程推荐系统，融合课程概念、先修关系、难度适配、冗余惩罚和 LLM 语义评分反馈，用于优化冷启动和学习路径推荐效果。
```

## 隐私与体积注意事项

- 如果上传 AI 平台账单或用量截图，请打码邮箱、手机号、API Key、订单号和余额。
- 不建议上传完整模型权重、checkpoint、原始数据集或大体积日志。
- 如果只能上传一张图，建议拼接：GitHub 首页 + 项目目录 + 终端运行 + 消融报告指标。

