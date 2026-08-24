# Submission Evidence Guide

这个文件用于准备评审系统中“使用证明与影响力证明”一栏的上传材料。建议把材料整理成一张长截图、一个 PDF，或者多个截图组合上传。

## 最推荐上传的材料

1. 项目 GitHub 链接

   `https://github.com/CurryQJY/MOOCCube`

2. 项目首页截图

   截图中最好能看到仓库名称、README、严格课程冷启动目标和主要结果表。

3. 项目目录截图

   建议截到这些文件：

   - `README.md`
   - `usim_feedback_fast3_content_delta.py`
   - `data_process_hin.py`
   - `run_usim_feedback_fast3_content_delta_static.ps1`
   - `outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_multiseed_summary.csv`

4. 终端运行截图

   推荐展示以下命令之一：

   ```powershell
   .\run_usim_feedback_fast3_content_delta_static.ps1 -Protocol strict_item_cold_balanced -ColdThresholds 1 -Seeds 2025,2026,2027 -Epochs 60 -EarlyStopAverageMode item_macro
   ```

   或：

   ```powershell
   $env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
   .\py.bat usim_feedback_fast3_standalone.py
   ```

5. 结果指标截图

   推荐截这些文件：

   - `outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_multiseed_summary.csv`
   - `outputs/content_delta_pop5/static_item_cold_balanced/main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120_cgrc_paper/main_table_item_macro_summary.csv`
   - `outputs/usim_feedback_fast3_course_ablation/ablation_report.md`

## 建议展示的指标

评审更容易看懂下面这些冷启动结论：

| 结论 | 指标 |
| --- | --- |
| FAST3 超过强冷启动基线 | 3 seeds 下 Cold item-macro R@10 / N@10 = 0.2667 / 0.1962 |
| CGRC-paper 作为主要 SOTA 对比 | 3 seeds 下 Cold item-macro R@10 / N@10 = 0.2589 / 0.1845 |
| 主评估协议更严格 | strict item-cold，测试冷启动课程在训练集中无交互 |
| 热启动指标定位 | 仅作为辅助稳定性检查，不作为项目主胜负目标 |

## 上传时可以写的说明

可以直接粘贴下面这段：

```text
本仓库保留了完整实验迭代过程，核心实现见 usim_feedback_fast3_content_delta.py，最终主结果见 outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_multiseed_summary.csv。项目基于 MOOCCube/MOOCCubeX 数据构建 MOOC 课程冷启动推荐系统，融合课程概念、先修关系、难度适配、冗余惩罚和课程内容表征，在 strict item-cold、full-ranking、item-macro 协议下验证 FAST3 对新课程推荐效果的提升。
```

## 隐私与体积注意事项

- 如果上传 AI 平台账单或用量截图，请打码邮箱、手机号、API Key、订单号和余额。
- 不建议上传完整模型权重、checkpoint、原始数据集或大体积日志。
- 如果只能上传一张图，建议拼接：GitHub 首页 + 项目目录 + 终端运行 + 消融报告指标。
