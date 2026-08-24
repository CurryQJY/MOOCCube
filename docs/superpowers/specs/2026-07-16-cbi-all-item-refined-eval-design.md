# CBI 全课程模拟评估设计

## 目标

在不重新训练、不修改主表训练代码、不覆盖现有 CBI 结果的前提下，加载 CBI seed-2025 的 Epoch 33 最佳 checkpoint，将测试阶段的 5 步 deterministic USIM 模拟从“仅严格冷课程”扩展到全部课程，重新计算 cold/hot item-macro full-ranking 指标。

本实验只回答一个问题：训练阶段所有目标课程都经过模拟，而原测试阶段只有严格冷课程经过模拟；若测试阶段让热课程也经过同一模拟，Cold 和 Hot 指标如何变化。

## 隔离边界

- 不修改 `usim_feedback_fast3_content_delta.py`。
- 不修改 `fast3_delta/eval.py`。
- 不修改主表、论文表格、现有输出或 checkpoint。
- 新增独立评估入口和对应测试。
- 新结果写入 `outputs/cbi_faithful_seed2025_eval_all_refined/`。
- 输入 checkpoint 固定为 `checkpoints/cbi_faithful_single_seed2025/strict_item_cold_balanced_thr1_seed_2025/finished.pt` 中的 `es_best_state`，其最佳轮次为 Epoch 33。

## 评估语义

1. 从原 CBI 协议清单恢复模型配置、数据路径和静态切分。
2. 加载 Epoch 33 最佳模型状态，不加载或更新优化器。
3. 根据训练流行度区分课程：
   - 严格冷课程使用 `force_cold=True`；
   - 热课程使用 `force_cold=False`。
4. 两组课程都调用 `infer_refined_item_vectors`，执行相同的 5 步 deterministic rollout，且测试阶段均保持 `target_emb=None`。
5. 将两组结果合并为单一、归一化的 all-refined item bank。
6. cold 和 hot full-ranking 都复用该 bank；正样本向量必须直接索引同一 bank，不能再次走旧的“冷模拟、热不模拟”正样本路径。
7. 评估过程只读模型参数，不执行 `backward()` 或 `optimizer.step()`。

## 输出

输出目录包含：

- `all_refined_fullrank.csv`：Cold/Hot 的 R@5、R@10、R@20、N@5、N@10、N@20。
- `all_refined_manifest.json`：checkpoint、最佳 epoch、配置、输入文件哈希、运行时、评估语义和耗时。
- `comparison.md`：与原“仅冷课程模拟”结果的绝对差值和相对变化。
- `evaluation.log`：完整运行日志。

## 测试策略

遵循测试驱动开发：

1. 先新增失败测试，构造含冷、热课程的最小假模型，断言生成的 bank 中冷、热行都来自模拟输出。
2. 断言 cold/hot 正样本均直接复用缓存后的 all-refined bank。
3. 断言评估结束后模型训练状态恢复，且参数未被修改。
4. 实现最小隔离评估辅助函数，使上述测试通过。
5. 运行现有 CBI 测试和新增测试，确认共享评估行为未被改变。

## 风险与处理

- **显存风险**：课程只有 698 个，但用户 bank 较大；沿用原推理的分批接口并只缓存一个最终 item bank。
- **正样本不一致风险**：不复用原 `build_eval_pos_item_vecs` 的热课程分支，强制从统一 bank 索引。
- **工作区污染风险**：只新增文件，提交时显式指定新增文件，不纳入现有未提交修改。
- **结果归因限制**：这是 checkpoint replay，只能衡量测试时对称模拟的影响，不能替代重新训练后的严格消融。

## 成功标准

- 全部 698 个课程均经过 deterministic simulator。
- cold/hot 正样本和候选项来自同一个 all-refined bank。
- 原 checkpoint 与现有 CBI 输出保持不变。
- 新增测试及相关回归测试通过。
- 独立目录中生成完整指标、日志和可复现清单。
