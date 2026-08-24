# CBI-Constrained Simulator 单 Seed 修正实验设计

## 目标

在不修改主表代码和既有 CBI 输出的前提下，构建一个独立的 CBI+模拟扩展实验，修复当前实现中的两个核心问题：

1. 训练模拟不再使用 `item_id_embedding` 作为 `target_emb`。
2. 每一步模拟更新后，将课程表示投影回冻结内容表示周围的角度信任域。

实验固定使用 MOOCCube、seed 2025、原 CBI 的数据切分和其余超参数，从头训练一次。结果写入独立输出与 checkpoint 目录，不进入主表。

## 方法定位

该方案不是原始 Content-Based Initialization 的严格复现，而是 **CBI-constrained simulator** 扩展。原始 CBI 约束静态课程表示；本扩展允许模拟器动态调整课程表示，但要求整个模拟轨迹始终处于内容语义信任域内。

## 表示与约束

设冻结并单位归一化的内容表示为 \(c_i\)，有界 CBI 修正为 \(d_i\)，模拟前表示为

\[
q_i^{(0)}=\operatorname{normalize}(c_i+d_i),
\qquad \lVert d_i\rVert_2\le \delta_{\max}=0.5.
\]

第 \(t\) 步模拟先产生未约束更新

\[
\tilde q_i^{(t+1)}=q_i^{(t)}+\eta g_t.
\]

随后投影至角度信任域

\[
\mathcal C_i=\left\{q:\lVert q\rVert_2=1,\ c_i^\top q\ge \tau\right\},
\qquad
\tau=\sqrt{1-\delta_{\max}^2}=\sqrt{0.75}\approx0.866025.
\]

最终更新为

\[
q_i^{(t+1)}=\Pi_{\mathcal C_i}\left(\tilde q_i^{(t+1)}\right).
\]

若归一化后的候选表示已满足 \(c_i^\top q\ge\tau\)，直接保留；否则将其投影到圆锥边界：保留相对 \(c_i\) 的正交方向，将平行分量设为 \(\tau\)，正交分量范数设为 \(\sqrt{1-\tau^2}\)。若正交分量数值退化，则回退为 \(c_i\)。

由此每一步及最终表示都满足

\[
\cos(c_i,q_i^{(t)})\ge0.866025.
\]

## 模拟锚点修正

当前训练调用传入 `target_emb=id_e_true.detach()`，而冷测试传入 `target_emb=None`。修正后，训练和测试统一使用

```text
target_emb = q_i^(0).detach()
```

即以模拟前的 CBI 表示作为轨迹稳定锚点：

- 热课程锚点为 `normalize(content + bounded_delta)`；
- 严格冷课程的 delta 为零，锚点为纯内容表示；
- 不再使用课程 ID embedding 指导模拟；
- 原有 target-alignment reward、step-gain reward 和 target-alpha 机制仍可工作，但其目标从协同 ID 空间改为 CBI 空间。

投影中心与奖励锚点承担不同职责：

- 投影中心 \(c_i\) 提供不可突破的内容语义边界；
- 奖励锚点 \(q_i^{(0)}\) 防止模拟产生没有收益的轨迹漂移。

## 训练与测试一致性

- 训练中的所有课程均经过受约束模拟。
- 验证和测试中的严格冷、热课程均经过同一受约束模拟。
- 冷课程使用 `force_cold=True`，热课程使用 `force_cold=False`。
- 正样本与 full-ranking 候选项复用同一个 all-refined item bank。
- 训练、验证和测试均使用相同的 CBI 锚点和投影规则。

## 隔离实现

不修改以下共享文件：

- `usim_feedback_fast3_content_delta.py`
- `fast3_delta/eval.py`
- `run_fast3_main_table_config.ps1`
- 主表输出、checkpoint 和论文表格

新增独立模块，通过子类与进程内局部适配实现：

1. 子类覆盖 `run_usim_episode`，忽略父调用传入的 ID target，统一使用初始 CBI 表示，并在每步更新后执行角度投影。
2. 独立评估适配器生成 all-refined item bank，并保证正样本直接索引同一 bank。
3. 独立入口在进程内将静态实验使用的模型类替换为受约束子类，不改变共享模块源文件。

## 固定实验配置

- Dataset: MOOCCube
- Protocol: `strict_item_cold_balanced`
- Seed: 2025
- Epochs: 60
- Patience: 60
- CBI delta max norm: 0.5
- Trust cosine floor: \(\sqrt{0.75}\)
- USIM steps: 5
- Rollout policy: PPO
- Refined evaluation: cold and hot both enabled
- 其余 simulator、PPO、课程奖励、KG sampler、先修辅助损失和学习率配置与已完成 CBI 实验一致。

为保证单因素解释，本轮不额外关闭 ID-content auxiliary loss、不调整 delta 正则、不更换 gate，也不修改 delta 范数。相关问题在本实验结果之后单独处理。

## 输出与复现

- Output root: `outputs/cbi_trust_sim_single_seed2025`
- Checkpoint root: `checkpoints/cbi_trust_sim_single_seed2025`
- Log root: `background_logs/cbi_trust_sim_single_seed2025`
- 独立 manifest 记录源文件哈希、配置、运行时、最佳 epoch、信任域统计和参数摘要。
- 每个 epoch 记录投影前违规比例、投影触发比例、最终最小/平均内容余弦相似度。
- 最终报告同时给出 Cold/Hot item-macro full-ranking 指标，并与原 CBI 和原 seed-2025 主表模型比较。

## 测试策略

遵循测试驱动开发：

1. 角度投影对域内向量保持不变。
2. 域外向量投影后单位范数且余弦不低于 \(\tau\)。
3. 退化共线输入稳定回退，无 NaN。
4. 模拟器忽略父调用传入的 ID target，实际使用初始 CBI 锚点。
5. 每一步状态均满足内容余弦下界。
6. all-refined 评估同时模拟冷、热课程并复用统一 bank。
7. 原共享模块和主表文件哈希在实验前后保持一致。

## 判定标准

该实验首先验证结构修正是否有效，而不是要求立即超过主表：

- 必须消除 ID target 的训练/测试不一致。
- 所有最终课程表示必须满足内容余弦下界。
- Hot R@10/N@10 应明显优于当前全课程无约束模拟结果 `0.1273/0.0730`，且 Cold 指标不应出现灾难性下降。
- 若 Hot 仍未恢复，则后续优先审查 delta 饱和和无效 ID auxiliary 分支，而不再归因于模拟轨迹漂移。
