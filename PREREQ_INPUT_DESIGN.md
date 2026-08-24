# 先修特征输入注入（路B-input）设计方案

## 背景与动机（一句话）
主表冷课表示 = 纯 BERT 内容投影（`content_proj(item_con_emb)`，768→128），
**无任何先修/行为信息进入冷课输入**。ID dropout(0.35)已在训练期制造"无id_e"伪冷样本，
但它们只喂内容特征。所以：把先修行为聚合 μ_pre 作为**额外输入特征**注入，
让共享通路在训练期（伪冷样本上）学会利用先修结构——这是唯一没被 dropout 覆盖、
也未被此前尝试（改reward/改target/加损失/pseudo-cold）触碰的注入点。

与路A的本质区别：路A是训练好后推理期硬改 rollout target（触顶，N@10仅+1.4%，
受重排序级天花板限制）；路B-input 让模型**训练时就学会用先修特征**，先修信息第一次有训练梯度。

## 关键结构事实（已查实）
- `item_con_emb`: 768维 BERT（冻结, freeze=True）
- `content_proj`: Linear768→256, ..., →128（4层MLP）
- `item_id_emb`: 698×128，**行为空间**；μ_pre = 先修课的 item_id_emb 质心，也在这个128维行为空间
- `gate_net`: 输入256(id_e‖content_e)→128，输出融合门 alpha
- `item_fused = alpha·id_e + (1-alpha)·content_e`（冷课 id_e=0 → item_fused=(1-alpha)·content_e）
- 先修索引已有: `outputs/prereq_target/prereq_index_topk10.pt`，440/698课有先修集合(has_prereq)

## 核心设计选择：μ_pre 在128行为空间，content输入在768 BERT空间——注入位置二选一

### 方案X（推荐）：作为与 content_e 并列的第三分支，喂进 gate 融合
不动 content_proj。新增一个先修分支 `prereq_e = prereq_proj(μ_pre)`（128→128小MLP），
把冷课表示改为三路融合：
```
content_e = content_proj(BERT)                    # 原内容，128
prereq_e  = prereq_proj(mu_pre)                    # 新，先修行为聚合投影，128；无先修则0
# 门控扩展为3路（或两级）：
item_fused = alpha_id·id_e + alpha_con·content_e + alpha_pre·prereq_e
```
- 冷课(id_e=0): item_fused = alpha_con·content_e + alpha_pre·prereq_e —— 先修行为信息**直接进表示**
- 只对 has_prereq 的课激活 prereq_e（无先修→prereq_e=0，退化回原式）
- gate 学习"何时信任先修 vs 内容"，防止先修盖过BERT
- **梯度**：主ranking loss → item_fused → prereq_proj + gate，训练期在伪冷样本(dropout清零id的has_prereq课)上有真实梯度

**为什么优于把μ_pre拼进content_proj输入**：μ_pre是128行为空间、BERT是768语义空间，
硬拼进768输入需先升维、且混淆两种异质信号；并列分支+独立投影更干净，
且gate能显式学融合权重（可解释、可消融）。

### 方案Y（备选）：μ_pre拼进content_proj输入
`content_proj(concat[BERT_768, mu_pre_128]) `,输入维度768→896。
缺点：改了content_proj第一层维度→**主表checkpoint无法直接加载**（形状不匹配），
必须从头训，且异质信号早融合难学。**不推荐**。

## 防退化/公平性约束（关键，避免重蹈attempt2/5双通路失败）
1. **只对 has_prereq 冷课激活**：prereq_e = prereq_proj(μ_pre) · has_prereq_mask；无先修课严格退化为原主表式（保证不伤这部分）
2. **μ_pre 用 detach 的 item_id_emb 质心**：先修课的行为嵌入作为"教师信号"，不让先修分支的梯度反向污染热课id_emb的学习（避免循环）
3. **小初始化 + gate 抑制**：prereq_proj 末层小初始化，alpha_pre 初始接近0，让模型从"几乎不用先修"平滑地学到"该用多少"——防止训练初期先修噪声压过已训好的BERT路
4. **train-only μ_pre**：先修课质心只用训练可见的先修课ID（本就都是热课，无泄漏）

## 训练期如何让先修获得梯度（回应"冷课held-out"）
冷课训练期held-out，但 **ID dropout(0.35) 会把 has_prereq 的热课随机清零id_e → 变成"伪冷+有先修"样本**。
这些样本走 item_fused = alpha_con·content_e + alpha_pre·prereq_e，
主ranking loss 对它们的梯度**同时训练 prereq_proj 和 gate**。
这就是先修信息第一次有训练梯度的机制——且是 dropout 已在制造的样本，不需新开 pseudo-cold。

## 实现落点（新增开关，默认关闭，不碰主表复现）
- `__init__`：读 `USIM_PREREQ_INPUT` 开关；构造 `prereq_proj`（128→128 MLP）；加载 prereq_index（复用现有产物）
- `get_item_vector`（~1028-1032融合处）：开关开启时，算 prereq_e 并改3路融合；gate_net 输入需从256扩到384(id‖content‖prereq)或用两级门控（见下）
- gate 扩展需谨慎：若扩gate输入维度→主表checkpoint的gate_net.0(128×256)形状变→**无法直接加载**。
  **解法**：用**两级门控**——先原gate融合(id,content)得 base_fused，再第二个小gate融合(base_fused, prereq_e)。
  这样原gate_net形状不变，可加载主表权重，只新增第二级gate + prereq_proj。**这是关键实现约束。**

## 验证方案
1. 烟雾：开关开启，epochs=1，确认 prereq_e 分支进loss、has_prereq课激活、无先修课退化、可加载主表gate权重
2. 单seed：主表配置+USIM_PREREQ_INPUT=1，60ep，对比主表 seed2025 (Cold R@10=0.2732 或 candidate 0.2939同checkpoint对照)
3. 判据：Cold 显著超基线（非千分位）且Hot不崩、无先修课不退化 → 有效，扩3seed
4. 消融：prereq_proj权重、alpha_pre初值、top-k(先修课数)

## 诚实风险标注
- 这次改的是**输入内容**（喂进先修特征），理论上能突破路A的重排序天花板——因为先修信息进了表示本身、有训练梯度。但**能否显著提升仍未知**，取决于μ_pre(先修行为质心)是否真携带BERT没有的、且对冷课排序有用的信号。
- 双通路历史上失败过（attempt2/5），但那些是"冷/热走不同通路"导致目标冲突；本方案是**同通路加一路输入特征、gate自适应融合、只对has_prereq激活**，机制不同，但仍需警惕先修分支被gate学成"永远关闭"（那就退化回主表=无害但无增益）。
