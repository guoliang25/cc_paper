# MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

> **Authors**: Xu Huang, Hao Zhang, Zhifang Fan, Yunwen Huang, Zhuoxing Wei, Zheng Chai, Jinan Ni, Yuchao Zheng, Qiwei Chen
> **Affiliation**: ByteDance（字节跳动）
> **Venue**: arXiv preprint, February 2026
> **Links**: [arXiv](https://arxiv.org/abs/2602.14110)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 工业推荐系统中，序列行为建模（sequence modeling）和特征交互建模（feature interaction）通常由**独立的模块**分别处理，再以串行（stacked）或并行（parallel）方式拼接。这种"分而治之"的架构在 scaling 时面临根本矛盾：给序列模块加参数还是给特征交互模块加参数？两者的计算预算如何分配？且分离架构难以在两者之间建立深层联合表示。
- **Method (方法)**: 提出 **MixFormer**，一种将序列建模和特征交互**统一到单一 Transformer backbone** 中的架构。核心设计包括三个组件：(1) **Query Mixer** 用无参数 HeadMixing 替代 self-attention 实现特征交互；(2) **Cross-Attention** 让特征表示作为 query 去聚合行为序列信息；(3) **Output Fusion** 深度融合序列与非序列信号。同时提出 **User-Item Decoupling** 策略，通过方向性 mask 实现 request-level batching，大幅降低推理开销。
- **Results (效果)**: 在抖音数据上，MixFormer-medium（1,226M 参数）以仅 2,242 GFLOPs 达到 Finish AUC +1.28%、Skip AUC +1.60%，超越 STCA→RankMixer（+1.12%/+1.43%，6,736 GFLOPs）的同时 FLOPs 减少约 67%。线上 A/B 测试在抖音上活跃天数 +0.04%、时长 +0.28%、评论 +0.70%。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

![MixFormer 整体架构](https://arxiv.org/html/2602.14110v1/x1.png)

MixFormer 的整体流程：

- **输入**: 两类特征——(1) 非序列特征（用户画像、物品属性、上下文等），经 embedding 后拼接并分为 $N$ 个 head；(2) 行为序列（用户历史交互序列），经序列 embedding 编码
- **主要模块**: $L$ 层堆叠的 **MixFormer Block**，每个 block 包含三个组件：Query Mixer → Cross-Attention → Output Fusion
- **输出**: 融合后的特征送入 task-specific heads 进行多任务预测（完播率、跳过率等）

**关键创新**: 不再将序列模型和特征交互模型分开设计，而是在每一层中同时进行两种建模，让序列信号和特征信号在每一层都深度交融。

### 2.2 Core Modules (核心模块详解)

**模块一：Feature Embedding & Split（特征嵌入与分割）**

- **功能**: 将异构非序列特征统一编码并分割为多个 head
- **非序列特征拼接**:

$$\mathbf{e}_{ns} = [\mathbf{e}_1; \mathbf{e}_2; \ldots; \mathbf{e}_M] \in \mathbb{R}^{D_{ns}}, \quad D_{ns} = \sum_{i=1}^{M} d_i$$

- $M$: 非序列特征域的数量
- $d_i$: 第 $i$ 个特征域的 embedding 维度

- **Head 投影**:

$$\mathbf{x}_j = \mathbf{W}_j \cdot \mathbf{e}_{ns}[d \cdot (j-1) : d \cdot j], \quad j = 1, \ldots, N$$

- $\mathbf{W}_j \in \mathbb{R}^{D \times d}$: 第 $j$ 个 head 的投影矩阵
- $\mathbf{x}_j \in \mathbb{R}^D$: 第 $j$ 个 head 的表示
- **直觉理解**: 把拼接后的特征切成 $N$ 份，每份通过独立的线性变换投影到统一维度，形成 $N$ 个"特征 query"，后续每个 query 将独立地去序列中检索信息

**模块二：Query Mixer（查询混合器）**

- **功能**: 在不使用 self-attention 的前提下实现跨 head 的特征交互
- **HeadMixing 操作**: 无参数的跨 head 信息交换

$$\mathbf{P} = [\mathbf{p}_1, \ldots, \mathbf{p}_N] = \text{HeadMixing}(\text{Norm}(\mathbf{X})) + \mathbf{X}$$

- HeadMixing 的具体操作：将 $\mathbf{X} \in \mathbb{R}^{N \times D}$ reshape 为 $\mathbb{R}^{N \times N \times D/N}$，转置前两个维度，再 flatten 回 $\mathbb{R}^{N \times D}$
- **直觉理解**: 这本质上就是 RankMixer 中的 multi-head token mixing——将每个 head 的 embedding 切成 $N$ 份小片段，然后在 head 之间"洗牌"重组，实现无参数的跨特征信息交换

- **Per-head SwiGLU FFN**:

$$\mathbf{q}_i = \text{SwiGLUFFN}_i(\text{Norm}(\mathbf{p}_i)) + \mathbf{p}_i$$

- 每个 head 拥有独立的 SwiGLU FFN，建模特征异质性
- **直觉理解**: HeadMixing 负责"让不同特征交流"，Per-head FFN 负责"让每个特征用自己的方式消化交流后的信息"

**模块三：Cross-Attention（交叉注意力）**

- **功能**: 让 Query Mixer 的输出作为 query，去行为序列中检索相关信息
- **序列特征变换**（逐层独立）:

$$\mathbf{h}_t = \text{SwiGLUFFN}^{(l)}(\text{Norm}(\mathbf{s}_t)) + \mathbf{s}_t \in \mathbb{R}^{ND}$$

$$\mathbf{h}_t^i = \mathbf{h}_t[iD : (i+1)D] \in \mathbb{R}^D$$

$$\mathbf{k}_t^i = \mathbf{W}_k^i \mathbf{h}_t^i, \quad \mathbf{v}_t^i = \mathbf{W}_v^i \mathbf{h}_t^i$$

- 每层对序列做独立的 SwiGLU 变换（而非共享），使不同层能从序列中提取不同粒度的信息
- 序列 embedding 按 head 切分，每个 head 有独立的 $K, V$ 投影

- **注意力聚合**:

$$\mathbf{z}_i = \sum_{t=1}^{T} \text{softmax}\left(\frac{\mathbf{q}_i^T \mathbf{k}_t^i}{\sqrt{D}}\right) \mathbf{v}_t^i + \mathbf{q}_i, \quad i = 1, \ldots, N$$

- $T$: 行为序列长度
- $\mathbf{q}_i$: 来自 Query Mixer 的第 $i$ 个 head，作为注意力的 query
- **直觉理解**: 每个"特征 query"根据自身的语义，去用户行为序列中寻找最相关的历史行为，比如"用户画像 head"可能关注整体偏好，"物品属性 head"可能关注与当前物品相似的历史交互。这比传统的"先做序列建模，再把序列表示拼接到特征中"更加精细

**模块四：Output Fusion（输出融合）**

- **功能**: 深度融合来自序列和非序列的信号

$$\mathbf{o}_i = \text{SwiGLUFFN}_i(\text{Norm}(\mathbf{z}_i)) + \mathbf{z}_i$$

- 使用 per-head SwiGLU FFN（而非共享），因为每个 head 融合的信号类型不同，需要独立的非线性变换
- **直觉理解**: Cross-Attention 的输出同时包含了"原始特征信息"（通过 skip connection）和"序列检索到的信息"，Output Fusion 负责让这两种信号深度交融

**模块五：User-Item Decoupling（用户-物品解耦）**

![User-Item 解耦架构](https://arxiv.org/html/2602.14110v1/x2.png)

- **功能**: 在保持模型效果的前提下，通过 request-level batching 大幅降低推理开销
- **核心思路**: 将 $N$ 个 head 划分为 $N_U$ 个 user-side heads 和 $N_G$ 个 item-side heads。同一个请求中，user-side 的计算对所有候选物品共享，只需算一次

- **方向性 Mask**:

$$\mathcal{M}[i, j] = \begin{cases} 0 & \text{if } i < N_U \text{ and } j \geq N_U \cdot D / N \\ 1 & \text{otherwise} \end{cases}$$

$$\text{HeadMixing}_{decouple}(\cdot) = \mathcal{M} \odot \text{HeadMixing}(\cdot)$$

- Mask 阻止 item 信号泄露到 user-side heads（保证 user-side 计算的 request-level 可复用性）
- 但允许 user 信号流向 item-side heads（保持用户-物品的单向信息融合）
- **直觉理解**: "用户的表示不能受物品影响（否则每换一个候选物品都要重算用户表示），但物品的表示可以利用用户信息"。通过这个简单的 mask，实现了 ~36% FLOPs 减少和 >30% 推理加速

### 2.3 Training Strategy (训练策略)

- **优化器**: Dense 部分使用 RMSProp（lr=0.01），Sparse 部分使用 Adagrad
- **归一化**: Pre-RMSNorm（在每个子模块前做归一化）
- **模型配置**:
  - MixFormer-small: $D=386, N=16, L=4$
  - MixFormer-medium: $D=768, N=16, L=4$
- **Batch size**: 1,500
- **Loss Function**: 标准多任务 CTR/CVR loss（论文未强调新 loss 设计）

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **数据集**: 字节跳动内部抖音推荐数据（大规模工业数据集）
- **评价指标**: Finish AUC、Skip AUC、Finish UAUC、Skip UAUC
- **主要 Baselines**（三种架构范式）:
  - **Stacked（串行）**: TA→DLRM, TA→DCNv2, TA→DHEN, TA→Wukong, TA→RankMixer, STCA→RankMixer
  - **Parallel（并行）**: OneTrans, STCA⊕RankMixer
  - **Unified（统一）**: MixFormer（本文）

### 3.2 Main Results (主实验结果)

**离线对比（Table 1）**:

| 架构类型 | 模型 | Finish AUC | Skip AUC | 参数量 | GFLOPs |
|---------|------|-----------|---------|--------|--------|
| Stacked | TA→DLRM | baseline | baseline | 9M | 52 |
| Stacked | TA→Wukong | +0.29% | +0.49% | 122M | 442 |
| Stacked | TA→RankMixer | +0.95% | +1.25% | 1,118M | 2,180 |
| Stacked | STCA→RankMixer | +1.12% | +1.43% | 1,255M | 6,736 |
| Parallel | OneTrans | +1.05% | +1.30% | 316M | 23,371 |
| Parallel | STCA⊕RankMixer | +1.11% | +1.42% | 1,255M | 6,736 |
| **Unified** | **MixFormer-medium** | **+1.28%** | **+1.60%** | **1,226M** | **3,503** |
| **Unified** | **UI-MixFormer-medium** | **+1.28%** | **+1.60%** | **1,226M** | **2,242** |

关键观察：
- MixFormer-medium 在所有指标上取得最优，Finish AUC +1.28% vs STCA→RankMixer 的 +1.12%，提升显著
- **计算效率优势巨大**: 与性能最接近的 STCA→RankMixer（6,736 GFLOPs）相比，UI-MixFormer 仅需 2,242 GFLOPs，**计算量减少 67%**
- 并行架构 OneTrans 虽效果尚可，但 FLOPs 高达 23,371，完全不可部署
- 统一架构在"效果-效率"帕累托前沿上全面优于串行和并行架构

### 3.3 Ablation Study (消融实验)

![消融实验结果](https://arxiv.org/html/2602.14110v1/x3.png)

关键发现：
- **去掉 HeadMixing**: 性能显著下降，说明跨 head 的特征交互是不可或缺的
- **HeadMixing → Self-Attention**: 性能未提升，反而增加计算成本，再次验证了无参数 HeadMixing 的性价比优势
- **Per-layer FFN（Cross-Attention 中）**: 逐层独立的序列 FFN 带来可观增益，说明不同层需要从序列中提取不同粒度的信息
- **Per-head FFN（Output Fusion 中）**: 逐 head 独立的融合 FFN 进一步提升效果，验证了特征异质性建模的重要性
- **Pre-RMSNorm vs Post-LayerNorm**: 差异较小，但 Pre-RMSNorm 略优

### 3.4 Scaling Analysis (扩展性分析)

**Dense Scaling（固定序列长度 512，Figure 4）**:

![FLOPs 维度的 Scaling 分析](https://arxiv.org/html/2602.14110v1/x4.png)

- MixFormer 在 FLOPs scaling 曲线上取得更大的 intercept（初始优势）和有竞争力的 slope（扩展速率）
- 相比仅做序列建模或仅做特征交互的模型，统一架构在同等计算预算下效果更好

**Sequence Length Scaling（Figure 5）**:

![序列长度维度的 Scaling 分析](https://arxiv.org/html/2602.14110v1/x5.png)

- 在序列长度从 512 扩展到 10,000 时，MixFormer 的 scaling slope 与 SOTA 序列模型相当
- 同时保持了更高的基线性能，说明统一架构没有牺牲序列建模能力

### 3.5 Online Results (线上实验)

**抖音 A/B 测试（Table 2）**:

| 指标 | 抖音 | 抖音极速版 |
|------|------|----------|
| 活跃天数 | +0.0415% | +0.0252% |
| 使用时长 | +0.2799% | +0.4105% |
| 点赞 | +0.1766% | +0.2125% |
| 完播 | +0.3897% | +0.2924% |
| 评论 | **+0.7035%** | **+1.9097%** |

- 评论指标提升最显著（抖音极速版 +1.91%），说明更精准的推荐能有效激发用户互动
- 低活用户收益更大（抖音极速版低活用户点赞 +3.06%），与 RankMixer 的观察一致

**推理延迟（Figure 6）**: UI-MixFormer 通过 request-level batching 实现 >30% 的推理加速，且随候选集增大延迟增长温和

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **"统一优于分离"的架构哲学**: 用一个统一的 Transformer backbone 同时处理序列建模和特征交互，避免了串行/并行架构中"两个模块争抢计算预算"的问题。论文通过系统对比 stacked/parallel/unified 三种范式，清晰展示了统一架构在帕累托前沿上的优势
2. **Cross-Attention 的精妙设计**: 让每个特征 head 作为独立的 query 去序列中检索信息，比传统的"先聚合序列再拼特征"更加精细——不同特征能从同一个序列中提取不同维度的信息
3. **User-Item Decoupling 的工程创新**: 通过一个简单的方向性 mask 实现 user-side 计算共享，FLOPs 减少 36%、延迟降低 30%，且几乎不损失效果。这种"理论上小改动、工程上大收益"的设计非常值得借鉴
4. **Per-layer / Per-head FFN 的细粒度异质性建模**: 在 Cross-Attention 和 Output Fusion 中都使用独立 FFN，消融实验充分验证了其必要性
5. **完整的 Scaling 分析**: 同时展示了 dense scaling（FLOPs 维度）和 sequence length scaling（序列长度维度），证明统一架构在两个方向上都能有效扩展

### 4.2 Limitations & Improvements (不足与改进方向)

1. **HeadMixing 仍是无参数的**: 与 RankMixer 一样，特征交互部分依赖无参数的 HeadMixing。UniMixer 已经证明了参数化 Token Mixing 的优势，将 MixFormer 的 HeadMixing 替换为可学习的混合权重可能带来进一步提升
2. **Cross-Attention 的二次复杂度**: 虽然通过 User-Item Decoupling 降低了整体开销，但 Cross-Attention 本身仍是 $O(T)$ 复杂度（对序列长度线性），当序列长度进一步扩展（如 100K+）时可能成为瓶颈
3. **线上提升幅度相对保守**: 活跃天数 +0.04% 的提升虽然在抖音规模上有商业价值，但相比 RankMixer 的 +0.29% 偏小，可能是因为 baseline 已经包含了 RankMixer 的能力
4. **缺乏公开数据集验证**: 与 RankMixer、UniMixer 一样的局限——全部实验在内部数据上进行
5. **可能的改进方向**:
   - 结合 UniMixer 的参数化 Token Mixing 替换 HeadMixing，在统一框架中引入可学习的特征交互
   - 探索 Flash Attention 或 Linear Attention 加速 Cross-Attention，支持更长序列
   - 将 User-Item Decoupling 的思路扩展到 MoE 场景，实现 user-side expert 共享
   - 探索在统一架构中引入生成式目标（如 next-item prediction），实现推荐系统的统一预训练
