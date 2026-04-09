# UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems

> **Authors**: Mingming Ha, Guanchen Wang, Linxun Chen, Xuan Rao, Yuexin Shi, Tianbao Ma, Zhaojie Liu, Yunqian Fan, Zilong Lu, Yanan Niu, Han Li, Kun Gai
> **Affiliation**: Kuaishou Technology（快手）
> **Venue**: arXiv preprint, April 2026
> **Links**: [arXiv](https://arxiv.org/abs/2604.00590)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 推荐系统领域缺乏类似 NLP 中 Transformer 那样的"统一基础架构"来支撑 scaling law。现有方法大致分三类——attention-based（如 FAT、HiFormer）、TokenMixer-based（如 RankMixer）和 factorization-machine-based（如 Wukong），但它们各自存在局限：attention 在异构特征上容易产生"尖锐稀疏"的权重导致梯度问题，TokenMixer 的 rule-based 混合缺乏可学习性，FM 方法的 scaling 能力不足。
- **Method (方法)**: 提出 **UniMixer**，通过将 rule-based TokenMixer 等价参数化为可学习矩阵运算，揭示 attention、TokenMixer、FM 三种范式共享统一的特征交互框架，并提出轻量版 **UniMixer-Lite**（基于 Kronecker 分解 + basis-composed 局部混合），大幅降低参数量和计算量。
- **Results (效果)**: 在快手广告投放数据（7亿+ 用户样本）上，UniMixer-Lite-4-Blocks（84.5M 参数）相比 baseline AUC 提升 0.8141%；scaling exponent 达 0.1419，优于 RankMixer 的 0.1160；线上 A/B 测试在多个快手广告场景中平均提升约 15% 的 30 天累计活跃天数。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

![UniMixer 整体架构](https://arxiv.org/html/2604.00590v2/x2.png)

UniMixer 的整体流程：

- **输入**: 推荐系统中的多源异构特征，按语义分组为多个 domain（用户画像、物品特征、行为序列、查询特征等），每个 domain 的特征经 embedding 后通过线性投影统一为 token embedding
- **主要模块**: 多个堆叠的 **UniMixer Block**，每个 block 包含：(1) Unified Token Mixing Module（统一 token 混合模块）进行特征交互；(2) Per-token SwiGLU 进行异质性建模；(3) SiameseNorm 保证深层训练稳定性
- **输出**: 融合后的特征表示，送入预测层输出 CTR/CVR 等目标

### 2.2 Core Modules (核心模块详解)

**模块一：Feature Tokenization（特征 Token 化）**

- **功能**: 将异构稀疏特征统一为同维度的 token 序列
- **输入**: 多个 domain 的原始特征（categorical + numerical），每个 domain 内的特征先拼接为一个 domain embedding
- **输出**: $T$ 个 token，每个维度为 $D$，即 $X \in \mathbb{R}^{T \times D}$
- **直觉理解**: 类似 NLP 中将 word 转为 token embedding，这里是将推荐系统的各类特征统一"说同一种语言"

**模块二：Unified Token Mixing（统一 Token 混合）**

- **功能**: 核心创新模块，将 rule-based 的 TokenMixer 转化为可学习操作，统一 attention/TokenMixer/FM 三种范式
- **关键公式——统一混合框架**:

$$\text{UniMixing}(X) = \text{reshape}\left(G(X, W_G) \cdot [x_1 W_{B}^{1} | \cdots | x_n W_{B}^{n}], 1, L\right)$$

- $G(X, W_G)$: global mixing weights（全局混合权重矩阵），控制 token 间的交互模式
- $W_B^i$: 第 $i$ 个 token 的 local mixing weights（局部混合权重），进行 token 内部的特征变换
- $L$: 总 embedding 维度
- **直觉理解**: 全局混合决定"哪些 token 之间要交互、交互多少"，局部混合决定"每个 token 内部怎么变换特征"，两者组合实现完整的特征交互

**等价参数化的核心发现**:

作者发现 TokenMixer 中的 rule-based 操作（如 permutation、cyclic shift）可以等价表示为置换矩阵（permutation matrix）：

$$W_{perm} = G \otimes I$$

- $G$: 一个小的 permutation 矩阵
- $\otimes$: Kronecker 积
- $I$: 单位矩阵
- **直觉理解**: 这个发现意味着 TokenMixer 的"规则混合"本质上就是一种特殊的矩阵乘法，把它推广为可学习的矩阵就能获得更强的表达能力

![不同方法的全局混合权重对比及 TokenMixer 等价参数化](https://arxiv.org/html/2604.00590v2/x3.png)

**约束条件——保持稳定性**:

全局混合权重 $\bar{W}_G$ 需满足四个约束：
1. **可压缩性（Compressibility）**: 通过 Kronecker 积分解降低计算复杂度，从 $O(T^2 D^2)$ 降至 $O(T^2 D^2 / B + T D B)$
2. **双随机性（Doubly Stochastic）**: 使用 Sinkhorn-Knopp 迭代归一化，保证行列和均为 1，避免信息坍缩
3. **稀疏性（Sparsity）**: 通过温度系数 $\tau$ 控制，低温时权重更稀疏集中
4. **对称性（Symmetry）**: $\bar{W}_G = \bar{W}_G^T$，保证特征交互的对称性

**模块三：UniMixing-Lite（轻量版统一混合）**

- **功能**: 进一步压缩参数和计算量，实现更高效的 scaling
- **全局混合——低秩近似**:

$$W_G^{LR} = U V^T, \quad U, V \in \mathbb{R}^{T \times r}$$

- $r$: 秩（rank），远小于 $T$
- **直觉理解**: 用两个小矩阵的乘积近似大矩阵，大幅减少参数量

- **局部混合——basis-composed 方式**:

$$W_B^{i} = \sum_{k=1}^{K} \alpha_k^i \cdot B_k$$

- $B_k$: 第 $k$ 个 basis 矩阵（所有 token 共享）
- $\alpha_k^i$: 第 $i$ 个 token 对各 basis 的组合系数
- $K$: basis 数量
- **直觉理解**: 不再为每个 token 独立存储局部混合矩阵，而是用少量共享 basis 的加权组合来表示，类似"用几种基础颜色调配出每个 token 专属的颜色"

**模块四：SiameseNorm（孪生归一化）**

- **功能**: 解决深层网络训练不稳定问题
- **设计思路**: Pre-Norm 利于深层训练但退化为恒等映射，Post-Norm 学习能力强但不稳定。SiameseNorm 在每层引入两条耦合流（coupled streams），同时获得两者优点
- **直觉理解**: 类似"一条路保稳定、一条路保表达力"，两条路相互配合使深层网络既能训练又能学到有效特征

**模块五：Per-token SwiGLU**

- **功能**: 对每个 token 独立施加 SwiGLU 激活，建模 token 间的异质性
- **直觉理解**: 不同 domain 的 token（用户特征 vs 物品特征）性质不同，需要独立的非线性变换来处理

### 2.3 Training Strategy (训练策略)

- **Temperature Annealing（温度退火）**:

训练过程中逐步降低温度系数 $\tau$：

$$\tau: 1.0 \rightarrow 0.05$$

- 高温阶段（$\tau=1.0$）：权重分布平滑，模型充分探索特征交互空间
- 低温阶段（$\tau=0.05$）：权重变得稀疏集中，模型聚焦于最重要的交互模式
- **直觉理解**: 先"广撒网"学习所有可能的特征交互，再"精聚焦"到最有效的交互模式

![不同温度下的全局和局部混合权重可视化](https://arxiv.org/html/2604.00590v2/x5.png)

- **Model Warm-up（模型预热）**: 先用高温训练一段时间作为 warm-up，再切换到低温微调，避免直接低温训练导致的次优收敛

- **Loss Function**: 标准的 CTR 预估 loss（论文未特别强调新 loss 设计，重点在架构创新）

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **数据集**: 快手广告投放数据，包含 **7亿+** 用户样本，涵盖多个广告场景
- **评价指标**: AUC 和 UAUC（User-level AUC），以及 ΔAUC（相对 baseline 的提升）
- **主要 Baselines**:
  - **Attention-based**: FAT, HiFormer
  - **TokenMixer-based**: TokenMixer-Large, RankMixer
  - **FM-based**: Wukong

### 3.2 Main Results (主实验结果)

**~100M 参数量模型对比（Table 2）**:

| 模型 | 参数量 | ΔAUC (%) |
|------|--------|----------|
| TokenMixer-Large | 79.7M | +0.2984 |
| FAT | 91.9M | +0.2716 |
| Wukong | 79.0M | +0.4469 |
| HiFormer | 79.2M | +0.5103 |
| RankMixer | 89.4M | +0.5419 |
| **UniMixer-Lite-2-Blocks** | **76.2M** | **+0.6824** |
| **UniMixer-Lite-4-Blocks** | **84.5M** | **+0.8141** |

关键观察：
- UniMixer-Lite-4-Blocks 以 84.5M 参数取得最优 AUC 提升（+0.8141%），在推荐系统这种对微小 AUC 差异极度敏感的场景中，这是非常显著的提升
- 相比最强 baseline RankMixer（+0.5419%），提升幅度约 50%
- 参数量并非最多，但效果最好，体现了更优的参数效率

**Scaling Law 对比**:

![Scaling Law 曲线对比](https://arxiv.org/html/2604.00590v2/x4.png)

论文拟合了幂律关系：

$$\Delta \text{AUC}_{\text{UniMixer-Lite}} = 0.003767 \cdot \text{Params}^{0.141903}$$

$$\Delta \text{AUC}_{\text{RankMixer}} = 0.003262 \cdot \text{Params}^{0.116043}$$

- UniMixer-Lite 的 scaling exponent（0.1419）**高于** RankMixer（0.1160），意味着同等增加参数量时 UniMixer-Lite 获得的性能提升更大
- 在 FLOPs 维度上，UniMixer-Lite 同样展现更优的 scaling 效率

![2-block 和 4-block 配置的 Scaling 曲线](https://arxiv.org/html/2604.00590v2/x6.png)

### 3.3 Ablation Study (消融实验)

**关键组件消融（Table 3）**:

| 消融设置 | ΔAUC 变化 (%) |
|---------|-------------|
| 去掉温度系数 (w/o temperature) | -0.1645 |
| 去掉模型预热 (w/o warm-up) | -0.0856 |
| 去掉对称性约束 (w/o symmetry) | -0.0573 |

关键发现：
- **温度退火**是最重要的设计选择，去掉后性能下降最大（-0.1645%），说明从"探索"到"聚焦"的训练策略至关重要
- **模型预热**贡献第二（-0.0856%），验证了先高温充分学习再低温精调的有效性
- **对称性约束**也有正面贡献（-0.0573%），保证特征交互的对称性是有意义的

**UniMixing-Lite 设计选择分析（Table 4）**:

- 增加 basis 数量比增加 rank 更高效——说明局部混合的多样性比全局混合的精度更重要
- 沿深度方向 scaling（增加 block 数）比沿宽度方向 scaling（增加维度）更高效——4-block 相比 2-block 额外提升 +0.1575% AUC
- 这一发现与 NLP 领域"更深优于更宽"的经验一致

### 3.4 Online Results (线上实验)

在快手多个广告场景进行 A/B 测试：
- 观测窗口为 30 天
- 平均提升约 **15%** 的累计活跃天数（Cumulative Active Days）
- 验证了 UniMixer 在工业级推荐系统中的实际价值

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **统一视角的理论贡献**: 通过等价参数化揭示 attention、TokenMixer、FM 三种范式的内在联系，这种"从不同方法中抽象出统一框架"的思路非常有价值，为推荐系统架构设计提供了新的理论基础
2. **工程与理论兼顾**: 不仅提出理论框架，还通过 Kronecker 分解 + basis-composed 等技巧给出了实际可用的轻量版 UniMixer-Lite，参数效率和计算效率都有显著提升
3. **训练策略的创新**: 温度退火 + 模型预热的组合策略简单有效，从消融实验看贡献显著，且容易迁移到其他场景
4. **完整的 scaling law 分析**: 不仅展示了性能，还拟合了参数量-性能的幂律关系，为工业界提供了"加多少参数能提多少点"的量化参考
5. **大规模工业验证**: 在快手 7 亿+ 数据和实际线上 A/B 测试中验证，说明方法具有工业可用性

### 4.2 Limitations & Improvements (不足与改进方向)

1. **数据集单一**: 仅在快手广告数据上进行实验，缺乏公开数据集的对比验证，难以评估方法在不同推荐场景（如电商、内容推荐）中的普适性
2. **Scaling law 的适用范围**: 目前的幂律关系是在特定参数范围内拟合的，是否在更大参数量级（如 1B+）上依然成立尚未验证
3. **线上指标披露有限**: 线上实验仅报告了"约 15%"的宏观提升，缺乏更细致的指标（如 CTR、CVR、RPM 等的具体数字），也缺乏置信区间
4. **SiameseNorm 的理论分析不足**: 虽然实验验证了有效性，但为什么两条耦合流能同时兼顾稳定性和表达力缺乏更深入的理论分析
5. **可能的改进方向**:
   - 将 UniMixer 的统一框架扩展到行为序列建模（论文 conclusion 中也提到了这一方向）
   - 探索动态温度调度（如根据训练 loss 的变化自适应调整温度），可能比固定的退火策略更优
   - 结合生成式推荐范式，探索 UniMixer 作为 backbone 在生成式推荐中的应用
