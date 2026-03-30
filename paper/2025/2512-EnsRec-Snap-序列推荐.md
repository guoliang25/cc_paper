# Exploiting ID-Text Complementarity via Ensembling for Sequential Recommendation

> **Authors**: Liam Collins, Bhuvesh Kumar, Clark Mingxuan Ju, Tong Zhao, Donald Loveland, Leonardo Neves, Neil Shah
> **Affiliation**: Snap Inc., Santa Monica, CA, USA
> **Venue**: arXiv preprint, 2025.12 (v2: 2026.01)
> **Links**: [arXiv](https://arxiv.org/abs/2512.17820)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 序列推荐（Sequential Recommendation, SR）领域中，ID embedding 和文本（text）特征是两种主流的 item 表示方式。现有方法要么完全用文本替代 ID，要么设计复杂的融合架构来整合两者。但这些方法是否真的充分利用了 ID 和文本特征之间的**互补性（complementarity）**？这个问题一直缺乏系统性的研究。
- **Method (方法)**: 本文提出 **EnsRec**，核心思路极其简单——分别独立训练一个 ID-based 模型和一个 text-based 模型，推理时直接将两个模型的打分**相加**即可。作者还定义了两个互补性度量指标（Jaccard Index 和 Genie Score）来定量分析 ID 与文本特征的互补程度。
- **Results (效果)**: 在 Amazon Beauty/Toys/Sports 和 Steam 四个数据集上，EnsRec 在 NDCG@10 上分别取得 +8.2%、+12.3%、+13.7% 的提升（相比最强 baseline），且在 Recall@10 上也有显著提升。简单的分数求和集成击败了所有复杂的融合方法。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

EnsRec 的架构非常直观：

1. **输入**: 用户的历史交互序列 $\mathcal{S}_{:k}^{(u)} = (i_1^{(u)}, \ldots, i_k^{(u)})$
2. **两个独立模型**:
   - **ID 模型 $M_{ID}$**: 使用 ID-indexed embedding table $E_{ID}$ 将 item 映射到向量空间，再通过序列编码器 $T$ 得到用户表示
   - **Text 模型 $M_{text}$**: 使用冻结的预训练语言模型（frozen SentenceT5-XXL）将 item 文本映射到向量空间，再通过序列编码器 $T$ 得到用户表示
3. **输出**: 推理时将两个模型的相似度分数直接相加，作为最终推荐排序分数

关键点在于两个模型**完全独立训练**，不共享任何参数，推理时才进行集成。

### 2.2 Core Modules (核心模块详解)

**Dense Retrieval SR 模型定义**

- **功能**: 将用户交互序列映射为一个用户 embedding，与 item embedding 做相似度计算
- **输入**: 用户历史交互序列 $\mathcal{S}_{:k}^{(u)}$，每个 item $i$ 有一个 embedding $E(i) \in \mathbb{R}^d$
- **输出**: 每个候选 item 的推荐分数
- **关键公式**:

$$s_{M,u,k}(i) = \text{sim}(M(\mathcal{S}_{:k}^{(u)}), E(i))$$

- $M = T \circ E$: 模型由 embedding 层 $E$ 和序列编码器 $T$ 组成
- $T(E(\mathcal{S}_{:k}^{(u)})) \in \mathbb{R}^d$: 编码后的用户上下文向量
- $\text{sim}(\cdot, \cdot)$: 相似度函数（如内积）
- **直觉理解**: 先把 item 序列通过 embedding 映射到向量空间，再用 Transformer 编码器压缩成一个用户兴趣向量，最后和候选 item 比较相似度

---

**互补性度量指标**

本文提出两个指标来量化两个模型的互补程度：

**Jaccard Index $\tilde{J}$**

- **功能**: 衡量两个模型"推荐成功"的用户集合的重叠程度，越低说明互补性越强
- **关键公式**:

$$\tilde{J}(M, M') = \frac{|\mathcal{C}_M \cap \mathcal{C}_{M'}|}{|\mathcal{C}_M \cup \mathcal{C}_{M'}|}$$

- $\mathcal{C}_M = \{u \in \mathcal{U} : \text{rank}_{\mathcal{I}}(s_{M,u,n-1}, i_n^{(u)}) \leq 10\}$: 模型 $M$ 在 Recall@10 指标下"推荐成功"的用户集合
- **直觉理解**: 如果两个模型擅长推荐的用户群体高度不重合（Jaccard 低），说明它们各自捕获了不同的信号，互补性强

**Genie Score**

- **功能**: 估算一个"理想组合器"（oracle）能达到的性能上界
- **关键公式**:

$$\text{Genie}(M, M') = \frac{|\mathcal{C}_M \cup \mathcal{C}_{M'}|}{|\mathcal{U}|}$$

- **直觉理解**: 把两个模型各自推荐成功的用户取并集，计算覆盖率。这代表了完美集成的理论上限——如果有一个 oracle 总能在两个模型中选出对的那个，能达到的最高 Recall

---

**EnsRec 集成公式**

- **功能**: 将两个独立模型的打分合并为最终推荐分数
- **关键公式**:

$$s_{\widetilde{M}_{Ens},u,k}(i) = s_{M_{ID},u,k}(i) + s_{M_{text},u,k}(i)$$

- **直觉理解**: 最简单的集成——直接把 ID 模型和 text 模型对同一个 item 的评分加起来。如果两个模型都觉得某个 item 好，它的分数就会特别高；如果只有一个模型觉得好，也能获得一定的分数提升

**参数化集成（用于消融实验）**

$$s_{Ens_{\alpha,\tau}}(i) = \alpha \cdot \exp(s_{ID}(i)/\tau) + (1-\alpha) \cdot \exp(s_{text}(i)/\tau)$$

- $\alpha \in [0, 1]$: 控制 ID 模型和 text 模型的权重比例
- $\tau$: 温度参数（temperature），控制分数分布的尖锐程度
- 默认 EnsRec 对应 $\alpha = 0.5, \tau \to \infty$（实际使用 $\tau = 100$）
- **直觉理解**: 当 $\tau$ 很大时，$\exp(s/\tau) \approx 1 + s/\tau$，退化为线性加权求和；$\alpha = 0.5$ 表示两个模型等权重

### 2.3 Training Strategy (训练策略)

- **Loss Function**:

$$\mathcal{L} = \text{InfoNCE (full-batch)}$$

所有模型（ID 模型和 text 模型）统一使用 full-batch InfoNCE loss 训练。对于某些 baseline 方法有额外的辅助 loss，但 EnsRec 本身只用标准的 InfoNCE。

- **直觉理解**: InfoNCE loss 是对比学习的标准 loss，将正样本（用户实际交互的下一个 item）拉近，将 batch 内其他所有 item（负样本）推远

- **训练细节**:
  - 序列编码器: 6 层 T5 encoder-only 架构，6 个 attention head，FFN 维度 1024，key-value 维度 64
  - Embedding 维度: 128（ID 和 text 均为 128 维）
  - 文本编码器: 冻结的 SentenceT5-XXL（不参与训练）
  - 学习率: 从 {0.0001, 0.0003, 0.001} 中选择
  - Embedding dropout: 从 {0.1, 0.5} 中选择
  - Text projection: linear 或 3 层 MLP
  - Early stopping: 基于验证集 Recall@10
  - 每个实验跑 3 个随机种子取平均

- **推理优化**: 将两个模型的用户 embedding 归一化后拼接（concatenate），利用 ANN（近似最近邻）检索实现高效推理

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **数据集**:

| 数据集 | 用户数 | 物品数 | 交互数 |
|--------|--------|--------|--------|
| Beauty | 22,363 | 12,101 | 198,502 |
| Toys | 19,412 | 11,924 | 167,597 |
| Sports | 35,598 | 18,357 | 296,337 |
| Steam | 47,761 | 12,012 | 599,620 |

- **评价指标**: NDCG@10, Recall@10
- **主要 Baselines**: 14 种方法，包括:
  - 纯 ID / 纯 Text 模型
  - 文本增强方法: WhitenRec, UniSRec, LLMInit, RLLMRec-Con, RLLMRec-Gen, LLM-ESR
  - 多模态融合方法: AlphaFuse, FDSA, LIGER-Dense, LIGER
  - 生成式推荐: TIGER

### 3.2 Main Results (主实验结果)

**互补性分析 (Q1)**:

ID-text 模型对的 Jaccard Index 在 Beauty 上为 0.47，Steam 上为 0.75，均低于同类型特征对（如 ID-ID 变体为 0.57，text-text 变体为 0.52），确认了 **ID 和 text 特征之间存在显著互补性**。

**主实验结果 (Q2)**:

| 方法 | Beauty NDCG@10 | Beauty R@10 | Sports NDCG@10 | Sports R@10 | Toys NDCG@10 | Toys R@10 | Steam NDCG@10 | Steam R@10 |
|------|---------------|-------------|----------------|-------------|-------------|-----------|--------------|------------|
| ID-Only | 4.893 | 8.211 | 2.785 | 4.858 | 5.460 | 8.812 | 15.70 | 19.79 |
| Text-Only | 4.790 | 9.373 | 2.927 | 5.667 | 4.860 | 9.853 | 15.73 | 20.07 |
| LLMInit | 5.220 | 9.504 | 3.019 | 5.714 | 5.276 | 10.01 | 16.02 | 20.40 |
| UniSRec | 5.033 | 9.674 | 2.885 | 5.536 | 4.913 | 9.889 | 15.91 | 20.19 |
| LLM-ESR | 5.181 | 9.717 | 2.913 | 5.542 | 4.910 | 9.535 | 15.50 | 19.77 |
| LIGER-Dense | 4.726 | 9.190 | 2.951 | 5.622 | 4.667 | 9.465 | 14.92 | 19.10 |
| **EnsRec** | **5.650** | **9.800** | **3.432** | **6.133** | **6.248** | **10.61** | **16.02** | **20.45** |
| Genie 上界 | 7.097 | 12.20 | 4.480 | 7.833 | 7.599 | 12.87 | 17.34 | 22.73 |

核心发现：
- **EnsRec 在所有数据集的 NDCG@10 上均达到最优**（或并列最优）
- Beauty: NDCG@10 达到 5.650，比第二名 LLMInit (5.220) 提升 **+8.2%**
- Sports: NDCG@10 达到 3.432，比第二名 LLMInit (3.019) 提升 **+13.7%**
- Toys: NDCG@10 达到 6.248，比第二名 RLLMRec-Gen (5.565) 提升 **+12.3%**
- 在 Steam 上与 LLMInit 持平（16.02），但 Recall@10 略高
- 所有复杂融合方法（AlphaFuse, FDSA, LIGER）表现均不如简单集成
- EnsRec 的性能与 Genie 上界之间仍有差距，表明未来还有提升空间

### 3.3 Ablation Study (消融实验)

**参数敏感性分析 (Q3)**:

![EnsRec sensitivity heatmaps](https://arxiv.org/html/2512.17820v2/x1.png)

通过在 $(\alpha, \log_{10}\tau)$ 网格上扫描参数化集成公式，发现：
- **等权重 $\alpha = 0.5$ 配合高温度 $\tau = 100$ 就能逼近最优性能**
- Beauty 数据集上，通过验证集调参可以将 Recall@10 从 9.80 进一步提升到 10.00
- Steam 数据集上，默认参数已经是最优（20.45）
- 结果对参数变化不敏感，简单的默认设定就足够好

**编码器架构消融**:
- 在互补性分析（Q1）中也使用了 2 层 1 头的 SASRec 风格解码器变体，确认了互补性发现在不同架构下的一致性
- 文本编码器也对比了 Sup-SimCSE-BERT 与 SentenceT5-XXL

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **极简而有效的方法**: 独立训练 + 简单分数相加就能超越所有复杂融合架构，体现了"less is more"的思想。这对工业界部署非常友好——两个模型可以独立迭代、独立上线
2. **互补性度量指标的提出**: Jaccard Index 和 Genie Score 提供了系统性分析两种特征互补程度的工具，可以推广到其他多模型/多特征融合场景
3. **实验设计严谨**: 统一了所有 baseline 的编码器架构（6 层 T5）和 loss（InfoNCE），消除了架构差异带来的混淆因素，使得对比更加公平
4. **推理效率**: 利用 embedding 拼接 + ANN 检索的方式，避免了推理时需要跑两个模型的开销

### 4.2 Limitations & Improvements (不足与改进方向)

1. **数据集规模偏小**: 四个数据集的用户和物品规模都在万级别，缺乏在百万/千万级工业数据集上的验证。ID 和 text 的互补性在大规模场景下是否依然成立值得探索
2. **简单求和的局限**: 虽然作者展示了等权重求和接近最优，但在不同 domain 或不同数据分布下，固定策略是否始终有效存疑。自适应加权（如 attention-based 或 learned weight）可能在某些场景下更优
3. **缺乏冷启动分析**: Text 特征在冷启动（cold-start）场景下应该有天然优势，但论文没有按冷启动/热门用户分组分析 EnsRec 的表现
4. **只考虑了文本模态**: 实际推荐系统中 item 往往有图像、视频等多种模态特征。将 EnsRec 的集成思想扩展到更多模态（如 ID + text + image 三路集成）是自然的下一步
5. **Genie 上界的差距**: EnsRec 与 Genie 上界之间仍有明显差距（如 Beauty 上 9.80 vs 12.20），说明简单求和并不能完美利用互补性，如何缩小这个差距是重要的研究方向
