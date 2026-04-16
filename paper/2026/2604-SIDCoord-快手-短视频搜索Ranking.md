# SID-Coord: Coordinating Semantic IDs for ID-based Ranking in Short-Video Search

> **Authors**: Guowen Li, Yuepeng Zhang, Shunyu Zhang, Yi Zhang, Xiaoze Jiang, Yi Wang, Jingwei Zhuo
> **Affiliation**: Kuaishou Technology（快手）
> **Venue**: SIGIR 2026 (July 20-24, Melbourne, Australia)
> **Links**: [arXiv](https://arxiv.org/abs/2604.10471)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 大规模短视频搜索系统中，基于哈希 ID（HID）的 ranking 模型擅长记忆高频交互（memorization），但对长尾低曝光内容的泛化能力（generalization）不足。传统 HID 是随机分配的，语义相似的视频之间无法共享知识，导致低曝光视频的 embedding 训练不充分、排序不准确。如何在保持头部内容记忆能力的同时提升长尾内容的泛化能力？
- **Method (方法)**: 提出 **SID-Coord**，将离散可训练的 Semantic ID（SID，通过 RQ-KMeans 生成）引入现有的 ID-based ranking 模型，并通过三个协调机制整合：(1) **Multi-resolution SID Modeling** 用层次化组合构造多粒度 SID，通过 attention 自适应融合；(2) **Target-aware HID-SID Gating** 根据 item 的热度自适应平衡记忆与泛化；(3) **User-Item Semantic Alignment** 通过分布级语义匹配建模用户兴趣与候选 item 的语义相关性。
- **Results (效果)**: 离线全量 AUC +0.33%、UAUC +1.1%；长尾 item UAUC +0.93%（提升尤为明显）；线上 A/B 测试 7 天，搜索场景长播放率（long-play rate）+0.664%，播放次数 +0.472%，观看时长 +0.369%，搜索延迟仅增加 +0.005%（可忽略）。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

![SID-Coord 框架概览](https://arxiv.org/html/2604.10471v1/x1.png)

SID-Coord 的整体流程：

- **输入**: 短视频 item 的传统哈希 ID（HID）+ 通过 RQ-KMeans 生成的三层 Semantic ID $(s^{(1)}, s^{(2)}, s^{(3)})$ + 用户历史交互序列 + item 热度统计特征
- **三个核心模块**: (1) Multi-resolution SID Modeling → 多粒度语义表示；(2) HID-SID Gating → 自适应融合；(3) User-Item Semantic Alignment → 语义兴趣匹配
- **输出**: 融合后的特征送入 ranking backbone 预测 CTR/长播放率等目标
- **部署**: 无需修改 ranking backbone 架构，仅新增 SID 相关特征，搜索延迟增加可忽略

### 2.2 Core Modules (核心模块详解)

**模块一：Semantic ID 生成（RQ-KMeans）**

- **功能**: 为每个视频生成层次化的语义标识
- **方法**: 使用 RQ-KMeans（Residual Quantization + K-Means）对 item 的内容 embedding 进行三层残差量化
- **配置**: 3 个残差 codebook，每个大小 8192
- **输出**: 每个 item 获得三个层次的 SID $\{s_v^{(1)}, s_v^{(2)}, s_v^{(3)}\}$，粒度从粗到细
- **直觉理解**: 与 TIGER 的 RQ-VAE 思路一致——第一层区分大类（如"美食 vs 游戏 vs 音乐"），第二层细分子类（如"中餐 vs 西餐"），第三层进一步精细化

**模块二：Multi-resolution SID Modeling（多分辨率 SID 建模）**

![SID 多级注意力融合机制](https://arxiv.org/html/2604.10471v1/x2.png)

- **功能**: 构造多粒度的 SID 表示并自适应融合
- **层次化组合**: 通过 radix-B 公式构造相邻层级的组合 SID：

$$s_v^{(1,2)} = B \cdot s_v^{(1)} + s_v^{(2)}$$

$$s_v^{(2,3)} = B \cdot s_v^{(2)} + s_v^{(3)}$$

- $B = 10000$: 基数，大于 codebook 大小 8192，保证组合后的整数唯一
- **直觉理解**: $s^{(1,2)}$ 相当于"中粒度"标识（同时编码第一层和第二层信息），$s^{(2,3)}$ 相当于"细粒度"标识。这样每个 item 有 5 种不同粒度的 SID：$\{s^{(1)}, s^{(2)}, s^{(3)}, s^{(1,2)}, s^{(2,3)}\}$

- **为什么只组合相邻层**: 论文发现非相邻组合（如 $s^{(1,3)}$）会导致表示过度稀疏碎片化——跳过中间层意味着语义不连续，组合出的 ID 难以在训练中获得充足样本

- **Attention-based 融合**: 5 种 SID 分别映射为 embedding，通过 learned attention 机制自适应加权融合：

$$\mathbf{e}_v^{sid} = \sum_{k} \alpha_k \cdot \mathbf{e}_v^{(k)}$$

- $\alpha_k$: 通过 attention 学习的权重，自动决定在当前 item 的语境下哪种粒度最重要
- **直觉理解**: 对于热门 item，细粒度 SID 可能更有区分度；对于长尾 item，粗粒度 SID 更可靠（因为细粒度 SID 样本太少）。Attention 机制让模型自动做出这种粒度选择

**模块三：Target-aware HID-SID Gating（目标感知的 HID-SID 门控）**

- **功能**: 根据 item 的热度（popularity）自适应平衡 HID（记忆）和 SID（泛化）
- **门控公式**:

$$\mathbf{e}_v^{shid} = g \cdot \mathbf{e}_v^{hid} + (1 - g) \cdot \mathbf{e}_v^{sid}$$

- $g \in [0, 1]$: 门控权重，由两层 MLP + sigmoid 根据 item 的统计特征预测
- $\mathbf{e}_v^{hid}$: 传统哈希 ID embedding（记忆能力强）
- $\mathbf{e}_v^{sid}$: 多分辨率 SID embedding（泛化能力强）

- **统计特征输入**: 7 天曝光次数、点击数、点赞数、分享数、评论数等热度信号

![热度感知门控行为分析](https://arxiv.org/html/2604.10471v1/fig3.png)

- **实验验证**（Figure 3）: 将 item 按 7 天曝光量分为十等分（decile），高曝光 item 的 $g$ 值更大（更依赖 HID 记忆），低曝光 item 的 $g$ 值更小（更依赖 SID 泛化）
- **直觉理解**: "热门视频靠记忆，冷门视频靠语义"——这是一个非常直觉且有效的设计。热门 item 的 HID embedding 已经训练充分，直接用就好；冷门 item 的 HID 不可靠，需要借助语义相似 item 的共享知识（SID）来补偿

**模块四：User-Item Semantic Alignment（用户-物品语义对齐）**

- **功能**: 建模用户历史兴趣与候选 item 在语义层面的匹配程度
- **语义相似度计算**:

$$S = [\text{sim}(\mathbf{e}_v^{sid}, \mathbf{e}_{v_1}^{sid}), \ldots, \text{sim}(\mathbf{e}_v^{sid}, \mathbf{e}_{v_T}^{sid})]$$

- 计算候选 item 与用户最近 $T$ 个历史交互 item 的余弦相似度向量
- **AutoDis 变换**: 将相似度分数转化为分布感知的表示（distribution-aware representation），捕捉高阶语义兴趣模式
- **双路池化**:
  - **Max Pooling**: 提取最强语义匹配（"用户历史中与候选最相似的那个 item 有多相似"）
  - **Average Pooling**: 提取整体一致性（"候选 item 与用户整体兴趣的平均契合度"）
- 两路池化结果作为特征注入 ranking backbone
- **直觉理解**: 传统 ranking 模型主要通过 ID 级别的交互特征建模用户兴趣，这里新增了"语义级别"的兴趣匹配信号——即使候选 item 从未被用户看过，只要它在语义上与用户历史兴趣高度相似，也能获得较高的排序分数

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **平台**: 快手短视频搜索
- **训练数据**: ~45 亿样本
- **测试数据**: ~1.3 亿样本（按时间划分 hold-out）
- **评价指标**: AUC、UAUC（离线）；观看时长、播放次数、长播放率（线上）
- **SID 配置**: RQ-KMeans，3 层 codebook × 8192 codewords
- **主要 Baselines**:
  - **Prefix-Ngram**: 基于前缀的层次化 SID 组合
  - **Ngram**: 相邻 n-gram 组合
  - **SPM-SID**: 子词分割 + 固定词表
  - **DAS**: 双对齐语义 ID + 匹配统计

### 3.2 Main Results (主实验结果)

**Table 1: 离线性能对比**:

| 模型 | 全量 AUC | 全量 UAUC | 长尾 AUC | 长尾 UAUC |
|------|---------|----------|---------|----------|
| base | 0.7683 | 0.6091 | 0.7691 | 0.5951 |
| Prefix-Ngram | 0.7688 | 0.6096 | 0.7698 | 0.5962 |
| Ngram | 0.7695 | 0.6100 | 0.7706 | 0.5961 |
| SPM-SID | 0.7700 | 0.6101 | 0.7711 | 0.5974 |
| DAS | 0.7693 | 0.6111 | 0.7705 | 0.5983 |
| **SID-Coord** | **0.7708** | **0.6158** | **0.7723** | **0.6045** |

关键观察：
- SID-Coord 在所有指标上全面领先，全量 UAUC 提升 +1.1%（0.6091 → 0.6158），这是一个很大的提升
- **长尾 UAUC 提升尤为突出**（+0.93%），验证了 SID 对泛化能力的核心价值
- 最强 baseline SPM-SID 的全量 AUC 为 0.7700，SID-Coord 达到 0.7708，差距体现在"协调机制"的系统性设计上
- DAS 在 UAUC 上表现较好（0.6111），但 AUC 不如 SPM-SID，说明单一的语义匹配策略不够，需要 SID-Coord 的多模块协调

### 3.3 Ablation Study (消融实验)

**Table 2: 各模块贡献**:

| 模型 | 全量 AUC | 长尾 AUC |
|------|---------|---------|
| FULL（SID-Coord） | 0.7708 | 0.7723 |
| w/o Multi-resolution SID（去掉模块 I） | 0.7691 (-0.22%) | 0.7703 (-0.26%) |
| w/o HID-SID Gating（去掉模块 II） | 0.7698 (-0.13%) | 0.7712 (-0.14%) |
| w/o Interest Alignment（去掉模块 III） | 0.7696 (-0.16%) | 0.7708 (-0.19%) |

关键发现：
- **Multi-resolution SID 贡献最大**（-0.22% / -0.26%），说明多粒度的语义表示是基础——没有好的 SID 表示，后续的门控和对齐都无从谈起
- **Interest Alignment 贡献第二**（-0.16% / -0.19%），语义级别的用户兴趣匹配提供了传统 ID 交互特征无法替代的信号
- **HID-SID Gating 贡献第三但不可或缺**（-0.13% / -0.14%），热度感知的自适应融合避免了"一刀切"地使用 SID 导致头部内容性能下降
- 在长尾 item 上，每个模块的贡献都更大（-0.26% vs -0.22%，-0.19% vs -0.16%），再次验证 SID 对长尾泛化的核心价值

### 3.4 Online Results (线上实验)

**Table 3: 线上 A/B 测试（7 天，10% 流量）**:

| 指标 | 提升 | 95% 置信区间 |
|------|------|-------------|
| 观看时长 (Watch Time) | **+0.369%** | [0.05%, 0.69%] |
| 播放次数 (Play Count) | **+0.472%** | [0.17%, 0.77%] |
| 长播放率 (Long-play Rate) | **+0.664%** (+0.226 pp) | [0.49%, 0.84%] |
| 搜索延迟 | +0.005% | 可忽略 |

关键观察：
- **长播放率提升最为显著**（+0.664%），说明 SID 帮助模型找到了用户真正感兴趣的内容（看完而非点了就走）
- 统计显著性充分——所有指标的 95% 置信区间均不包含 0
- **延迟增加仅 +0.005%**，说明 SID 的引入对生产系统几乎无额外开销
- 实验采用了严格的统计验证：先进行 7 天 A/A 随机化测试，再用 difference-in-differences 估计器评估

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **"记忆 vs 泛化"的显式建模**: 论文将推荐系统中的 memorization-generalization trade-off 显式化为 HID（记忆）和 SID（泛化）的融合问题，并通过热度感知门控让模型自适应地在两者间平衡——这个问题意识和建模思路非常清晰
2. **多分辨率 SID 的实用设计**: radix-B 组合方式简单高效，既保留了层次化语义结构，又避免了非相邻组合的稀疏问题。5 种粒度 + attention 融合的设计既灵活又易于实现
3. **完整的统计验证**: 线上实验报告了 95% 置信区间，并使用 A/A 测试 + difference-in-differences 估计器，统计严谨性在推荐系统论文中属于较高水平
4. **生产级可行性**: 延迟仅增加 0.005%，无需修改 ranking backbone，展示了工业部署的务实态度
5. **消融实验设计合理**: 三个模块的贡献清晰可分离，且在全量和长尾上分别报告，帮助读者理解每个设计选择的价值

### 4.2 Limitations & Improvements (不足与改进方向)

1. **SID 生成与 ranking 模型分离**: RQ-KMeans 生成 SID 是独立的预处理步骤，SID 的语义结构未针对下游 ranking 任务优化。端到端联合训练 SID 生成与 ranking 可能带来进一步提升
2. **仅在搜索场景验证**: 论文实验仅在快手搜索上进行，未验证在推荐 Feed、广告等场景中的效果。搜索有明确 query 指导，SID 的语义匹配天然有优势；在无 query 的推荐场景中效果可能不同
3. **Backbone 架构未公开**: 论文未描述 ranking backbone 的具体架构，读者难以评估 SID-Coord 与不同 backbone 的兼容性
4. **Gate 机制的简单性**: 两层 MLP + sigmoid 的门控网络相对简单，更复杂的条件门控（如 attention-based gating、基于 item embedding 本身的 gating）可能效果更好
5. **可能的改进方向**:
   - 将 SID-Coord 的思路与 UniMixer 等 scaling 架构结合——在 UniMixer 的 Token Mixing 中引入 SID 作为额外的语义 token
   - 探索 SID 在用户侧的应用——为用户也生成语义 ID（基于其兴趣画像），实现 user SID 与 item SID 的直接匹配
   - 结合 IDProxy 的思路，用 MLLM 生成更丰富的内容 embedding 作为 RQ-KMeans 的输入，提升 SID 的语义质量
   - 探索动态 SID——随着 item 积累交互数据，定期更新 SID 以反映协同过滤信号
