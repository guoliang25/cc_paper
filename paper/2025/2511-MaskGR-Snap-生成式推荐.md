# Masked Diffusion for Generative Recommendation

> **Authors**: Kulin Shah, Bhuvesh Kumar, Neil Shah, Liam Collins
> **Affiliation**: Snap Inc. / UT Austin（Shah 实习于 Snap）
> **Venue**: arXiv preprint, November 2025
> **Links**: [arXiv](https://arxiv.org/abs/2511.23021) | [Code](https://github.com/snap-research/MaskGR)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 生成式推荐（Generative Recommendation, GR）使用语义 ID（Semantic ID, SID）将 item 编码为离散 token 序列，然后用语言模型自回归（AR）生成推荐。但 AR 建模存在两个核心问题：(1) 推理时必须逐 token 解码，速度慢；(2) 每个 token 只能看到左边的上下文，无法捕捉 SID 序列内的全局依赖关系，导致数据利用效率低。
- **Method (方法)**: 提出 **MaskGR**，用掩码扩散（Masked Diffusion）替代 AR 建模。训练时随机掩盖 SID 序列中的部分 token，模型学习预测被掩盖的 token；推理时从全掩码状态出发，通过少量去噪步骤并行解码所有 token。同时集成密集检索（dense retrieval）头增强粗粒度召回能力。
- **Results (效果)**: 在 Amazon Beauty/Sports/Toys 和 MovieLens-1M 四个数据集上，MaskGR 在 NDCG@5 上平均超越 TIGER（经典 AR 基线）**21.9%**，仅需 3 次前向传播即可完成推理，且在数据稀疏场景下优势更明显。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

![MaskGR 训练与推理流程](https://arxiv.org/html/2511.23021/x1.png)

MaskGR 的整体流程：

- **输入**: 用户历史交互序列，每个 item 用 SID 表示（$m$ 个 token），最近 $n$ 个交互 item 组成序列 $S^u_0 \in \{1,...,V\}^{mn}$，其中 $V$ 为词表大小
- **SID 编码**: 使用多层残差 K-Means 聚类将 item 的文本 embedding 量化为离散 token 序列
- **训练**: 随机掩盖序列中的 token（前向扩散），encoder-only Transformer 预测被掩盖位置的原始 token
- **推理**: 从全掩码序列出发，迭代去噪，每步根据不确定性贪心选择最确定的位置先解码
- **输出**: 预测的下一个 item 的 SID token 序列 → 检索最近邻 item

**模型配置**:
- 8 层 encoder-only Transformer，7M 参数
- Embedding 维度: 128
- 注意力头数: 8
- MLP 隐层维度: 3072
- 位置编码: RoPE（旋转位置编码）

### 2.2 Core Modules (核心模块详解)

**模块一：Semantic ID 生成（语义 ID）**

- **功能**: 将每个 item 编码为一组离散 token，使语义相近的 item 共享前缀
- **流程**:
  1. 用 Flan-T5-XXL 提取 item 的文本描述 embedding（4096 维）
  2. 对所有 item embedding 做 K-Means 聚类（$K=256$ 个簇）
  3. 取残差（原始 embedding - 簇中心）
  4. 对残差再做 K-Means 聚类
  5. 重复 $m=4$ 层，得到每个 item 的 SID = $(c_1, c_2, c_3, c_4)$，每个 $c_i \in \{1,...,256\}$
- **直觉理解**: 类似于"邮政编码"——第一层聚类像"省"，第二层像"市"，逐层细化，使得语义相近的 item 有相似的前缀编码。

---

**模块二：Masked Diffusion 前向过程**

- **功能**: 在训练时对 SID 序列施加噪声（随机掩盖 token）
- **输入**: 原始序列 $S^u_0$，噪声水平 $t \in [0, 1]$
- **输出**: 带噪声的序列 $S^u_t$

**前向转移概率**:

$$p(S^u_t | S^u_0) = \prod_{i=1}^{mn} \text{Cat}\left((1-t) \cdot e_{S^u_0(i)} + t \cdot e_{[M]}\right)$$

- $S^u_0(i)$: 序列中第 $i$ 个位置的原始 token
- $e_{S^u_0(i)}$: 该 token 的 one-hot 向量
- $e_{[M]}$: 掩码 token `[M]` 的 one-hot 向量
- $t$: 噪声水平，$t=0$ 为原始序列，$t=1$ 为全掩码
- **直觉理解**: 每个位置以概率 $t$ 被替换为 `[M]`，以概率 $1-t$ 保持原值。$t$ 越大，序列被破坏得越严重。这与连续扩散的"加高斯噪声"类似，只不过这里是离散的"掩盖"操作。

---

**模块三：反向去噪过程**

- **功能**: 从带噪序列恢复原始序列
- **输入**: 带噪序列 $S^u_t$，噪声水平 $t$，步长 $\ell = t - s$（$s$ 为目标噪声水平）
- **输出**: 去噪后的序列 $S^u_s$

**反向后验分布**:

对于未被掩盖的位置 $S^u_t(i) \neq [M]$：
$$p(S^u_s(i) | S^u_t) = \text{Cat}(e_{S^u_t(i)})$$

对于被掩盖的位置 $S^u_t(i) = [M]$：
$$p(S^u_s(i) | S^u_t) = \text{Cat}\left(\frac{\ell}{t} \cdot e_{[M]} + \frac{1 - \ell/t}{1} \cdot e_{S^u_0(i)}\right)$$

- **直觉理解**: 未被掩盖的位置保持不变；被掩盖的位置有两种可能——要么继续保持掩盖（概率 $\ell/t$），要么被"揭露"为模型预测的原始 token（概率 $1-\ell/t$）。

---

**模块四：训练目标**

**交叉熵损失**:

$$\mathcal{L} = \mathbb{E}_{t, S^u_0, S^u_t} \left[ -\frac{1}{t} \sum_{i=1}^{mn} \mathbb{I}[S^u_t(i) = [M]] \cdot \log p_\theta(S^u_0(i) | S^u_t) \right]$$

- $\mathbb{I}[S^u_t(i) = [M]]$: 指示函数，只对被掩盖的位置计算损失
- $1/t$: 按噪声水平加权，噪声越小（$t$ 越小）权重越大
- $p_\theta(S^u_0(i) | S^u_t)$: 模型预测第 $i$ 个位置原始 token 的概率
- **直觉理解**: 模型学习根据未被掩盖的上下文（双向注意力）预测被掩盖位置的 token。关键区别于 AR：每个位置可以看到序列中**所有**未被掩盖的位置，而非只看左侧。

---

**模块五：密集检索（Dense Retrieval）头**

- **功能**: 在 SID 离散检索之外，增加一个连续 embedding 空间的粗粒度召回通道
- **输入**: Transformer 最后一层的 hidden states $\tilde{H}$
- **输出**: 查询 embedding $g_\theta(\tilde{H})$

**对比学习损失**:

$$\mathcal{L}_{dense} = -\log \frac{\exp(g_\theta(\tilde{H})^\top \cdot h_i^{text})}{\sum_{j \in \mathcal{I}} \exp(g_\theta(\tilde{H})^\top \cdot h_j^{text})}$$

- $g_\theta(\tilde{H})$: 通过投影头将 Transformer 输出映射为查询 embedding
- $h_i^{text}$: 目标 item 的文本 embedding（Flan-T5-XXL 生成）
- $\mathcal{I}$: 全体 item 集合（分母为 batch 内负采样近似）
- **直觉理解**: 让模型在连续空间中也能"找到"目标 item，作为离散 SID 检索的补充。先用 dense retrieval 粗召回一批候选，再用 SID 匹配精排，提升覆盖率。

### 2.3 Training Strategy (训练策略)

- **总损失函数**:

$$\mathcal{L}_{total} = \mathcal{L} + \lambda \cdot \mathcal{L}_{dense}$$

  - $\lambda$: dense retrieval 损失的权重

- **推理解码策略 — 贪心不确定性解码**:
  1. 初始：所有 $m$ 个目标位置设为 `[M]`
  2. 模型前向传播，预测每个掩盖位置的 token 分布
  3. 计算每个位置的预测不确定性（熵）
  4. 选择不确定性最低的位置，将其"解码"为预测 token
  5. 重复步骤 2-4，直到所有位置解码完成
  6. 仅需约 3 次前向传播（NFE=3）即可达到良好性能

- **直觉**: 先解码最确定的位置（如 SID 的高层聚类 token），后解码不确定的位置（低层细粒度 token），让已解码的信息帮助后续位置的预测。

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **数据集**:
  - Amazon Beauty（美妆）
  - Amazon Sports（运动）
  - Amazon Toys（玩具）
  - MovieLens-1M（电影，~100 万交互）
- **评价指标**: NDCG@K, Recall@K（K=5, 10）
- **主要 Baselines**:
  - **TIGER**: 经典 AR 生成式推荐方法，使用 RQ-VAE + T5 decoder
  - **LETTER**: TIGER 改进版，优化了 SID 分配
  - **Zero-shot LLM**: 零样本大模型
  - MaskGR 变体: 随机 SID、Item ID、不同解码策略

### 3.2 Main Results (主实验结果)

**Table 1 - 主要性能对比**:

| 数据集 | 方法 | NDCG@5 | Recall@5 | NDCG@10 | Recall@10 |
|--------|------|--------|----------|---------|-----------|
| **Beauty** | TIGER | 2.88 | 4.29 | 3.67 | 6.59 |
| | **MaskGR** | **3.51** | **5.38** | **4.36** | **8.15** |
| | 提升 | +21.9% | +25.4% | +18.8% | +23.7% |
| **Sports** | TIGER | 1.64 | 2.45 | 2.05 | 3.72 |
| | **MaskGR** | **1.91** | **3.02** | **2.42** | **4.63** |
| | 提升 | +16.5% | +23.3% | +18.0% | +24.5% |
| **Toys** | TIGER | 2.91 | 4.42 | 3.51 | 6.36 |
| | **MaskGR** | **3.75** | **5.48** | **4.55** | **8.06** |
| | 提升 | +28.9% | +24.0% | +29.6% | +26.7% |
| **ML-1M** | TIGER | 8.85 | 12.83 | 10.83 | 18.66 |
| | **MaskGR** | **11.12** | **16.72** | **13.06** | **22.78** |
| | 提升 | +25.6% | +30.3% | +20.6% | +22.1% |

关键发现：
- MaskGR 在**所有 4 个数据集、所有指标**上全面超越 TIGER
- NDCG@5 平均提升约 **21.9%**，Recall@10 平均提升约 **24.3%**
- 在 ML-1M（最大数据集）上提升最显著：NDCG@5 从 8.85 → **11.12**
- 在 Toys（最小数据集之一）上也有大幅提升，说明 MaskGR 并非依赖大数据量

### 3.3 Ablation Study (消融实验)

**SID 策略消融（Beauty 数据集）**:

| 变体 | Recall@5 | Recall@10 |
|------|----------|-----------|
| MaskGR + 随机 SID | 3.78 | 5.53 |
| MaskGR + Item ID（原始 ID） | 4.69 | 6.71 |
| **MaskGR（语义 SID，完整版）** | **5.38** | **8.15** |

- 语义 SID 比随机 SID 提升 42.3%（Recall@5），证明语义聚类的 ID 分配至关重要
- 语义 SID 也优于原始 Item ID，说明层次化语义结构比平面 ID 更有效

**解码策略消融**:

| 解码策略 | Recall@5 | Recall@10 |
|----------|----------|-----------|
| 随机顺序 | 5.04 | 7.54 |
| 从左到右（模拟 AR） | 5.24 | 8.09 |
| **贪心（基于不确定性）** | **5.38** | **8.15** |

- 贪心策略最优，验证了"先解码最确定位置"的直觉
- 即使用随机顺序，MaskGR（5.04）仍优于 TIGER（4.29），说明双向注意力本身就是核心优势

**密集检索增强效果**:

| 配置 | Recall@10 |
|------|-----------|
| MaskGR（仅 SID） | 8.15 |
| MaskGR + Dense Retrieval | **8.59** |

集成密集检索后 Recall@10 提升 5.4%，说明连续空间和离散 SID 空间的互补性。

**推理效率 (NFE vs 性能)**:

![推理效率与性能权衡](https://arxiv.org/html/2511.23021/x4.png)

| 前向传播次数 (NFE) | NDCG@5 | Recall@5 |
|---------------------|--------|----------|
| 1（单步） | ~2.8 | ~4.2 |
| 3 | 3.51 | 5.38 |
| 4（= TIGER 步数） | ~3.55 | ~5.45 |

- 仅 3 次前向传播即可达到接近最优性能，**比 TIGER 少 1 步但效果更好 13%**
- NFE=1 时性能已接近 TIGER，展现了并行解码的潜力

**数据效率**:

![数据约束性能对比](https://arxiv.org/html/2511.23021/x3.png)

- 删除 50% 训练数据时，MaskGR 性能下降幅度 < TIGER
- 删除 75% 时，MaskGR 仍能维持合理性能，而 TIGER 大幅退化
- **直觉**: 双向注意力能从有限数据中学到更多 token 间的全局关系，而 AR 的单向注意力浪费了一半的上下文信息

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **"掩码扩散替代 AR" 的思路很直观且有效**: 将 NLP 领域 masked language model 的思想引入生成式推荐，核心洞察是 SID 的 token 之间不一定存在严格的从左到右依赖关系（与自然语言不同），双向注意力更适合捕捉 SID 内部的层次语义结构。

2. **模型极其轻量**: 仅 7M 参数的 encoder-only Transformer，训练和部署成本远低于基于 T5 等大模型的 TIGER。这对工业界落地非常友好。

3. **推理效率出色**: 3 次前向传播即可完成推理，且每步可并行解码多个 token。对比 TIGER 需要 4 步串行自回归，MaskGR 在速度和质量上实现了"双赢"。

4. **Dense Retrieval 的混合检索设计**: SID 离散匹配和连续 embedding 检索的结合，两个空间互补，提升了召回的鲁棒性。这种"离散+连续"的混合范式值得其他推荐系统借鉴。

5. **数据效率优势**: 在数据稀疏场景下优势扩大，这对冷启动和长尾场景有实际价值。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **实验规模偏小**: 四个数据集中最大的 ML-1M 也只有约 100 万交互，缺乏在工业级数据（亿级交互、百万级 item）上的验证。SID 的层次聚类在 item 量极大时是否依然有效需要进一步探索。

2. **SID 生成与模型训练解耦**: SID 由预训练的 Flan-T5-XXL embedding + 离线 K-Means 聚类生成，与下游推荐模型没有端到端联合优化。可以考虑 learnable tokenizer（如 VQ-VAE 联合训练）来进一步提升 SID 质量。

3. **缺少与更多 GR 方法的对比**: 主要 baseline 是 TIGER 和 LETTER，缺少与 GENRE、DSI、TIGER 后续改进工作（如 LC-Rec、TransRec）的对比，难以全面定位 MaskGR 的竞争力。

4. **密集检索的 item embedding 来自固定模型**: $h_i^{text}$ 使用 Flan-T5 的冻结 embedding，可以考虑让 item embedding 也可训练，或使用更适合推荐场景的预训练模型。

5. **解码策略可以进一步优化**: 当前的贪心策略基于简单的熵度量，可以探索更高级的解码策略，如 confidence-based parallel decoding 或 consistency model 风格的单步生成。
