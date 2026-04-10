# RankMixer: Scaling Up Ranking Models in Industrial Recommenders

> **Authors**: Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, et al.（共 21 位作者）
> **Affiliation**: ByteDance（字节跳动）
> **Venue**: arXiv preprint, July 2025
> **Links**: [arXiv](https://arxiv.org/abs/2507.15551)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 工业推荐系统的 ranking 模型面临两大挑战：(1) 严格的延迟和 QPS 约束限制了模型规模的扩展；(2) 现有架构（如 DLRM-MLP）的硬件利用率极低（Model FLOPs Utilization 仅 4.5%），导致大量算力被浪费在访存而非计算上。如何在不增加推理延迟的前提下，将模型参数扩展 100 倍？
- **Method (方法)**: 提出 **RankMixer**，一种硬件感知（hardware-conscious）的 ranking 架构：用 **multi-head token mixing** 替代二次复杂度的 self-attention 实现特征交互，用 **per-token FFN** 保持特征子空间建模，并通过 **Sparse-MoE**（稀疏专家混合）+ DTSI（Dense-Training/Sparse-Inference）策略将参数扩展到 10 亿级别。
- **Results (效果)**: MFU 从 4.5% 提升到 45%（10 倍提升），参数量扩展 70 倍但推理延迟保持不变（14.3ms）；线上在抖音 Feed 推荐中活跃天数 +0.29%、使用时长 +1.08%，低活用户活跃天数 +1.74%、时长 +3.64%；广告场景 AUC +0.73%、广告主价值 +3.90%。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

![RankMixer Block 架构](https://arxiv.org/html/2507.15551v2/x1.png)

RankMixer 的整体流程：

- **输入**: 推荐系统中的多源特征，按语义分组后进行 Feature Tokenization，转化为 $T$ 个维度为 $D$ 的 token
- **主要模块**: $L$ 层堆叠的 RankMixer Block，每个 block 包含：(1) Multi-head Token Mixing 进行跨 token 特征交互；(2) SMoE-based Per-token FFN 进行 token 内部非线性变换
- **输出**: 融合后的特征表示，送入预测层

每个 RankMixer Block 的前向过程：

$$\mathbf{S}_{n-1} = \text{LN}(\text{TokenMixing}(\mathbf{X}_{n-1}) + \mathbf{X}_{n-1})$$

$$\mathbf{X}_n = \text{LN}(\text{PFFN}(\mathbf{S}_{n-1}) + \mathbf{S}_{n-1})$$

- 采用 Pre-Norm + Skip Connection 的标准结构
- **直觉理解**: 先混合 token 间的信息（"谁和谁交互"），再对每个 token 独立做非线性变换（"怎么变换"）

### 2.2 Core Modules (核心模块详解)

**模块一：Feature Tokenization（特征 Token 化）**

- **功能**: 将异构特征统一为 token 序列
- **方法**: 将特征按语义分组（用户画像、物品特征、上下文等），每组内的 embedding 拼接后通过线性投影统一为固定维度 $d$：

$$x_i = \text{Proj}(e_{input}[d \cdot (i-1) : d \cdot i])$$

- **直觉理解**: 类似 NLP 中的 tokenization，把推荐系统的各类特征"翻译"为统一格式的 token

**模块二：Multi-head Token Mixing（多头 Token 混合）**

- **功能**: 实现跨 token 的特征交互，替代计算量为 $O(T^2 D)$ 的 self-attention
- **输入**: $T$ 个 token，每个维度 $D$
- **方法**: 将每个 token 的 embedding 分为 $H$ 个 head，然后跨 token 重组：

$$\mathbf{s}^h = \text{Concat}(\mathbf{x}_1^h, \mathbf{x}_2^h, \ldots, \mathbf{x}_T^h)$$

- 每个 head $h$ 收集所有 token 在该 head 位置的子向量，拼接为一个混合向量
- **输出**: $T$ 个混合后的 token（维度不变）
- **直觉理解**: 想象每个 token 是一个人，每个 head 是一个话题。Token Mixing 就是"让所有人在同一个话题下交流"——每个话题维度上，所有 token 的信息被拼接混合，实现了无参数的特征交叉
- **关键优势**: 完全无参数（parameter-free），计算复杂度 $O(TD)$，远低于 self-attention 的 $O(T^2D)$

**模块三：Per-token FFN（逐 token 前馈网络）**

- **功能**: 对每个 token 独立施加非线性变换，建模特征子空间
- **关键设计**: 每个 token 拥有**独立的**权重矩阵，而非共享：

$$\mathbf{v}_t = f_{pffn}^{(t,2)}(\text{GeLU}(f_{pffn}^{(t,1)}(\mathbf{s}_t)))$$

- $f_{pffn}^{(t,1)}, f_{pffn}^{(t,2)}$: 第 $t$ 个 token 专属的两层线性变换
- **直觉理解**: 不同 domain 的特征（用户 vs 物品 vs 上下文）性质差异很大，共享 FFN 会"强迫不同类型的特征用同一种方式变换"，而 per-token FFN 让每类特征有自己的"处理器"

**消融实验验证**（Table 2）:

| 消融设置 | AUC 变化 |
|---------|---------|
| 去掉 Multi-head Token Mixing | **-0.50%** |
| 共享 FFN（替代 Per-token FFN） | **-0.31%** |
| 去掉 Skip Connection | -0.07% |
| 去掉 Layer Normalization | -0.05% |

→ Token Mixing 和 Per-token FFN 是两个最关键的组件

**模块四：Sparse MoE（稀疏专家混合）**

- **功能**: 在不增加推理 FLOPs 的前提下大幅扩展模型参数
- **ReLU Routing（替代传统 Top-k Softmax）**:

$$G_{ij} = \text{ReLU}(h(\mathbf{s}_i))$$

$$\mathbf{v}_i = \sum_{j=1}^{N_e} G_{ij} \cdot e_{ij}(\mathbf{s}_i)$$

- $G_{ij}$: token $i$ 对 expert $j$ 的路由权重
- $e_{ij}$: 第 $j$ 个 expert 对 token $i$ 的输出
- ReLU 天然产生稀疏激活（大量为零），无需预设 Top-k

- **自适应 $\ell_1$ 正则化**:

$$\mathcal{L} = \mathcal{L}_{task} + \lambda \mathcal{L}_{reg}, \quad \mathcal{L}_{reg} = \sum_i \sum_j G_{ij}$$

- $\lambda$ 自适应调整以维持目标稀疏率
- **直觉理解**: 通过惩罚路由权重的总和来控制激活的 expert 数量，越大的 $\lambda$ 越少的 expert 被激活

- **DTSI-MoE（Dense-Training / Sparse-Inference）**:

![不同 token 的 expert 激活率](https://arxiv.org/html/2507.15551v2/x2.png)

- 训练时使用一个路由器 $h_{train}$（密集激活更多 expert，充分训练）
- 推理时使用另一个路由器 $h_{infer}$（稀疏激活，满足延迟约束）
- 正则化仅施加在 $h_{infer}$ 上
- **直觉理解**: "训练时让所有 expert 都充分学习（避免欠训练），推理时只激活最重要的几个（节省计算量）"，解决了传统 Sparse-MoE 中 expert under-training 的问题

### 2.3 Scaling Strategy (扩展策略)

模型可沿四个轴扩展：

- **Token 数量 $T$**: 增加特征分组的粒度
- **模型宽度 $D$**: 增加 token embedding 维度
- **层数 $L$**: 增加 RankMixer Block 的堆叠数
- **Expert 数量 $E$**: 通过 MoE 扩展（不增加 FLOPs）

Dense 版本的参数量和计算量：

$$\text{Param} \approx 2kLTD^2, \quad \text{FLOPs} \approx 4kLTD^2$$

- $k$: FFN hidden dimension 的倍数

**硬件效率分析**: RankMixer 将 FLOPs/Param 比从 6.8 G/M 降至 1.9 G/M（↓3.6×），意味着同等参数下需要的计算量更少，更适合 GPU 的计算密集型特性。

### 2.4 Training Strategy (训练策略)

- **优化器**: Dense 部分使用 RMSProp（lr=0.01），Sparse 部分使用 Adagrad
- **分布式训练**: 混合并行框架——异步更新 sparse embedding，同步更新 dense 参数
- **量化**: 使用 fp16 量化，在保持精度的同时获得 2× 硬件加速
- **配置**:
  - RankMixer-100M: $D=768, T=16, L=2$
  - RankMixer-1B: $D=1536, T=32, L=2$

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **数据集**: 字节跳动内部抖音推荐数据，涵盖 Feed 推荐和广告两个场景
- **评价指标**: Finish AUC、Skip AUC、UAUC（离线）；活跃天数、使用时长、点赞、完播、评论（线上）
- **主要 Baselines**:
  - **MLP-based**: DLRM-MLP
  - **FM-based**: Wukong
  - **Attention-based**: HiFormer, DHEN
  - **MoE-based**: DLRM-MoE

### 3.2 Main Results (主实验结果)

**离线对比（~100M 参数，Table 1）**:

| 模型 | 参数量 | Finish AUC Gain | Skip AUC Gain | Finish UAUC Gain |
|------|--------|-----------------|---------------|-------------------|
| DLRM-MLP | 95M | +0.15% | +0.15% | +0.14% |
| Wukong | 122M | +0.29% | +0.49% | +0.27% |
| HiFormer | 102M | +0.48% | +0.67% | +0.41% |
| **RankMixer-100M** | **107M** | **+0.64%** | **+0.86%** | **+0.72%** |
| **RankMixer-1B** | **1.1B** | **+0.95%** | **+1.25%** | **+1.22%** |

关键观察：
- RankMixer-100M 在同等参数量下全面领先，Finish AUC 比最强 baseline HiFormer 高出 +0.16%
- 扩展到 1B 后进一步提升至 +0.95%，展现出强大的 scaling 能力
- Skip AUC 提升尤为明显（+1.25%），说明模型在"用户不喜欢什么"的判断上尤其强

**Scaling Law 对比**:

![Scaling Law 曲线](https://arxiv.org/html/2507.15551v2/img/scaling_law.png)

- RankMixer 在 Params 和 FLOPs 两个维度上都展现出比 Wukong、HiFormer、DHEN 更陡峭的 scaling 曲线
- 不同扩展方向（深度 $L$、宽度 $D$、token 数 $T$）的 scaling 曲线几乎重合，与 LLM 中的 scaling 规律一致

### 3.3 Token Mixing 方案对比（Table 3）

| 方案 | ΔAUC | ΔParams | ΔFLOPs |
|------|------|---------|--------|
| All-Concat-MLP | -0.18% | 0% | 0% |
| All-Share | -0.25% | 0% | 0% |
| Self-Attention | -0.03% | +16% | **+71.8%** |
| **Multi-head Token Mixing** | **baseline** | **0%** | **0%** |

→ Self-Attention 性能接近但 FLOPs 增加 71.8%，在工业场景中不可接受；Multi-head Token Mixing 以零额外开销达到最优性价比

### 3.4 Deployment Efficiency (部署效率，Table 6)

| 指标 | Baseline (16M) | RankMixer-1B | 变化 |
|------|----------------|--------------|------|
| 参数量 | 15.8M | 1.1B | ↑70× |
| FLOPs | 107G | 2106G | ↑20.7× |
| FLOPs/Param | 6.8 G/M | 1.9 G/M | **↓3.6×** |
| MFU | 4.47% | 44.57% | **↑10×** |
| 延迟 | 14.5ms | 14.3ms | **持平** |

→ 参数扩展 70 倍、FLOPs 仅增 20 倍、延迟甚至微降——这是硬件感知设计的核心价值

### 3.5 Online Results (线上实验)

**抖音 Feed 推荐（Table 4）**:

| 用户群体 | 活跃天数 | 使用时长 | 点赞 | 完播 | 评论 |
|---------|---------|---------|------|------|------|
| 全量用户 | +0.29% | +1.08% | +2.39% | +1.99% | +0.79% |
| **低活用户** | **+1.74%** | **+3.64%** | **+8.16%** | **+4.54%** | **+2.94%** |

- 低活用户获益最大（活跃天数 +1.74%），说明更强的模型能更好地理解行为稀疏的用户

**抖音极速版**: 活跃天数 +0.20%、时长 +0.99%

**广告场景（Table 5）**: AUC +0.73%、广告主价值（ADVV）**+3.90%**

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **硬件感知的设计哲学**: 不是单纯追求模型效果，而是从 MFU（硬件利用率）出发设计架构，"用更多参数换更少计算"的思路非常适合工业部署场景。MFU 从 4.5% 到 45% 的 10 倍提升是非常实质性的工程突破
2. **Multi-head Token Mixing 的简洁高效**: 无参数的跨 token 特征交互，复杂度 $O(TD)$，效果却能接近甚至超过 self-attention，展现了"简单设计 + 充分训练"的威力
3. **DTSI-MoE 策略**: "训练时密集、推理时稀疏"巧妙解决了 Sparse-MoE 的 expert under-training 问题，是一个可广泛迁移的工程技巧
4. **完整的工业级验证**: 从离线实验到 scaling law 分析，再到抖音全场景（Feed + 广告 + 极速版）的线上验证，形成了完整的证据链
5. **对低活用户的显著提升**: 低活用户活跃天数 +1.74%，说明大模型对长尾用户的建模能力有本质提升，这对平台的用户增长极具价值

### 4.2 Limitations & Improvements (不足与改进方向)

1. **Token Mixing 的可学习性不足**: Multi-head Token Mixing 完全无参数，虽然高效但缺乏可学习性。后续工作 UniMixer 已经在这个方向上进行了改进，通过参数化 TokenMixer 获得更强的表达能力
2. **仅两层的架构局限**: 实验配置中 $L=2$，未深入探索更深架构的效果。结合 SiameseNorm 等深层训练技巧，更深的 RankMixer 可能有更大潜力
3. **缺乏公开数据集验证**: 所有实验在字节内部数据上进行，难以被学术界复现和对比
4. **MoE 的负载均衡**: 虽然 ReLU routing 天然稀疏，但论文未深入讨论 expert 的负载均衡问题和长尾 expert 的利用率
5. **可能的改进方向**:
   - 结合 UniMixer 的参数化思路，在 Token Mixing 中引入可学习的混合权重
   - 探索跨场景（Feed + 广告）的统一大模型，利用 MoE 实现场景间的知识共享
   - 将 DTSI-MoE 策略与 KD（知识蒸馏）结合，进一步压缩推理成本
