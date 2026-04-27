# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

> **Authors**: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin
> **Affiliation**: Qwen Team, Alibaba Group 等
> **Venue**: arXiv preprint, May 2025 (arXiv:2505.06708)
> **Links**: [arXiv](https://arxiv.org/abs/2505.06708) · [PDF](https://arxiv.org/pdf/2505.06708)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 标准 softmax attention 存在三大顽疾：(1) $QK^\top V$ 本质上是低秩线性映射，缺乏非线性表达；(2) 注意力权重被 softmax 归一化后虽然"软稀疏"但数值上并不真正稀疏，计算浪费严重；(3) "Attention Sink" 现象——模型过度把注意力集中到序列最前面的几个 token（尤其是首个 token），使得长上下文信息利用率下降，也为量化、KV-cache 压缩、长度外推带来困难。作者系统地问：能否用一个极简的 **gating** 模块同时缓解这三个问题？
- **Method (方法)**: 在标准 softmax attention 之上引入 **head-specific sigmoid gate**，关键结论是：**在 SDPA (Scaled Dot-Product Attention) 输出之后、output projection 之前** 施加一个逐元素、按 query 依赖的 sigmoid 门控，效果最好。形式上（简化）：

$$\text{Output} = \big(\sigma(XW_g) \odot \text{SDPA}(Q,K,V)\big) W_o$$

- 门控是 **query 依赖** 且 **head-specific**，因此每个 head 可以独立地"开/关"自己的输出通道；sigmoid 产生天然的 0~1 稀疏激活，给线性 SDPA 输出注入非线性；当某个位置门被关闭（接近 0），该位置即使在 softmax 中获得很大权重，也不会真正贡献输出，从而消解 attention sink。
- **Results (效果)**: 作者在 **1.7B dense** 与 **15B MoE** 两个量级共 30 个变体上、用 **3.5T token** 语料充分训练，观察到：(i) loss 与下游任务一致优于基线，训练更稳定、可用更大学习率；(ii) 首 token 的 attention sink 比例显著下降；(iii) 长上下文外推能力增强；(iv) 开销近似零（只多一个与 $W_o$ 同规模的 $W_g$，可以与 $W_v$ 融合）。代码与权重已开源。

## 2. Detailed Methodology (详细方法解读)

### 2.1 从"注意力为何是低秩线性"谈起

标准多头注意力每个 head 内部：

$$\text{SDPA}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

- 从 $V$ 到 head 输出是一个 **数据依赖的线性变换**：输出是 $V$ 行的凸组合。
- 所以 attention block 的全部非线性几乎都来自 softmax 内部（计算权重用）和后面 FFN 的激活函数，**从 value 路径看几乎没有非线性**。
- 论文的核心直觉：**在 value 传递路径上补一个 gate**，既能引入非线性（类似 GLU 家族），又能按 query 自适应地抑制冗余/异常位置。

### 2.2 Gated Attention 的设计空间

作者系统地枚举了"在哪里加 gate、gate 怎么算、gate 怎么共享"三个维度，穷举出近 30 个变体，主要轴：

1. **加在哪个位置**（按信息流从前到后）：
   - 对 $Q$ 加 gate
   - 对 $K$ 加 gate
   - 对 $V$ 加 gate
   - 对 **softmax 后的 attention score** 加 gate
   - 对 **SDPA 的输出** 加 gate（论文主推）
   - 对 output projection 之后 加 gate
2. **gate 的计算方式**：
   - 由当前 token 的输入 $x$ 计算（query-dependent，即 $g = \sigma(xW_g)$）
   - 由 key/value 侧计算
   - 静态可学习参数（与输入无关）
3. **gate 的粒度**：
   - Head-shared（所有 head 共享一个 gate）
   - **Head-specific**（每个 head 独立 gate，最终胜出）
   - 元素级（element-wise，每个隐藏维度都有独立 gate 值）
4. **激活函数**：
   - **Sigmoid**（主推，数值在 [0,1]）
   - SiLU / Swish（GLU 家族常见）
   - 不带激活的线性 gate

### 2.3 最终方案：SDPA-Output Head-Specific Sigmoid Gate

以单层 attention 为例，给定输入 $X \in \mathbb{R}^{n \times d}$：

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right),\quad H = A V \quad (\text{SDPA 输出})$$

引入一个与输入同源的门控投影 $W_g \in \mathbb{R}^{d \times d}$（按 head 切分）：

$$G = \sigma(XW_g) \in (0,1)^{n \times d}$$

最终输出：

$$\text{Output} = (G \odot H) W_O$$

- $\odot$ 是按元素乘；$G$ 的切分让每个 head 的每个通道都有独立的 0~1 门控值。
- **Query 依赖**：$G$ 来自当前 token 的 $X$，也就是当前 query 位置，因此每个位置都能独立决定"我要不要采纳这次 attention 读出的信息"。
- **Head-specific**：不同 head 看到不同的子空间、承担不同功能（语法、长程、局部等），独立 gate 让它们各自决定是否要 "静音"。

### 2.4 为什么能同时解决三个问题？

**(a) 非线性（Non-linearity）**

- sigmoid 是非线性的，$G \odot H$ 之后整个 attention block 从 value 路径上首次获得 "门控式非线性"。
- 本质上和 GLU / SwiGLU 在 FFN 里做的事类似，只是搬到了 attention 的 value 通路上。
- 作者把这看作对 attention "低秩线性混合" 的补丁：原先 $H$ 只是 $V$ 的凸组合；现在 $G \odot H$ 可以非线性地缩放每个通道，等效秩被放大。

**(b) 稀疏（Sparsity）**

- sigmoid 容易饱和到 0 或 1；大量实际实验中，$G$ 的分布呈现明显双峰，很多位置实际输出接近 0。
- 这带来两方面好处：一是推理时可以做 gate-based skip（门值接近 0 的通道可以跳过 output projection 的对应行计算，潜在稀疏加速）；二是作为正则，让每个 head 专注于自己真正需要的通道。

**(c) 消除 Attention Sink**

- 传统 attention sink 的根本原因：softmax 要求 $\sum_j A_{ij} = 1$，即使当前 query 根本不需要从任何 token 读信息，它也必须把概率质量 "倾倒" 到某个位置——实际观察中这个位置往往是 BOS / 起始 token。
- 有了 SDPA-output gate 之后，query 可以通过把 $G$ 压到接近 0 来**退出这次 attention**：softmax 该怎么归一化就怎么归一化，但最终 $G \odot H \approx 0$，不会污染残差流。
- 因此首 token 不再需要充当 "垃圾桶"，attention sink 现象显著减弱；同时 KV-cache eviction、量化、长度外推都因此更稳。

### 2.5 实现与开销

- $W_g$ 形状与 $W_O$ 前端一致（$d \times d$），新增参数量 ≈ 一个 $W_V$ 的量级。
- 可以把 $W_g$ 与 $W_V$ **合并实现**为一个更宽的线性层再 chunk，工程上近似零开销。
- 没有新超参（除了 sigmoid 本身）；不改 KV-cache 结构；与 GQA / MoE / RoPE / FlashAttention 等正交兼容。

## 3. Experiment Analysis (实验结果解读)

### 3.1 实验设置

- **模型规模**：
  - 1.7B **dense** 模型
  - 15B **Mixture-of-Experts** 模型
- **变体数量**：共 **30 个 gating 变体**，覆盖 2.2 节中"位置 × gate 来源 × 粒度 × 激活函数"的组合
- **训练语料**：3.5T tokens（大规模预训练级别，结论不是玩具实验）
- **评价**：训练 loss、下游基准、long-context 外推、attention sink 可视化、训练稳定性（大 LR 是否爆炸）

### 3.2 主实验结论（与 baseline softmax attention 比较）

1. **全面的 loss / 下游提升**：SDPA-output + head-specific sigmoid gate 在所有规模下稳定优于基线，且在 30 个变体中综合排名第一。
2. **训练更稳定、更抗大学习率**：加 gate 后可用比 baseline 更大的 peak LR 而不发散，意味着实际可更激进地训练。
3. **Attention sink 显著削弱**：可视化显示首 token 的平均注意力份额大幅下降；量化 / KV 压缩更鲁棒。
4. **长上下文外推更好**：在未训练过的更长序列上 perplexity 降低，外推曲线更平。
5. **稀疏度可观**：$G$ 的分布高度双峰，大量通道事实上被关闭。

### 3.3 Ablation 要点（消化版）

作者系统报告了"在哪加、怎么加"的对比，要点：

| Gate 位置 | 效果 | 备注 |
|-----------|------|------|
| 对 $Q$ 加 gate | 改进有限 | 只影响打分，不改 value 路径非线性 |
| 对 $K$ 加 gate | 改进有限 | 同上 |
| 对 $V$ 加 gate | 部分改进 | 已经进入 value 路径，但被 softmax 再平均稀释 |
| 对 softmax 权重加 gate | 不稳定 | 破坏 softmax 的归一化，容易数值不稳 |
| **对 SDPA 输出加 gate** | **最佳** | 同时拿到非线性 + 允许 "退出 attention" |
| 对 output proj 之后加 gate | 次佳 | 已经混合完，再 gate 效果打折 |

| 粒度 | 效果 |
|------|------|
| Head-shared | 明显不如 head-specific |
| **Head-specific** | **最佳** |

| 激活函数 | 效果 |
|----------|------|
| **Sigmoid** | 最佳（天然 0~1，真正"开关") |
| SiLU | 接近但稍弱 |
| 线性（不加激活） | 显著变差（失去门控/稀疏语义） |

### 3.4 对 Attention Sink 的定量观察

- 基线 softmax attention 中，首 token 的平均注意力权重往往远高于其"信息含量"应得的份额，这被之前的工作命名为 attention sink。
- 加了 SDPA-output sigmoid gate 后，作者观察到：
  - 首 token 的 attention 比例下降到接近均匀水平；
  - 中间层的 attention 分布更分散、信息流更有效；
  - 对 KV-cache 的 eviction 策略更鲁棒（丢首 token 不再导致崩溃）；
  - RULER / 长文本评测指标变好。

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **小改动、大收益、零风险**：整个方法只是往 attention 塞一个 sigmoid 门，但同时解了非线性、稀疏、attention sink 三个老大难问题，这种"结构收益 / 工程成本"的比值非常高。
2. **系统性的设计空间搜索**：不是拍脑袋选一种 gating，而是做了 30 变体 × 大规模预训练的详尽消融，把"最佳 gate 放哪"这件事做成了可信结论，而不是偶然现象。
3. **对 attention sink 的机理解释很干净**：把 sink 归因为 softmax 强制归一化的"强制发言"，然后用 gate 给模型一个"闭嘴"的出口，这个解释简洁且有实验支撑，可能会成为后续相关工作的标准解释框架。
4. **和现有栈完全兼容**：不影响 KV-cache、GQA、MoE、RoPE、FlashAttention，也不需要新超参，几乎可以作为 drop-in replacement 推广到所有 Transformer 变体。
5. **开源**：1.7B dense 与 15B MoE 级别的实证 + 权重 + 代码，复现门槛低。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **额外参数仍存在**：虽然可以与 $W_V$ 融合，但 $W_g$ 的参数量不小；在极端小模型或端侧部署下，是否仍然值得，需要更细的 trade-off。
2. **稀疏优势尚未转化为实际加速**：门控稀疏只是"数值上接近 0"，要想变成真正的 FLOPs 节省，需要配套的稀疏 kernel（类似 Top-k / block-sparse），论文未深入。
3. **与显式稀疏 / 线性 attention 的组合**：Gated Attention 本质仍是 $O(n^2)$ softmax，对超长上下文仍贵。它与 Mamba / 线性 attention / Native Sparse Attention 如何叠加，是很自然的下一个问题。
4. **理论刻画较弱**：对"为什么 head-specific sigmoid 恰好是最优解"的理论分析偏实证，期待后续从秩、谱、优化动力学角度给出更干净的解释。
5. **可能的改进方向**：
   - 将 gate 做成 **learnable sparsity target**，直接在训练中约束稀疏度以兑现推理加速；
   - 与 **KV-cache 压缩 / 量化 / eviction** 协同设计，把"无 sink"这一性质转化为端到端的内存和延迟收益；
   - 把该 gate 移植到 **推荐系统中的 attention 组件**（如用户序列建模、target attention），验证同样的非线性/稀疏/sink 优势是否迁移；
   - 在 MoE 场景下与 **expert-gate** 协同（两个 gate 在同一 token 上共同决定信息流），可能形成更强的条件计算结构。
