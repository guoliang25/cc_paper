# Parameter-Efficient Transfer Learning for NLP（Adapter 奠基之作）

> **Authors**: Neil Houlsby, Andrei Giurgiu*, Stanisław Jastrzȩbski*, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly
> **Affiliation**: Google Research（+ Jagiellonian University）
> **Venue**: **ICML 2019**（arXiv:1902.00751，v2 · 2019.06）
> **Links**: [arXiv](https://arxiv.org/abs/1902.00751) · [PDF](https://arxiv.org/pdf/1902.00751) · [Code (google-research/adapter-bert)](https://github.com/google-research/adapter-bert)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: BERT 等大规模预训练模型已成为 NLP 主流迁移学习方式，但**"full fine-tuning"有两个致命缺陷**：(1) 每个下游任务都要保存一份完整参数副本，$N$ 个任务就是 $N\times|w|$ 的存储成本——对云服务这种"任务流式到达"的场景完全不可持续；(2) 新任务训练会污染旧任务的参数，不支持"incremental / 在线学习"。**问题一句话**：能不能设计一种迁移方法，**每个新任务只加极少参数、不动预训练权重、且性能接近 full fine-tune**？
- **Method (方法)**: 提出 **Adapter 模块**——在每个 Transformer 子层（Multi-Head Attention 和 FFN）之后、残差相加之前**串联一个小型 bottleneck 网络**：down-project 到维度 $m\ll d$ → 非线性 → up-project 回 $d$ 维，带一个 skip-connection。down-project 权重**近零初始化**，使得 Adapter 初始行为 ≈ 恒等变换（不破坏预训练模型）。训练时**只更新 Adapter 权重、Layer Norm 参数和分类头**，冻结 BERT 全部 330M 参数。Adapter 的 bottleneck 维度 $m$ 是唯一的任务超参，可在 $\{2, 4, 8, 16, 32, 64, 256\}$ 间调。
- **Results (效果)**: 在 **GLUE** 9 个任务上，Adapter 平均分 **80.0 vs full fine-tune 80.4**（仅差 **0.4%**），但每任务只加 **3.6%** 参数；在另外 **17 个公开文本分类任务**上同样接近 full fine-tune；**SQuAD v1.1** 上 Adapter(size=64) 达 F1 **90.4** vs fine-tune 90.7，甚至 size=2（0.1% 参数）都有 F1 89.9。**9 个任务总参数开销**：full fine-tune 需 $9\times$ BERT 参数，Adapter 仅需 $1.3\times$。在 17 任务上更夸张——fine-tune 需 $17\times$，Adapter 仅 $1.19\times$（每任务 1.14% 新参数）。这是 **PEFT（Parameter-Efficient Fine-Tuning）家族的开山之作**，后续 LoRA / Prefix Tuning / Prompt Tuning / P-Tuning / BitFit 都是在这个框架下的演化。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Adapter 的数学与架构

![Adapter 模块结构及其在 Transformer 中的位置（来源 arXiv 1902.00751）](https://arxiv.org/html/1902.00751v2/x1.png)

设 Transformer 的某个子层（如 MHA 后的 projection）输出维度为 $d$。一个 Adapter 模块定义为：

$$
\text{Adapter}(h) = h + W_\text{up}\,\sigma(W_\text{down}\,h)
$$

其中：
- $h \in \mathbb{R}^d$：子层输出
- $W_\text{down} \in \mathbb{R}^{m \times d}$：down-projection（bottleneck dimension $m \ll d$）
- $W_\text{up} \in \mathbb{R}^{d \times m}$：up-projection
- $\sigma(\cdot)$：非线性激活（GeLU / ReLU）
- **外层 skip-connection** $h + (\cdot)$ 是 Adapter 的**核心设计**——让模块初始化时近似恒等

**参数量**（加上 bias）：
$$
|\text{Adapter}| = 2md + d + m
$$

当 $m = d/16$（典型值），Adapter 参数量约为一个 FFN 的 $1/32$ 量级。

### 2.2 在 Transformer 里"插两次"

每个 Transformer 层有两个 sub-layer：MHA + FFN。Adapter 在这两处各插一次：

```
Input
  │
  ▼
Multi-Head Attention
  │
  ▼
[Projection 回到 d 维]
  │
  ▼
★ Adapter #1 ★
  │
  ▼
Add & LayerNorm  ← 残差连接
  │
  ▼
Feed-Forward (2 层)
  │
  ▼
★ Adapter #2 ★
  │
  ▼
Add & LayerNorm
  │
  ▼
Output
```

**注意两个关键细节**：
1. Adapter 插在 **"projection 回到 $d$ 之后、残差相加之前"**——这样 Adapter 的输入/输出维度都是 $d$，接口统一；
2. Adapter **内部**已有 skip-connection，**外部**还有 Transformer 原生残差——两层嵌套。

### 2.3 Near-Identity 初始化：稳定训练的关键

Adapter 的核心设计哲学是 **"初始化时等价于没插入"**：

- $W_\text{down}$、$W_\text{up}$ **都近零初始化**（论文用 zero-mean Gaussian，std $=10^{-2}$，截断到 2σ）
- 这样初始 $W_\text{up}\sigma(W_\text{down}h) \approx 0$，Adapter 近似输出 $h$ 本身（恒等）
- **后果**：训练开始时整个带 Adapter 的网络等价于原 BERT，预训练权重完全不被扰动；之后 Adapter 逐步"激活"，调整局部分布

论文做了初始化尺度消融（Figure 6 右）——**std $\le 10^{-2}$ 都稳，$>10^{-2}$ 开始不稳**，CoLA 比 MNLI 更敏感。

### 2.4 训练时更新哪些参数

冻结：
- ❌ BERT 所有 Attention / FFN 权重（330M，占大头）

训练：
- ✅ 每层两个 Adapter 的 $W_\text{down}, W_\text{up}$ 及 bias
- ✅ 每层的 **Layer Normalization** 参数（$\gamma, \beta$，每层 $2d$ 个）
- ✅ 最后的任务分类头

### 2.5 和已有方法的概念对比

| 方法 | 参数规模 | 任务隔离 | 在线增量 | 性能 |
|------|---------|---------|---------|------|
| **Feature-based**（冻结 BERT 提 feature） | $\sim d$/任务 | ✅ | ✅ | 差（论文基线 TS） |
| **Full fine-tune** | $|w|$/任务 | ❌ 共享底层时 | ❌ | 最强 |
| **Top-$k$ fine-tune**（只训顶部 $k$ 层） | $0.5\sim 1 |w|$ | 部分 | 部分 | 接近 full |
| **LayerNorm-only**（只调 LN） | $2d L$ | ✅ | ✅ | 不足 |
| **Adapter**（本文）| **$\sim 3\%|w|$/任务** | ✅ 完美 | ✅ 完美 | **接近 full** |

## 3. Experiment Analysis (实验结果解读)

### 3.1 GLUE Benchmark（Table 1，用 BERT-Large 24 层）

| 方法 | 总参数 | 每任务新参数 | 平均 GLUE 分 |
|------|--------|-------------|-------------|
| Full Fine-tune（9 个任务）| **9.0×** | 100% | **80.4** |
| Adapters (size 8–256) | **1.3×** | **3.6%** | **80.0** |
| Adapters (fixed size 64) | 1.2× | 2.1% | 79.6 |

→ Adapter 只损失 **0.4 分** GLUE，但总参数从 9× 降到 1.3×。不同任务最优 bottleneck 不同——MNLI 选 256，RTE 选 8（样本少任务用小 Adapter）。

### 3.2 17 个额外文本分类任务（Table 2，BERT-Base 12 层）

| 方法 | 总参数 | 每任务新参数 |
|------|--------|-------------|
| Full Fine-tune | **17×** | 100% |
| Variable Fine-tune（只训顶部 $n$ 层）| 9.9× | ~52% | 稍好于 full |
| **Adapters** | **1.19×** | **1.14%** | 与 full 相差 0.4% |

**关键发现**：
- 即使只多 ~1% 参数，Adapter 能打到与 full fine-tune 相差 0.4% 的水平
- AutoML baseline（搜 10k+ 模型结构）平均表现**不如**基于 BERT 的方法——再次验证"pre-train + adapt" > "from-scratch AutoML"

### 3.3 SQuAD v1.1 抽取式问答（Figure 5）

- Full fine-tune：F1 = **90.7**
- Adapter(size=64, 2% 参数)：F1 = **90.4**
- Adapter(size=2, 0.1% 参数)：F1 = **89.9**

SQuAD 对 Adapter 尤其友好，连 size=2 都几乎追平 full——说明对于信息相对集中的任务，超小 Adapter 就够用。

### 3.4 消融：哪些层的 Adapter 最重要（Figure 6 左/中）

对训练好的模型**移除某些层的 Adapter**，重新 evaluate（不重训）：

- **单层 Adapter 移除**：最大下降 **2%**——单层影响不大
- **所有 Adapter 全移除**：MNLI 从 84% 跌到 **37%**（退化到多数类基线），CoLA 跌到 69%——**整体影响很大**
- **底层（0–4 层）Adapter 移除**：MNLI 性能**几乎不变**——说明 Adapter 主要在**高层**起作用

→ **Adapter 自动聚焦到高层**，这和"低层学通用特征、高层学任务特征"的经验吻合，也解释了为什么"只调顶部 $k$ 层"的 fine-tune 在某些任务上甚至略胜 full fine-tune。

### 3.5 鲁棒性分析

- **Adapter size 的鲁棒性**（Figures 3, 4, 5）：8/64/256 三档的验证准确率分别为 86.2 / 85.8 / 85.7——**尺寸变化几乎不影响性能**，跨几个数量级都稳定
- **初始化尺度**：std ∈ $[10^{-7}, 10^{-2}]$ 都稳；std > $10^{-2}$ 开始不稳（Figure 6 右）

### 3.6 论文尝试但未采纳的改进（节选自 3.6 节）

作者诚实列出了**试过但没提升**的 Adapter 变体：
1. 在 Adapter 内加 Batch/Layer Norm
2. 增加 Adapter 的层数（从 2 层加深）
3. 把激活从 GeLU 换成 tanh
4. 只在 Attention 层插 Adapter，不在 FFN 插
5. Adapter **并联**（parallel）而非串联
6. 带乘法交互的 Adapter

→ "All cases we observed the resulting performance to be similar to the bottleneck proposed"——**最简单的 down-up bottleneck 就是最优解**，这个结论直接决定了后续 LoRA / Prefix Tuning 等方法的设计空间。

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **PEFT 时代的开山之作**：Adapter 是第一个系统性地证明"**冻结预训练主干 + 小模块微调**"能打平 full fine-tune 的工作。2021 年后整个 PEFT 生态（LoRA、Prefix Tuning、Prompt Tuning、P-Tuning、IA³、BitFit）都在这个框架下发展。
2. **极简的设计哲学贯穿全文**：Adapter = bottleneck + skip-connection + near-identity init。三句话讲完的设计，却在 26 个任务上证明够用。作者甚至在 3.6 节诚实报告"试过所有复杂变种都没提升"——**承认简单即最优**。
3. **Near-Identity 初始化**：零初始化 $W_\text{down}$/$W_\text{up}$ 让 Adapter 启动时等价于没插入，**训练初期预训练权重完全不被扰动**。这是所有后续 PEFT 方法的共同要求（LoRA 也是把 $B$ 矩阵零初始化）。
4. **两塔接口完美**：Adapter 的输入输出都是 $d$ 维，和 Transformer 的残差接口天然兼容——不需要改 attention mask、不需要改 positional encoding、不需要改 LM head。**接口洁癖让这个模块能无痛移植到任何 Transformer**。
5. **任务隔离 + 在线增量**：冻结主干 + 每任务独立 Adapter → 天然支持"流式到达的任务"（云服务典型场景），新任务不会污染旧任务。这是 full fine-tune 完全做不到的。
6. **自动定位高层**：Ablation 揭示 Adapter 主要影响高层——无需手工指定"只调顶部 $k$ 层"，Adapter 自己就知道该在哪使劲。这省了一个重要超参。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **串联 Adapter 带来推理延迟**：每个 Transformer 层多插两个 Adapter → 推理时多 2 次串行矩阵乘（无法与原 attention/FFN 并行）。**LoRA (2021)** 的关键贡献正是把 Adapter 改成"可合并到原权重的低秩增量"，推理零额外延迟——这是 Adapter 到 LoRA 的核心进化。
2. **只验证了文本分类和抽取式 QA**：生成任务（翻译、摘要、对话）、多模态任务、RL 场景都没测。后续 **Prefix Tuning (2021)** 把 PEFT 拓展到生成任务，**MAD-X (2020)** 拓展到跨语言迁移。
3. **Bottleneck size $m$ 仍是超参**：不同任务最优 size 差 30×（RTE 选 8、MNLI 选 256）——虽然论文发现 "fixed 64" 损失不大，但理论上最优 size 和任务难度的关系没刻画。LoRA 的 rank 也有同样问题。
4. **和 pretraining-fine-tune 范式深绑定**：当时背景是 BERT 的 MLM 预训练；当下 LLM 的 instruction tuning / RLHF 场景下，Adapter 的地位被 **LoRA + QLoRA / PEFT 库** 接管了，但核心思想一脉相承。
5. **仅 encoder，没测 encoder-decoder / decoder-only**：BART / T5 / GPT 的 Adapter 方案需要另行验证（后来 **MAM Adapter**、**Compacter** 等工作补上了）。
6. **可能的改进方向**（后世已实现的）：
   - **LoRA (2021)**：用低秩分解 $W = W_0 + BA$（$B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times d}$）代替 bottleneck，推理可合并
   - **Prefix Tuning (2021)**：在 Attention 的 K/V 前拼接可学习 prefix，完全不改 FFN
   - **Compacter (2021)**：用 Kronecker 乘积进一步压缩 Adapter 参数
   - **IA³ (2022)**：只学 3 个缩放向量（key/value/ffn），比 Adapter 再小一个数量级
   - **AdapterFusion (2020)**：多任务 Adapter 融合，用 attention 组合不同任务学到的 Adapter

### 4.3 Adapter 思想在推荐系统里的映射

这篇论文虽然在 NLP 领域提出，但**思想可以直接迁移到 RecSys**：

1. **多任务 Ranking**（同一个 backbone 服务 CTR / CVR / 停留时长 / 完播率）—— 任务专属 Adapter 可以替代任务塔
2. **多场景 Ranking**（Feeds / Mall / Live Streaming / 搜索）—— 场景专属 Adapter 让共享 backbone 无感切换场景
3. **HLLM 式 Item Encoder 的任务适配** —— Item LLM 固定，每个下游 Ranking 任务学一组 Adapter，避免对全量 LLM 做任务 fine-tune
4. **冷启动 / 新行业快速接入** —— 新行业只需训一个 Adapter，继承已有 backbone 的 scaling 红利（对比 RankMixer / OneTrans 这类统一 backbone 范式，Adapter 是天然的"行业扩展口"）
