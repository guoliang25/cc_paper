# TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders

> **Authors**: Yuchen Jiang*, Jie Zhu*, Xintian Han*, Hui Lu*, Kunmin Bai*, Mingyu Yang*, Shikang Wu*, Ruihao Zhang*, Wenlin Zhao*, Shipeng Bai, Sijin Zhou, Huizhi Yang, Tianyi Liu, Wenda Liu, Ziyan Gong, Haoran Ding, Zheng Chai, Deping Xie, Zhe Chen†, Yuchao Zheng†, Peng Xu
> **Affiliation**: ByteDance AML + ByteDance
> **Venue**: arXiv:2602.06563 v2（2026.02）
> **Links**: [arXiv](https://arxiv.org/abs/2602.06563) · [PDF](https://arxiv.org/pdf/2602.06563v2)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: RankMixer（2507.15551）用 **TokenMixer**（无参数 token mixing 替代 self-attention）成功把工业 ranking 的 MFU 从 4.5% 拉到 45%、参数扩到 1B。但当作者团队尝试**把 RankMixer 继续往上推到 7B/15B 参数规模**时，遇到了四个致命瓶颈：**(1) 残差设计破缺**——RankMixer 的 Token Mixing 会改变 token 数 $T\to T'$，原始 token 的语义信息在残差连接里无法完整传递，加上 `x' + x` 的 pre/post mixing token 语义不对齐，性能次优；**(2) 模型不纯**——多年迭代积累的 LHUC / DCNv2 等小算子（memory-bound、低算力高访存）严重拖低整体 MFU；**(3) 深层梯度更新不足**——RankMixer 一般只用 2 层，加深到 8+ 层后训练不稳、梯度消失；**(4) MoE 稀疏化不彻底**——RankMixer 用 ReLU-MoE 走 "Dense Train / Sparse Infer"，训练成本没省；且 ReLU 激活数量不可预测，推理需要截断 fallback 不友好。**一句话**：RankMixer 的设计在 1B 以下有效，但**不适合 7B+ 规模的极端 scaling**。
- **Method (方法)**: 提出 **TokenMixer-Large (TML)**，对 RankMixer 的 TokenMixer block 做系统性进化，核心四件套：**(1) Mixing & Reverting**——两层 TokenMixer 结构，第一层做 token 间混合（$T\to H$），第二层**反转回原维度**（$H\to T$），保证残差前后 token 语义对齐 + 维度一致；**(2) Inter-Residual + Auxiliary Loss**——每 2–3 层加跨层残差 + 浅层 logits 参与 joint loss，缓解深网络梯度消失；**(3) Sparse-Pertoken MoE (S-P MoE)**——把 pertoken SwiGLU 进一步切分成 $E$ 个 sub-expert，top-k + softmax 路由 + shared expert + gate value scaling + down-matrix 小初始化，实现 "**Sparse Train / Sparse Infer**" 统一范式；**(4) Pure Model 哲学**——随着 TML block 加深，LHUC / DCNv2 等历史遗留小算子的收益被 TML 自己吸收，直接**全部移除**，把广告主干 MFU 拉到 **60%**。配套工程优化：自研 MoEPermute / MoEGroupedFFN / MoEUnpermute 算子 + FP8 后训练量化（serving **1.7× 加速**）+ **Token Parallel** 模型并行策略。
- **Results (效果)**: 离线在抖音广告 Feed 扩到 **15B**，电商扩到 **7B**，直播扩到 **4B**；线上广告扩到 **7B**，电商扩到 **4B**，直播扩到 **2B**，覆盖抖音大部分业务，服务数亿用户。**离线**：500M 参数对比 SOTA 时 TML 以 15.1T FLOPs 取得 +1.14% CTCVR AUC，比 RankMixer (+0.03%) 显著；4B S-P MoE 激活 2.3B 参数即可达到全激活 4.6B 的效果。**线上 A/B**（Table 7）：**Feed Ads ADSS +2.0%**、**电商 Order/u +1.66% / GMV/u +2.98%**、**直播 Pay +1.4%**。至此 "TokenMixer 家族" 从 RankMixer 的 1B 时代推进到 7B–15B 时代，是工业 ranking 架构历史上**首次稳定跨过 10B 参数量级**的公开工作。

## 2. Detailed Methodology (详细方法解读)

### 2.1 RankMixer 的四个瓶颈 —— 为什么必须重新设计

TML 论文第 1 节系统梳理了 RankMixer (记为 TokenMixer) 在大规模 scaling 时遇到的四个根本问题：

| 问题 | 本质 | 后果 |
|------|------|------|
| **Sub-optimal Residual** | Token Mixing 让 $T\to T'$，若 $T\ne T'$ 或每层 head 数不一致，残差无法平滑传播 | 深层语义失真、性能次优 |
| **Impure Model** | 历史遗留的 LHUC / DCNv2 等 memory-bound 小算子 | 整体 MFU 被拉低 |
| **Insufficient Deep Gradients** | RankMixer 只有 2 层；加深到 4+ 层就训不稳 | 深模型不收敛 / 浅层学不好 |
| **Inadequate MoE Sparsification** | ReLU-MoE: "Dense Train, Sparse Infer"，激活数不可预测 | 训练成本不省 + 推理需要 fallback |

→ 前三个问题对应 **TML Block 的三个新模块**；第四个问题对应 **Sparse-Pertoken MoE**。

### 2.2 核心创新 1：Mixing & Reverting（解决残差破缺）

**RankMixer 的残差问题**（论文 Eq.5–8）：一层 Mixing 把 $\mathbf{X}\in\mathbb{R}^{T\times D}$ 先按 head 切分成 $T\times H\times(D/H)$，再按 head 重组为 $H\times(T\cdot D/H)$，最后 concat 回去。这个过程中 **token 数从 $T$ 变成 $H$**，每个位置的语义也变了——直接做 `F(x') + x` 残差就会语义错位。

**TML 的解决方案**：设计**两层对称结构**（Eq.9–16）——第一层做 Mixing（$T\to H$），第二层做 **Reverting**（$H\to T$），让 block 的输入输出维度和语义完全一致：

$$
\text{Mixing}:\; \mathbf{X}\in\mathbb{R}^{T\times D}\to\mathbf{H}\in\mathbb{R}^{H\times(T\cdot D/H)}
$$
$$
\text{pSwiGLU on Mixed}:\; \mathbf{H}^\text{next}=\text{Norm}(\text{pSwiGLU}(\mathbf{H})+\mathbf{H})
$$
$$
\text{Reverting}:\; \mathbf{H}^\text{next}\to\mathbf{X}^\text{revert}\in\mathbb{R}^{T\times D}
$$
$$
\text{Output}:\; \mathbf{X}^\text{next}=\text{Norm}(\text{pSwiGLU}(\mathbf{X}^\text{revert})+\mathbf{X})
$$

关键三性质（论文 Sec 4.3 Table 3 消融）：

| 性质 | 含义 |
|------|------|
| **SR (Standard Residual)** | block 间有标准残差 |
| **OTR (Original Token Residual)** | 原始 token 的语义能传播到深层 |
| **TSA (Token Semantic Alignment)** | $F(x')+x$ 里 $x'$ 和 $x$ 的 token 位置语义一致 |

RankMixer 只满足 SR + OTR，**不满足 TSA**；TML 三个都满足——这是 TML 比 RankMixer 稳定训深的根本原因。消融（Table 3）：RankMixer 无 SR 时 −0.20% AUC；加上 OTR 到 +0.03%；TML 三性质齐全达到 +0.13%。

### 2.3 核心创新 2：Inter-Residual + Auxiliary Loss（解决深模型梯度消失）

加深 TML 到 8+ 层会遇到典型的梯度消失问题。TML 从两个方向缓解：

**(a) Inter-Residual（跨层残差）**：每 2–3 层额外加一条残差路径（Fig 2）。低层特征能直接传到高层，缓解"底层梯度逐层衰减"。注意：**最后一层不建议加 inter-residual**——最后一层要做高级特征抽象，额外混入底层噪声反而伤害性能。

**(b) Auxiliary Loss（辅助损失）**：把浅层的 logits 和深层 logits 一起参与 joint loss：
$$\mathcal{L}_\text{total} = \mathcal{L}_\text{final} + \lambda\sum_\ell \mathcal{L}_\ell^\text{aux}$$

让浅层学会"估计深层特征的偏差"，强化浅层参数的训练信号。

**(c) Down-matrix Small Init**：SwiGLU 的最后一层 $W_\text{down}$ 用 xavier_uniform 但 std=0.01（远小于默认 1），让 $F(x)$ 项训练初期近零，模块初始近似恒等——**这和 NLP 里 Adapter / ReZero 的思路完全一致**，保证训练稳定。

### 2.4 核心创新 3：Sparse-Pertoken MoE（实现稀疏训练 + 稀疏推理）

**RankMixer 的 ReLU-MoE 问题**：
- ReLU 激活数量依赖输入，**每个 batch 不可预测**
- 推理时需要 truncation / fallback 兜底
- 训练时仍是 **Dense**（没省算力）

**TML 的 S-P MoE**（Eq.19–21）：

$$
\text{S-P MoE}(\cdot) = \alpha\cdot\sum_{i=1}^{k-1} g_i(\cdot)\cdot\text{Expert}_i(\cdot) + \text{SharedExpert}(\cdot)
$$

其中：
- 每个 pertoken SwiGLU 被切成 $E$ 个 sub-expert（token-specific，不是全局共享）
- **Top-k softmax router** $g(\cdot)$：激活数量固定（训练/推理一致）
- **Shared Expert**：每 token 自己的 shared expert，不是所有 token 共用（这点和 DeepSeek-V2 等 LLM 里的 shared expert 设计不同）
- **Gate Value Scaling** $\alpha$：补偿 softmax 的 sum-to-one 约束对梯度的压制
- **Down-matrix small init**：std 从 1 降到 0.01

**"First Enlarge, Then Sparse"**：先把模型整体扩大（pertoken SwiGLU 变大），再做 sub-expert 切分和稀疏化。这样先吃到参数扩展的收益，再吃到稀疏计算的效率。结果：**激活 1/2 参数即可达到 Dense 同等效果**（Table 2）。

消融（Table 6）：
- 无 Shared Expert：−0.02% AUC
- 无 Gate Value Scaling：损失更大
- 无 Down-matrix small init：训练不稳

### 2.5 核心创新 4：Pure Model（去掉所有 memory-bound 小算子）

**观察**：工业代码库里年代久远的 LHUC（个性化缩放）、DCNv2（显式 cross）、GCN-style 等算子，在 TML 块充分堆叠后**贡献可以被 TML 自己吸收**。这些算子的共同特点是：
- **算力低、访存高**（memory-bound）
- **kernel launch 开销大**
- 对 MFU 的贡献为负

**TML 的做法**：**全部移除**。TML 本身只剩"无参数 mixing/reverting + 大量 GroupedGemm"——**纯粹的 compute-bound 模型**。

**收益**：广告场景的 backbone MFU 从未报告值（RankMixer 是 45%）进一步**拉到 60%**——这是本论文最硬核的工程成就之一。

### 2.6 Tokenization：Semantic Group + Global Token

**Semantic Group-wise Tokenizer**（Eq.1–2）：按特征语义分组（短期 DIN / 长期 SIM / 超长期 LONGER / cross feature / 基础特征），每组独立的 MLP 投影到统一维度 $D$，保持语义异质性。

**Global Token**（Eq.3）：额外引入一个 `[GLOBAL]` token（借鉴 BERT `[CLS]`），MLP 聚合全部组后得到，作为全局信息枢纽并传播到其它 token。

最终输入：$\mathbf{X}=\text{concat}[\mathbf{X}_G, \mathbf{X}_0, \ldots, \mathbf{X}_{T-1}]\in\mathbb{R}^{T\times D}$（Eq.4）。

消融：无 Global Token −0.02% AUC（小但稳定的收益）。

### 2.7 工程优化栈

**(a) High-Performance Custom Operators**（Fig 3 + Table 1）：
- `MoEPermute`：把输入从 batch-first 变 expert-first（保证每个 expert 输入连续）
- `MoEGroupedFFN` (GroupedSwiglu + GroupedGemm)：**一个 kernel 算所有 expert**，占训练时间 89.18%、推理时间 98.35%
- `MoEUnpermute`：合并多个 expert 输出

**(b) FP8 Quantization**：
- 训练保持 BF16，推理用 **FP8 E4M3 post-training quantization**
- 实测 serving **1.7× 加速**，精度无损

**(c) Token Parallel**（Sec 3.5.3）：
- 针对 TML "Mixing → pertoken → Reverting → pertoken" 的流程专门设计的**模型并行策略**
- 把 pertoken 权重按 token 维度切到多个 device
- 通过 2 次 `all2all` 完成数据重排（而非朴素模型并行的 4 次）
- 与 TML 的数据流严丝合缝

## 3. Experiment Analysis (实验结果解读)

### 3.1 ~500M 参数同级对比（Table 2）

| Model | ΔAUC (CTCVR) | Params | FLOPs/Batch |
|-------|-------------|--------|-------------|
| DLRM-MLP-500M（baseline）| — | 499 M | 125.1 T |
| DCNv2 | — | ~500M | — |
| HiFormer | — | ~500M | — |
| Wukong | — | ~500M | — |
| Group Transformer | — | ~500M | ~4.5T |
| RankMixer | +0.03% | 567 M | 4.6 T |
| **TokenMixer-Large 7B** | **+1.20%** | 7.6 B | 49.0 T |
| **TokenMixer-Large 4B S-P MoE** | **+1.14%** | 2.3B **in** 4.6B | **15.1 T** |

→ S-P MoE 版本**参数 4.6B、激活 2.3B、FLOPs 15.1T**，打出 +1.14% AUC——在 FLOPs 效率和性能的帕累托前沿上大幅领先。

### 3.2 与 RankMixer 的细粒度对比（Table 3）

| Model | SR | OTR | TSA | ΔAUC | Params | FLOPs |
|-------|----|----|----|------|--------|-------|
| Group Transformer | ✓ | ✓ | ✓ | — | 500M | 4.5T |
| RankMixer w/o SR&OTR | ✗ | ✗ | ✗ | −0.20% | 510M | 4.2T |
| RankMixer w/o OTR | ✓ | ✗ | ✗ | −0.13% | 510M | 4.2T |
| RankMixer | ✓ | ✓ | ✗ | +0.03% | 567M | 4.6T |
| **TokenMixer-Large** | ✓ | ✓ | ✓ | **+0.13%** | 500M | 4.2T |

→ 三个残差性质缺一少一都掉点；只有 TML 三者齐全，且参数和 FLOPs 都更小。

### 3.3 Scaling Law（Figure 4 + 5）

三个业务场景分别 scaling 到：

| 场景 | 离线最大规模 | 线上最大规模 |
|------|-------------|-------------|
| Feed Ads | **15B** | 7B |
| E-Commerce | 7B | 4B |
| Live Streaming | 4B | 2B |

两条关键 scaling 发现：
1. **超过 1B 后"平衡扩展"更重要**：单独扩 width / depth / scaling factor 都会先涨后瓶颈；需要**三者同步扩大**才能继续拿收益
2. **大模型需要更多数据**：30M → 90M 只需 14 天训练收敛；500M → 2B 需要 60 天才能收敛（Table 4）。2.3B 用 30 天 ΔUAUC +0.41%，用 60 天 +0.70%——**数据量是深大模型收敛的必要条件**

### 3.4 模块消融（Table 5，TML-4B）

| Ablation | ΔAUC |
|----------|------|
| w/o Global Token | −0.02% |
| **w/o Mixing & Reverting** | **−0.27%** |
| w/o Residual | −0.15% |
| w/o Internal Residual & AuxLoss | −0.04% |
| Pertoken SwiGLU → SwiGLU（去 pertoken）| **−0.21%** |
| Pertoken SwiGLU → Pertoken FFN（去 SwiGLU）| −0.10% |

→ **最关键的两个模块：Mixing & Reverting (−0.27%)** 和 **Pertoken SwiGLU (−0.21%)**。Mixing & Reverting 解决残差破缺的价值在数据上铁证如山。

### 3.5 线上 A/B（Table 7）

| 场景 | ΔAUC | Core Metric |
|------|------|-------------|
| **Feed Ads** | +0.35% | **ADSS +2.0%** |
| **E-Commerce** | +0.51% | **Order +1.66% / GMV +2.98%** |
| **Live Streaming** | +0.7% UAUC | **Pay +1.4%** |

→ 三个场景都是显著 p<0.05 线上收益，覆盖抖音大部分业务。

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **把 "TokenMixer 家族" 从 1B 推进到 15B**：这是工业 ranking 首次稳定跨越 10B 参数的公开工作。在 OneTrans / MixFormer 等 "统一 backbone" 路线之外，**证明了 TokenMixer 非 attention 路线也能 scaling 到这个量级**。
2. **问题定义极其清晰**：开篇就把 RankMixer 的四个瓶颈（残差破缺 / 模型不纯 / 深层梯度 / MoE 稀疏不彻底）摊开——每个问题对应一个解决方案，**整篇论文结构是围绕 "修复 RankMixer"** 来组织的，教科书级的工业论文写法。
3. **Mixing & Reverting 是个漂亮的结构设计**：通过两层对称结构（mixing → reverting）同时解决"token 数变化"和"残差语义对齐"两个问题。TSA（Token Semantic Alignment）这一维度是作者在 Table 3 里显式提出来的**新概念**，把一个模糊的"残差哪里不对"问题精确化。
4. **S-P MoE 的 "First Enlarge, Then Sparse" 策略**：先把 pertoken SwiGLU 整体放大、再拆 sub-expert 稀疏化——**两个阶段分别吃参数扩展的收益和稀疏计算的效率**。这个工程路线图对实际做 MoE 落地的团队很有借鉴意义。
5. **Pure Model 哲学 + MFU 60%**：在 RankMixer 已经 45% MFU 基础上，**靠"移除历史遗留小算子"再拉 15 个点**——这是对"不够纯"的工业代码的一次彻底清洗。值得每个工业团队反思自己模型里还有多少 memory-bound 历史包袱。
6. **工程栈完整**：自研 fused operators + FP8 量化 + Token Parallel，**训推一致 + serving 1.7× 加速**。这一套打下来，TML 在 7B–15B 规模下能上线是必然的。
7. **大规模 scaling 的经验性发现**：超过 1B 后必须 "平衡扩展 width/depth/scaling factor"；大模型需要更多数据收敛——这两条都是字节内部长期跑大模型才能攒出来的一手经验。
8. **线上效果数字硬**：广告 ADSS +2.0%、电商 GMV +2.98%、直播 Pay +1.4%——这种量级的收益在已经非常成熟的主场景上非常难拿。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **仍未解决 "序列建模怎么融入" 的问题**：TML 的主战场还是 "特征交互侧" 的 scaling——它把 FI 侧做到 15B，**但序列建模仍然外挂 DIN/SIM/LONGER**（论文 Fig 1 明确说"raw tokens include features from sequence aggregation such as DIN/LONGER"）。TML **没有吸收 OneTrans/MixFormer 的"统一 backbone"哲学**。下一代自然方向：**TML-Large × OneTrans 混合** = 15B 的 FI 侧 + unified S/NS backbone。
2. **Mixing 的等价性论证偏弱**：Mixing & Reverting 的"对称性"好处在 Table 3 里有消融，但**为什么两层 TokenMixer 就能 reverting 回原维度**——理论上是 permutation-invariant 的论证没展开，读者需要自己脑补。
3. **Global Token 收益较小**（−0.02%）：对比 LONGER 里 Global Token 的关键作用（缓解 attention sink、稳 attention 分布），TML 里 Global Token 的收益小得多——**可能因为 TokenMixer 本身没有 attention sink 问题**，那 Global Token 是不是还需要存在就成了 open question。
4. **Data scale 依赖**：Table 4 显示 2.3B 模型要 60 天数据才能充分收敛——**对数据量不足的场景（小业务 / 新业务 / 冷启动）这条路线可能走不通**。需要配合预训练 + 迁移学习等策略（类似 HLLM / MoRec）。
5. **缺乏和 OneTrans/MixFormer 的直接对比**：TML 的主要对比是 RankMixer、HiFormer、DCNv2、Wukong 等"FI 派"，**没有和同时期的 OneTrans（WWW'26）/ MixFormer（2602.14110）这两个"统一 backbone"派**同规模对比。实际上 OneTrans-L 330M 就已经打过 RankMixer，TML-7B vs OneTrans-7B 是哪家更强，论文不给答案。
6. **MFU 60% 的上限**：TML 已经把 MFU 推到 60%，但相比 LLM 预训练常见的 60–70% MFU 还有一定差距。剩下 10–20% 的空间如何填（kernel 融合 / 数据加载 / 通信重叠）是工程上下一个课题。
7. **可能的改进方向**：
   - **TML × OneTrans 混合路线**：FI 侧用 TML 的 Mixing-Reverting + S-P MoE 扩到 15B，序列侧用 OneTrans 的 Mixed Parameterization + Cross-Request KV Cache，结合成下一代工业 ranking backbone
   - **TML 和 LLM-based Item Encoder 协同**：类似 HLLM 用 Item LLM 离线预提 embedding 喂给 TML 做 FI，把内容理解和特征交互彻底解耦
   - **把 MoE 跟 Adapter 思想结合**：每场景/任务一组 Adapter expert + 共享 backbone expert，实现 "统一 backbone + 可插拔任务/场景" 的进一步升级
   - **探索 "无 LHUC、无 DCNv2" 之后的下一步 Pure 模型**：TML 已经砍掉 memory-bound 小算子，但 embedding 查表 / MoE permute 仍是 memory-bound 大头，**如何进一步纯化**是开放问题

### 4.3 在字节 Ranking 演化线上的位置

把 TML 放在之前整理的字节四大架构对比图里：

```
2025.05  LONGER       (SM 侧 scaling)
2025.07  RankMixer    (FI 侧 scaling, 1B, MFU 45%)
2025.10  OneTrans     (unified backbone)
2026.02  MixFormer    (精细版 unified)
2026.02  TokenMixer-Large  ← 本文 (FI 侧 scaling 到 15B, MFU 60%)
```

TML 是 **RankMixer 的直接继任者**——"同门同路线的下一代"。它**不否定 RankMixer 的核心哲学**（Token Mixing 替代 attention、pertoken FFN、硬件感知），而是**在这条路线上把所有能优化的细节都修了一遍，把规模天花板从 1B 推到 15B**。

从"哲学分野"看：
- **OneTrans / MixFormer** = "统一 backbone 派"（FI 和 SM 合为一体）
- **TokenMixer-Large** = "极限 scaling 派"（专注 FI 侧但把规模推到极致）

两条路线在 2026 年并行推进，**下一代工业 ranking 架构大概率是两者融合的产物**——用 TML 的 Mixing-Reverting + S-P MoE 做 FI 侧的主干 15B 级 scaling，用 OneTrans 的 Mixed Parameterization + Cross-Request KV Cache 做 SM 侧的统一 token 流。
