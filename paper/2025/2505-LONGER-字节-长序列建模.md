# LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders

> **Authors**: Zheng Chai*, Qin Ren*, Xijun Xiao*, Huizhi Yang*, Bo Han*, Sijun Zhang, Di Chen, Hui Lu, Wenlin Zhao, Lele Yu, Xionghang Xie, Shiru Ren, Xiang Sun, Yaocheng Tan, Peng Xu, Yuchao Zheng†, Di Wu
> **Affiliation**: ByteDance (北京 / 杭州 / 上海 / 圣何塞)
> **Venue**: RecSys '25 (Prague, Sep 22–26 2025)，arXiv:2505.04421 (2025.05)
> **Links**: [arXiv](https://arxiv.org/abs/2505.04421) · [PDF](https://arxiv.org/pdf/2505.04421) · DOI 10.1145/3705328.3748065

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 工业推荐对"用户超长行为序列"的利用长期卡在**三种折衷方案**上——(1) **两阶段检索**（SIM / TWIN），先从超长序列里 top-k 召回再喂给下游短序列模型，上下游不一致、信息丢失；(2) **预训练 user embedding**（UE），离线把长序列压成一个定长向量再迁移，下游模型看不到原始序列；(3) **Memory-augmented 模型**（MIMN / LMN / MARM），靠外部 memory 慢慢"攒" hit，训练周期长、结构复杂。根本问题是**序列从未端到端地、完整地**进入模型。随着 GPU 算力和 scaling law 的成熟，作者问：**能不能直接端到端建模 10k 级别的长序列，同时满足工业级的延迟/吞吐/GPU 效率？**
- **Method (方法)**: 提出 **LONGER = Long-sequence Optimized traNsformer for GPU-Efficient Recommenders**。核心四件套：(1) **Global Tokens**——把候选物品表示、UID、CLS、交叉特征做成"锚点 token"前置，拥有全局感受野，**稳定长序列 attention 分布、缓解 attention sink**；(2) **Token Merge + InnerTrans**——每 K 个相邻历史 token 合并成 1 个，用小 Transformer 做组内交互（保持细粒度），把 $O(L^2)$ 压低，同时参数量按 $K^2$ 增长，相当于用"组合爆炸式"参数换掉注意力算力；(3) **Hybrid Causal Attention**——第一层用 **cross-causal attention**（query = global tokens + 最近 k 个历史 token，K/V = 全序列）做"查询压缩"，后续 N 层用 **self-causal attention** 堆叠建高阶交互；(4) **系统工程优化**——全同步 Dense+Sparse GPU 参数存储（HBM/MEM/SSD 分层）、BF16/FP16 混合精度 + activation recompute、KV Cache Serving（跨候选共享用户序列 KV）。
- **Results (效果)**: 在 **52 亿样本、130 天** 的抖音广告 CVR 数据上离线 AUC **0.85290**，比 Base 基线 **+1.57% AUC**、比最强 Transformer baseline 再提 **+0.21% AUC**（工业场景 0.1% 即显著）；FLOPs 只用 Token Merge 就从 3.73 × 10⁹ 降到 3.03 × 10⁹（**−19%**），加 query-sampling 至 100 只剩 1.91 × 10⁹（**−49%**），且 AUC 更高。线上 A/B：**抖音广告** Live Streaming ADVV **+1.17%**、Short Video ADVV **+2.15%**、Mall ADVV **+1.41%**；**抖音电商** Live Streaming GMV/u **+6.54%、Order/u +7.92%**，Short Video GMV/u **+5.28%**。KV Cache 让 serving 吞吐退化从 −40% 救到 −6.8%。现已在**字节几十个业务场景部署**，服务数十亿用户。

## 2. Detailed Methodology (详细方法解读)

### 2.1 为什么要端到端建长序列

> 工业 *de facto* 的三条现状路径都是"绕路"，**根因是把长序列从主模型里踢了出去**：

| 现有做法 | 代表 | 问题 |
|----------|------|------|
| 两阶段检索（top-k） | SIM, TWIN | 上下游不一致、候选相关的 top-k 必然丢长程信息 |
| Pre-trained User Embedding | BERT4Rec-style UE | 下游只看到一个压缩向量，丢失原序列 |
| Memory-Augmented | MIMN, LMN, MARM | memory 命中率需要长训练期，且是间接的感知 |

→ LONGER 的立场：**随着 GPU/框架/scaling law 成熟，直接把 10k 长序列塞进主模型端到端训，已经可行**。

### 2.2 任务形式化

CVR/CTR 二分类：给定用户历史 $S_u=[i_1^{(u)},\ldots,i_L^{(u)}]$、用户基础特征 $u_d$、候选 $v$：
$$P(y=1\mid S_u, u_d, v)\in[0,1]\tag{Eq.1}$$
BCE Loss：
$$\mathcal{L}=-\frac{1}{|\mathcal{D}|}\sum_{(S_u,u_d,v,y)}\big[y\log\hat y+(1-y)\log(1-\hat y)\big]\tag{Eq.2}$$

整体架构（Fig.1）：**Global Tokens + User Long Sequence → Token Merge (+ InnerTrans) → Cross Attention → Self Attention × N → Concat → MLP → Prediction**。

### 2.3 模块一：Global Tokens —— 长序列的"锚点"

把 $m$ 个特殊 token 前置到序列首部，每个都有**全局感受野**（能看全序列、也被全序列看到）。内容典型包括：
- 候选物品表示 token（让模型"知道要评什么"）
- 可学习 CLS token
- UID embedding
- 高阶用户-物品交叉特征 token

**两大作用**：
1. **信息枢纽**：让用户历史、上下文、候选在 attention 里显式汇合；
2. **稳定 attention 分布 / 缓解 attention sink**（借鉴 StreamingLLM 的发现）——深层 attention 会过度聚焦到靠前 token，加了 global anchors 之后注意力更分散、长程依赖保留更好。

### 2.4 模块二：Token Merge + InnerTrans

设序列长度 $L$、embedding 维 $d$（典型工业场景 $L=2000, d=32$，即 $L\gg d$，quadratic 复杂度 $O(L^2d)$ 非常致命）。

**Token Merge**：相邻 $K$ 个 token → 合成 1 个，序列长度缩到 $L/K$，按 $K$ 倍空间压缩。合并方式两种：
- 简单 **concat**（$K$ 个 $d$ 维向量拼成 $Kd$ 维）；
- **InnerTrans**（组内小 Transformer）：
$$\mathbf{M}_i=\text{TransformerBlock}([e_{i1},\ldots,e_{iK}])\tag{Eq.6}$$
  保证组内 token 间交互，避免 concat 的"硬拼"丢细节。由于 $K$ 和 $d$ 都很小（$K=4\sim 8, d=32$），InnerTrans 的算力几乎可忽略。

**FLOPs / 参数公式**：
- Vanilla Transformer：$\text{FLOPs}=24Ld^2+4L^2d$，$\text{Params}=12d^2+13d$（Eq.3, 4）
- Token Merge 后 FLOPs 比：
  $$\frac{\text{FLOPs}_{\text{Merge}}}{\text{FLOPs}_{\text{vanilla}}}=\frac{6dK+KL}{6d+L}$$
  典型 $L=2048, d=32, K=4$：587M → 336M（**−42.8%**）
- 参数量反而放大：$\Theta_{\text{merge}}=12K^2d^2+13Kd$（Eq.5）
  → "少算 attention、多用参数"是刻意的 trade-off，让模型表达力不降反升。

### 2.5 模块三：LONGER 主体 —— Hybrid Causal Attention

#### 2.5.1 输入构造 (Eq.7)

- **全序列表示** $R=[G; H]\in\mathbb{R}^{(m+L)\times d}$：$m$ 个 global tokens + $L$ 个 sequence tokens
- 每个 item embedding 补两种位置信息：
  - **绝对时间差特征**（当前行为与 target item 的时间距离）concat 到 embedding；
  - **可学习的绝对位置 embedding** 加到每个 token 上。
- **Query 矩阵** $O=[G; H_S]$：$m$ 个 global tokens + 从 $H$ 里 **采样出的 $k$ 个 token**。论文试了多种采样策略，结论：**"Recent-k"（最近 k 个）最优**（Table 2 底部 4 行）。

  **重要的工业 insight**：对性能-算力的**边际效应极强**——**采样 40% 就能保留 95% 的性能提升，FLOPs 降 50%**。

#### 2.5.2 Cross-Causal Attention（第 1 层）

把 $O$ 当 query 去查 $R$（全序列），做**查询压缩**——把 $m+L$ 长度压成 $m+k$：
$$Q=OW_Q,\;K=RW_K,\;V=RW_V\tag{Eq.8}$$
$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^T}{\sqrt d}+M\right)V\tag{Eq.9}$$
因果 mask：
$$M_{i,j}=\begin{cases}0,&j\ge i\\-\infty,&\text{otherwise}\end{cases}\tag{Eq.10}$$

**两个作用**：
1. 保序列内部的时序因果关系；
2. **切断"序列 → 候选"的可见性**——候选的 global token 看不到自己之后的（不存在的）位置，也让 **用户序列侧** 的计算与 **候选侧** 的计算**相互独立**，是后面 KV Cache Serving 能成立的前提。

#### 2.5.3 Self-Causal Attention × N（后续层）

Cross 层已把序列压到 $m+k$，接下来 N 层 self-attention 在这个短序列上堆叠，每层跟 FFN 做高阶建模：
$$\text{SelfAttn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^T}{\sqrt d}+M\right)V\tag{Eq.11}$$

简写成：
$$\underbrace{\text{CrossAttn}(O,R)}_{\text{长序列压缩}}\;\rightarrow\;\underbrace{\text{SelfAttn}(\cdot)\times N}_{\text{高阶交互}}\tag{Eq.12}$$

这就是 "Perceiver / Q-Former / BLIP-2 系列的 query-compression 思路 + causal 语义" 在 RecSys 里的落地版本。

### 2.6 模块四：训练与部署工程优化

**(a) 全同步训练框架（Fig.2）**：

- **统一 Dense & Sparse 参数存储**：都放 GPU 同步更新，**取消 Parameter Server**，消除通信瓶颈。
- **分层 embedding 存储**：
  - 高频特征 → **HBM**（GPU 显存，最快）
  - 中频 → **CPU 主存 (MEM)**
  - 低频 → **SSD**
  - 按访问频率 offload，兼顾延迟/吞吐/容量。
- **Fountain** 模块做数据预处理，batch / streaming 都支持。

**(b) Mixed Precision + Recompute**：
- 用 BF16/FP16 混合精度，各层可分别指定精度；
- 用 `custom_gradient`（TF 没官方 recompute）做活性重算——前向丢掉中间激活、反向重新算。
- 综合收益：**训练吞吐 +18%，训练时间 −16%，显存 −18%，部分 dense 层显存 −28%**。

**(c) KV Cache Serving（Fig.3）**：

工业 ranking 场景——**同一请求内几百候选，用户序列完全不变**，唯一变的是候选 global token。

- **Step 1**（每请求一次）：把用户序列的 K/V 预计算并缓存；
- **Step 2**（每候选一次）：候选 global token 只和缓存 KV 做一次 attention。

Cross-Causal 层的"候选对序列不可见"设计正好让 Step 1/2 的拆分成立（序列的计算与候选无关，才能 cache）。

**线上效果**：吞吐退化从 **−40% 降到 −6.8%**，量级上变成"几乎无损"。

## 3. Experiment Analysis (实验结果解读)

### 3.1 实验设置

- **数据集**：Douyin Ads CVR 任务，2024.10.16 – 2025.02.23，**52 亿样本 / 130 天**，按时间切分（前 123 天 train，后 7 天 eval）；每条样本含 UID、性别、超长行为序列、候选广告。
- **行为类型**：page view / click / conversion。
- **集群**：48 × A100。
- **指标**：AUC / LogLoss（0.1% AUC 即工业显著）。
- **Baselines**：
  - **短序列**：DIN(Recent50)、TWIN（最近 50 条）
  - **长序列**：SumPooling、DIN、HSTU、Transformer

### 3.2 主实验（Table 1）

| Model | AUC ↑ | LogLoss ↓ | ΔAUC(%) | ΔLogLoss(%) |
|-------|-------|-----------|---------|-------------|
| Base | 0.83968 | 0.48758 | — | — |
| SumPooling | 0.84201 | 0.48538 | +0.28 | −0.45 |
| TWIN | 0.84472 | 0.48168 | +0.60 | −1.21 |
| DIN (Recent50) | 0.84698 | 0.47830 | +0.87 | −1.90 |
| DIN | 0.84982 | 0.47452 | +1.21 | −2.68 |
| HSTU | 0.84994 | 0.47490 | +1.22 | −2.60 |
| Transformer | 0.85111 | 0.47293 | +1.36 | −3.00 |
| **LONGER** | **0.85290** | **0.47103** | **+1.57** | **−3.39** |

→ 比最强 Transformer baseline 再 **+0.21% AUC**（工业 0.1% 即显著），且算力更低。

### 3.3 消融（Table 2）

| 配置 | FLOPs (×10⁹) | AUC | ΔAUC |
|------|-------------|-----|------|
| LONGER (w/o Merge, L=2000) | 3.73 | 0.85111 | +1.36% |
| **+TokenMerge4 (Concat, L=500)** | 2.13 | 0.85232 | +1.51% |
| **+TokenMerge8 (Concat, L=250)** | 3.03 | 0.85291 | +1.58% |
| +TokenMerge8 + InnerTrans | 3.52 | **0.85332** | **+1.63%** |
| **Query #=100（主配置）** | **1.91** | **0.85290** | **+1.57%** |
| Query #=250 | 3.52 | 0.85332 | +1.63% |
| Learnable 100 (queries) | 1.91 | 0.84946 | +1.17% |
| **Recent 100** | 1.91 | 0.85290 | +1.57% |
| Uniform 100 | 1.91 | 0.85183 | +1.45% |

**五条关键结论**：
1. **Token Merge 稳赚**：不但 FLOPs 降、效果还涨（Merge8 比 w/o Merge 再 +0.22% AUC）；
2. **InnerTrans 有用但收益有限**（再 +0.05% AUC，算力略涨）；
3. **Query 数的边际效应极强**：从 50 涨到 250，AUC 从 +1.51% 到 +1.63%，几乎打平；**Query=100 是工业甜点**，AUC 与 Query=250 持平，FLOPs 只用 54%；
4. **Query 选 Recent-k 最好**：Learnable random query（Perceiver/Q-Former 风格）反而最差（+1.17%），说明在 RecSys 里"最近行为 token 作初始化 query"比"让模型自学抽象 query"更有信息量；
5. **40% query 保留 95% 性能、减 50% FLOPs**——这是论文最适合部署的 punchline。

### 3.4 Scaling Analysis（Fig. 4/5）

拟合 $y=\alpha x^\beta+\gamma$（Eq.13）：
- **Sequence Length**（300 → 5k）：AUC / LogLoss 随 length 呈 power-law 提升；深模型从长序列中获益更多，但有递减效应。
- **参数量（Fig.5a）**：固定 2 层 / 序列 2000，扫 hidden dim → 强 power-law（$R^2=0.987$），当前参数范围内无饱和迹象。
- **FLOPs（Fig.5b）**：扫层数+序列长度 → 强 power-law（$R^2=0.967$）。

→ 三条轴都单调、都遵守 scaling law，是工业 RecSys 里少见的"完整 scaling 证据链"。

### 3.5 线上 A/B

**抖音广告（Table 3）**：

| 广告类型 | ADSS | ADVV |
|---------|------|------|
| Live Streaming | +1.063% | +1.168% |
| **Short Video** | **+2.097%** | **+2.151%** |
| Mall | +1.816% | +1.407% |

**抖音电商（Table 4）**：

| 内容类型 | Order/U | GMV/U |
|---------|---------|-------|
| **Live Streaming** | **+7.9222%** | **+6.5404%** |
| Short Video | +4.6125% | +5.2771% |

→ 电商直播的收益尤其突出（GMV/u 单测 +6.54%），说明**长行为序列对"用户深度意图"任务格外重要**。

### 3.6 系统级表现

- **Token Merge + query 采样**：总 FLOPs 减 ~50%，性能几乎无损；
- **混合精度 + Recompute**：吞吐 +18%、训练时间 −16%、显存 −18%；
- **KV Cache Serving**：serving 吞吐退化从 −40% 救到 **−6.8%**。

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **第一个把 RecSys 的长序列直接端到端做到 10k 的工业级工作**。跳出"两阶段 / 预训练 UE / memory-augment"的历史窠臼，给后续类 LLM 式 RecSys backbone（HSTU / OneTrans / MARM / LONGER 这一家族）定了一个可部署的基线。
2. **Token Merge + InnerTrans 的算力-参数互换非常精准**。"把序列压 K 倍、参数涨 K² 倍" 这种"用参数买算力"的 trade-off，把 RecSys 长序列从 quadratic 拽下来却不掉点，而且更涨点；InnerTrans 顺手解决组内信息丢失，设计非常干净。
3. **Hybrid Attention 把 Perceiver/Q-Former 的 cross-compression 成功迁移到 RecSys**。第 1 层 cross 做压缩、后面 self 做交互的组合，是长序列建模"先压后抽"的经典范式，在推荐里首次被大规模验证。
4. **工程优化与架构设计耦合得很紧**：Cross-Causal 的 "候选对序列不可见" 直接支撑了 KV Cache Serving；Recent-k query sampling 直接支撑了工业级 FLOPs budget；分层 HBM/MEM/SSD 存储是为推荐特征分布量身定制——**架构决策和系统决策是同一条链**。
5. **Scaling Law 证据链完整**：length / params / FLOPs 三个维度分别拟合 power-law（$R^2>0.96$），这是工业推荐里少见的系统 scaling 分析。
6. **多线部署佐证**：抖音广告 + 抖音电商双场景、Live/Short Video/Mall 多格式、数十个业务场景"fully deployed"，证据力很强。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **"Recent-k 胜过 Learnable query" 有点反直觉**，需要更多分析解释——是训练数据量不够、是 RecSys 任务对长程依赖需求有限、还是 causal mask 下 learnable query 收到的梯度稀疏？论文只给结果没给解释。
2. **Token Merge 的 $K$ 是 hard-coded** (4 或 8)，固定 stride 可能对不同用户活跃度不合适（重度用户该合得更多、轻度用户少合）。**自适应 merge schedule** 或者 **learnable compression**（类似 Mamba selective scan）是自然的下一步。
3. **Global Tokens 的数量与构成是启发式选择**：候选表示 + UID + CLS + 交叉特征，没有系统化的"哪些 anchor 更有用"的消融。
4. **InnerTrans 收益偏小**（+0.05% AUC），但带来了结构复杂度，可能在更大规模上才凸显价值，当前配置下是否值得开启需要权衡。
5. **KV Cache 的失效语义**论文未细说——用户序列一旦 append 新行为或 embedding 热更新，cache 如何增量更新？线上 hit rate 多少？对 p99 的最坏情况影响？
6. **序列长度 10k 只是"能做"，没有做到更长**（例如 100k 全生命周期）——未来方向包括：
   - 结合 **MARM / LMN 的 memory** 把更早期行为蒸馏成 memory slot，LONGER 主体处理近 10k，memory 补全远端；
   - 把 **OneTrans 的 mixed parameterization** 引入 LONGER（global tokens 本就是"非序列" NS-tokens，天然适配）；
   - 把 **Gated Attention** (arXiv:2505.06708) 的 SDPA-output sigmoid gate 塞到 self-causal layers 里，进一步稳训练、加稀疏、消 sink；
   - 扩展到 **多行为 / 多场景**（类似 OneTrans 的 timestamp-aware fusion），用统一 LONGER 栈同时吃广告 + 电商 + 内容。
