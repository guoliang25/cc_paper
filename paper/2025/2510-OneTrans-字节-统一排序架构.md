# OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender

> **Authors**: Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, Aixin Sun
> **Affiliation**: ByteDance (Singapore / Hangzhou) + Nanyang Technological University
> **Venue**: WWW 2026 (accepted), arXiv:2510.26104 (2025.10 / v3 2026.02)
> **Links**: [arXiv](https://arxiv.org/abs/2510.26104) · [PDF](https://arxiv.org/pdf/2510.26104)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 工业推荐系统的 Ranking 模块长期被拆成两条独立的路径——**序列建模**（DIN / SASRec / BST / LONGER，编码用户行为序列）与 **特征交互**（DCNv2 / Wukong / HiFormer / RankMixer，学 user/item/context 之间的高阶 cross），两者先后串行："encode-then-interaction"。这种设计有两个本质缺陷：(a) 两个模块只能单向传递压缩后的表示，上下文静态特征无法反向去塑造序列表示；(b) 执行图被切开，无法统一套用 LLM 成熟的工程优化（KV cache / FlashAttention / 混合精度 / 统一 scaling）。问题是：**能否只用一个 Transformer 同时吃下序列 token 和非序列特征 token，实现端到端的联合建模与联合 scaling？**
- **Method (方法)**: 提出 **OneTrans**，一个 "一条 Transformer 栈走到底" 的 ranking backbone。核心三件套：(1) **Unified Tokenizer**——把多行为序列 (S-tokens) 和静态/上下文/用户/物品特征 (NS-tokens) 统一拼成一条 token 序列 $X^{(0)}=[\text{S-tokens};\text{NS-tokens}]$；(2) **Mixed Parameterization OneTrans Block**——S-tokens 同质，共享一套 QKV/FFN 权重；NS-tokens 异质（price、userID、category 语义/量纲差异大），每个 NS-token 分配自己专属的 QKV/FFN；(3) **Causal-Attention + Pyramid Stack + Cross-Request KV Caching**——用因果 mask + 金字塔式逐层裁剪 S-token 查询数 + 请求间共享用户侧 KV cache，把复杂度从 $O(C)$（C 为候选数）降到 $O(1)$ 级别的用户侧计算。
- **Results (效果)**: 在 29.1B 条曝光、27.9M 用户、10.2M 物品的工业数据上训练：离线相对于已部署基线 DCNv2+DIN，**OneTrans-L 取得 CTR AUC +1.53%、CTR UAUC +2.79%、CVR AUC +1.14%、CVR UAUC +3.23%**，全面领先 RankMixer+Transformer 等强基线；线上 A/B 在 Feeds 场景 **click/u +7.74%、order/u +4.35%、GMV/u +5.68%、p99 延迟 −3.91%**，Mall 场景 click/u +5.14%、GMV/u +3.67%；冷启动 order/u **+13.59%**。且 OneTrans 的 scaling 曲线（ΔUAUC vs log-TFLOPs）**斜率显著高于 RankMixer**，展现更好的计算效率。

## 2. Detailed Methodology (详细方法解读)

### 2.1 从 "encode-then-interaction" 到 "one Transformer"

传统 DLRM ranking 架构（Fig.1(a)）：

```
Sequential feats → Sequence Modeling Block → Compressed Seq vector ↘
                                                                    Concat → Feature Interaction Block → Tower
Non-seq feats   ───────────────────────────────────────────────────↗
```

两个问题：
- **单向信息流**：序列被压成一个向量再和静态特征 concat，静态/候选上下文没法反过来调制序列表示（InterFormer 做过类似尝试，但仍是两个独立模块 + 额外 cross 桥接）。
- **执行图割裂**：序列模块是 attention-heavy、特征交互模块是 cross/MLP-heavy，统一 scaling 和 LLM 风格的 kernel/cache 优化都很难复用。

OneTrans（Fig.1(b)）的替代方案：
- 把所有东西都 tokenize 成一条长序列，交给**一条** Transformer 栈处理；
- 问题转化为：**怎样的 tokenization + 怎样的 Transformer block 设计，才能同时兼顾两类异质信息？**

### 2.2 统一分词器（Unified Tokenizer）

任务预测目标（Eq.1–2）：$\hat y_{u,i}=f^i(\mathcal{NS},\mathcal{S};\Theta)$，典型输出 CTR / CVR。

**(a) Non-Sequential Tokenization**，给出两种候选：

- **Group-wise Tokenizer**（对齐 RankMixer）：人工按语义分组 $\{g_1,\ldots,g_{L_{NS}}\}$，每组一个 MLP：
  $$\text{NS-tokens}=\big[\text{MLP}_1(\text{concat}(g_1)),\ldots,\text{MLP}_{L_{NS}}(\text{concat}(g_{L_{NS}}))\big]\tag{Eq.6}$$
- **Auto-Split Tokenizer**（论文默认，性能更好）：所有 NS 特征拼一起，一个大 MLP 投一次再 split：
  $$\text{NS-tokens}=\text{split}\big(\text{MLP}(\text{concat}(\mathcal{NS})),L_{NS}\big)\tag{Eq.7}$$

  **优点**：单次 dense 投影，kernel launch 开销低；**消融显示**不依赖人工分组反而更好，CTR AUC 比 Group-wise 高 +0.10%。

**(b) Sequential Tokenization**（多行为序列 $\mathcal{S}=\{S_1,\ldots,S_n\}$，每条 $S_i=(e_{i1},\ldots,e_{iL_i})$）：

- 每种行为序列用自己的投影 MLP 对齐到统一维度 $d$：
  $$\tilde S_i=\big[\text{MLP}_i(e_{i1}),\ldots,\text{MLP}_i(e_{iL_i})\big]\in\mathbb{R}^{L_i\times d}\tag{Eq.9}$$

- 多序列合并成一条：
  - **Timestamp-aware**（论文默认）：按时间戳交错所有事件，类型用一个 indicator 区分；
  - **Timestamp-agnostic**：按用户意图强度排序（purchase → add-to-cart → click），序列之间插入可学习的 `[SEP]` token。

  **消融**：时间戳可用时，timestamp-aware **始终胜出**（CTR AUC +0.09%）；timestamp-agnostic 场景下去掉 `[SEP]` 会再掉 −0.13% / −0.32%。

最终输入序列：
$$X^{(0)}=\big[\text{S-tokens};\text{NS-tokens}\big]\in\mathbb{R}^{(L_S+L_{NS})\times d}\tag{Eq.3}$$

### 2.3 OneTrans Block：Mixed Parameterization（核心创新）

每个 block 是 **pre-norm 的因果 Transformer**，但 Q/K/V 和 FFN 权重都做"分段共享"：

$$Z^{(n)}=\text{MixedMHA}\big(\text{Norm}(X^{(n-1)})\big)+X^{(n-1)}\tag{Eq.4}$$
$$X^{(n)}=\text{MixedFFN}\big(\text{Norm}(Z^{(n)})\big)+Z^{(n)}\tag{Eq.5}$$

**(a) Mixed Causal Attention（Eq.11–12）**

对第 $i$ 个 token $x_i\in\mathbb{R}^d$：
$$(q_i,k_i,v_i)=(W^Q_i x_i,W^K_i x_i,W^V_i x_i)$$

$$W^\Psi_i=\begin{cases}W^\Psi_S,& i\le L_S\quad(\text{S-tokens 共享}) \\ W^\Psi_{NS,i},& i>L_S\quad(\text{每个 NS-token 一套})\end{cases},\quad \Psi\in\{Q,K,V\}$$

- 所有 S-tokens（同质的序列事件）共享一套投影，保证参数效率；
- 每个 NS-token（异质的 price、userID、category 等）拥有 **token-specific** 投影，保住语义差异。

**(b) Causal mask 的作用 (非常关键的三条)**：

1. **S-side**：S-token 只能看更早的 S-token。对 timestamp-aware 序列 → 每个事件以历史为条件；对 timestamp-agnostic（按意图强度排序）→ 高意图行为可以下行过滤低意图行为，信息天然汇聚到"更 recent / 更接近 NS 段"的位置；
2. **NS-side**：每个 NS-token 能看到**整个 S 历史** + 它自己之前的 NS-tokens。这天然等价于 "target-attention 式序列聚合 + NS 之间的多层交互"，顺带让 CTR/CVR 头部可以对齐到最后几个 NS-token；
3. **Pyramid 兼容**：因果 mask 让信息沿 token 轴单向向后聚合，天然支持"后面用几条查询就够了"的裁剪策略。

**(c) Mixed FFN（Eq.13）**

完全相同的 mixed 策略：
$$\text{MixedFFN}(x_i)=W^2_i\phi(W^1_i x_i)$$
权重 $W^1_i,W^2_i$ 遵循 Eq.12 的分段共享。

**(d) Pre-norm + RMSNorm**

S-token 和 NS-token 的数值尺度差异很大（NS 的 price、CTR 都做了 bucketize，但内在分布仍异质），**post-norm 会导致 attention collapse / 训练不稳定**；论文选用 **RMSNorm as pre-norm** 对齐不同来源 token 的尺度。

**(e) 参数对照**：对比"全共享"baseline，Mixed Parameterization 多出的只是 $L_{NS}$ 份 token-specific QKV+FFN——这是参数增加的主力来源，也是 OneTrans-L 能长到 330M 的关键位置；**消融显示一旦去掉 NS 的 token-specific（全共享），CTR AUC −0.15%、UAUC −0.29%、参数骤降到 24M → 证明这个设计是"性能/参数" 的主要贡献者**。

### 2.4 Pyramid Stack：金字塔式逐层裁剪 S-tokens

观察：因果 mask 本身会把信息 **往序列尾部集中**（后面的 token 能看前面所有）。那么深层其实不需要那么多 S-token 发出 query——只保留尾部最近的一段即可。

形式化：设当前层输入 $X=\{x_i\}_{i=1}^L$，定义尾部查询集合 $\mathcal{Q}=\{L-L'+1,\ldots,L\}$：
$$q_i=W^Q_i x_i,\quad i\in\mathcal{Q}\tag{Eq.14}$$
- **Keys/Values 仍然在全序列 $\{1,\ldots,L\}$ 上计算**（因为后续层仍要往前看）；
- 只保留 $i\in\mathcal{Q}$ 的 attention 输出，序列长度缩到 $L'$；
- 堆叠后就形成金字塔：token 长度逐层递减。

**实际 schedule**（论文默认启发式）：
- OneTrans-S：从 1190 线性缩到 12（6 层），每层对齐到 32 的倍数；
- OneTrans-L：从 1500 线性缩到 16（8 层）。
- 顶层 S-token 数 = NS-token 数。

**两个好处**：
1. **Progressive distillation**：长行为历史被逐层压缩、蒸馏进 NS-tokens；
2. **算力缩减**：attention 从 $O(L^2 d)$ 变成 $O(L L' d)$，FFN 从 $O(L)$ 变成 $O(L')$。

**消融**：拿掉 pyramid、每层都保全长序列，CTR AUC −0.05%、UAUC −0.42%（CVR 掉更多），**且 TFLOPs 从 2.64 飙到 8.08（3× 开销）**；反向证据：同等 TFLOPs 预算下，pyramid 能多吃 **~1.75× 的长度**。

### 2.5 训练 / 推理系统优化

**(a) Cross-Request KV Caching（Sec 3.5.1）**——工业 ranking 的特有捷径

同一个请求内，几百个候选的 **S-tokens 完全相同**，只有 NS-tokens 在变。利用这一点拆成两阶段：
- **Stage I (S-side, 每请求一次)**：对 S-tokens 做一次 causal attention，**缓存 K/V + attention 输出**；
- **Stage II (NS-side, 每候选一次)**：计算该候选的 NS-tokens，做 cross-attention 到缓存的 S-side KV，再过 NS 专属 FFN。
- 把复杂度从 $O(C)$（$C$ 个候选都重算一遍 S 侧）降到 $O(1)$。
- 此外，**用户行为序列是 append-only 的**，跨请求复用上次的 cache + 只增量算新行为 → 每请求实际 S 侧开销从 $O(L)$ 降到 $O(\Delta L)$，$\Delta L$ 是新增行为数。

**(b) LLM 风格优化组合拳**：FlashAttention-2（I/O 级 attention 优化）+ BF16/FP16 混合精度 + activation recomputation。

**系统收益**（Table 4，相对无优化 OneTrans-S）：

| 叠加优化 | 训练时间 | 训练显存 | 推理 p99 延迟 | 推理显存 |
|--------|---------|---------|--------------|---------|
| + Pyramid Stack | −28.7% | −42.6% | −8.4% | −6.9% |
| + Cross-Request KV Caching | −30.2% | −58.4% | **−29.6%** | **−52.9%** |
| + FlashAttention | −50.1% | −58.9% | −12.3% | −11.6% |
| + 混合精度 + Recompute | −32.9% | −49.0% | **−69.1%** | −30.0% |

→ KV Caching 对推理**延迟/显存最猛**（因为它直接消掉了 S 侧在候选轴上的冗余）；混合精度对推理 p99 也是数量级收益。

## 3. Experiment Analysis (实验结果解读)

### 3.1 实验设置

- **数据集**：ByteDance 内部工业数据，29.1B 曝光、27.9M 用户、10.2M 物品；按时序切分，特征按曝光时刻 snapshot 防止穿越。
- **任务**：CTR、CVR（post-click conversion rate），指标 AUC / UAUC（user-level impression-weighted AUC）。
- **Next-batch evaluation**：时间序推进，每个 batch 先预测（eval mode）再训练，按天算 AUC 再 macro-平均。
- **优化器**：双优化器——sparse embedding 用 Adagrad($\beta_1$=0.1,$\beta_2$=1.0)，dense 用 RMSProp(lr=0.005)；grad-clip 90(dense)/120(sparse)；16×H100。
- **配置**：
  - **OneTrans-S**：6 层，$d=256$，$H=4$ 头，91M 参数
  - **OneTrans-L**（默认）：8 层，$d=384$，330M 参数，TFLOPs 8.62

### 3.2 主实验（Table 2）—— 与 12 个强基线对比

锚定产线基线 **DCNv2+DIN**（10M 参数，0.06 TFLOPs），所有 delta 都以它为 0：

| Type | Model | CTR AUC | CTR UAUC | CVR AUC | CVR UAUC | Params | TFLOPs |
|------|-------|---------|----------|---------|----------|--------|--------|
| Base | DCNv2+DIN* | 0.79623 | 0.71927 | 0.90361 | 0.71955 | 10M | 0.06 |
| Feature-Inter | Wukong+DIN | +0.08% | +0.11% | +0.14% | +0.11% | 28M | 0.54 |
| Feature-Inter | HiFormer+DIN | +0.11% | +0.18% | +0.23% | −0.20% | 108M | 1.35 |
| Feature-Inter | RankMixer+DIN* | +0.27% | +0.36% | +0.43% | +0.19% | 107M | 1.31 |
| Feature-Inter | RankMixer+StackDIN | +0.40% | +0.37% | +0.63% | −1.28% | 108M | 1.43 |
| Seq-Model | RankMixer+LONGER | +0.49% | +0.59% | +0.47% | +0.44% | 109M | 1.87 |
| Seq-Model | RankMixer+Transformer* | +0.57% | +0.90% | +0.52% | +0.75% | 109M | 2.51 |
| **Unified** | **OneTrans-S*** | **+1.13%** | **+1.77%** | **+0.90%** | **+1.66%** | 91M | 2.64 |
| **Unified** | **OneTrans-L*** | **+1.53%** | **+2.79%** | **+1.14%** | **+3.23%** | 330M | 8.62 |

★ 标"*"的是该团队产线曾经/正在部署的模型，时间顺序：DCNv2+DIN → RankMixer+DIN → RankMixer+Transformer → OneTransS → OneTransL。

**关键观察**：
- OneTrans-S 以 91M 参数、2.64 TFLOPs **全面胜过 RankMixer+Transformer (109M, 2.51 TFLOPs)**，CTR UAUC +1.77% vs +0.90%，几乎翻倍。
- OneTrans-L 以 330M 参数拿下 **CTR UAUC +2.79%**，是最强基线的 3× 增益；CVR UAUC +3.23% 进一步说明 "深度转化" 任务受益于联合建模。
- 产线改进的幅度通常 >+0.1% 就算显著、>+0.3% 上线大概率有 p<0.05 的正收益——OneTrans-L 远超此门槛。

### 3.3 设计消融（Table 3，以 OneTrans-S 为 reference）

| Type | Variant | ΔCTR AUC | ΔCTR UAUC | 结论 |
|------|---------|----------|-----------|------|
| Input | Group-wise Tokenizer | −0.10% | −0.30% | Auto-Split 胜 |
| Input | Timestamp-agnostic Fusion | −0.09% | −0.22% | 时间戳比意图排序好 |
| Input | Timestamp-agnostic w/o SEP | −0.13% | −0.32% | `[SEP]` 必要 |
| Block | Shared parameters（NS 也共享） | −0.15% | −0.29% | **Mixed 参数化是关键** |
| Block | Full Attention | +0.00% | +0.01% | 持平，但会**禁用 KV cache**，必须用 causal |
| Block | w/o Pyramid Stack | −0.05% | +0.06% | 性能几乎一样，**但 TFLOPs 从 2.64 → 8.08** |

→ 六条消融结论支撑六个设计选择，非常清晰：**Auto-Split tokenizer / 时间戳优先 / SEP token / NS token-specific 参数 / Causal attention / Pyramid**。

### 3.4 Scaling Law（Fig. 3）

作者沿三条轴扫 OneTrans：**length（序列长度 T）、depth（层数 L）、width（$d_{\text{model}}$ D）**：

- **Length 收益最大**（更多行为证据比更深/更宽更有效）；
- **Depth 比 Width 好**：深度能叠出更高阶交互，但并行性差；
- **OneTrans vs RankMixer（scaling 到 ~1B）**：ΔUAUC-vs-log(TFLOPs) 两者都是 log-linear，但 **OneTrans 斜率更陡**——论文解释为 RankMixer 的 MoE-scaling 主要在扩 FFN 宽度，且没有统一 backbone 串起序列。

### 3.5 系统侧对比（Table 5）：OneTrans-L vs DCNv2+DIN

| 指标 | DCNv2+DIN | OneTrans-L | 含义 |
|------|-----------|------------|------|
| TFLOPs | 0.06 | 8.62 | 算力增 143× |
| 参数 | 10M | 330M | 33× |
| MFU | 13.4% | **30.8%** | 硬件利用率翻倍多 |
| 推理 p99 延迟 | 13.6ms | **13.2ms**（更低） | 算力涨 140× 延迟反而降 |
| 训练显存 | 20GB | 32GB | 可控 |
| 推理显存 | 1.8GB | **0.8GB**（更低！） | KV cache 复用的红利 |

→ "参数 33×、FLOPs 143×、**延迟不增反降**" 这组数据是这篇论文最亮眼的工程成就，背后全靠 KV cache + pyramid + FlashAttention + 混合精度的组合。

### 3.6 线上 A/B（Table 6）

OneTrans-L（treatment）对战 RankMixer+Transformer（control，约 100M 参数）：

| 场景 | click/u | order/u | gmv/u | 延迟 p99 |
|------|---------|---------|-------|----------|
| **Feeds** | **+7.737%** ** | +4.351% * | **+5.685%** * | −3.91% |
| **Mall** | **+5.143%** ** | +2.577% ** | +3.670% * | −3.26% |

其中 `*`: $p<0.05$，`**`: $p<0.01$。额外效果：
- 用户 **Active Days +0.75%**（留存层面的改动）；
- **冷启动商品 order/u +13.59%**（联合建模+大模型对长尾/新品的泛化极强）。

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **把"序列建模 + 特征交互"的历史性隔阂用一条 Transformer 直接抹平**。这不是简单的 "再加一个 transformer block"，而是架构层面把两个赛道合流，信号双向流通、scaling 统一规划、工程优化一次复用。RecSys 里少见的"干净"架构变更。
2. **Mixed Parameterization 是"通用 Transformer 移植到 RecSys"的关键桥梁**。同质 token 共享/异质 token 独享这条原则，正好精确对应"用户行为序列 vs 异构静态特征"的本质差异；既保住 NS 的语义区分（对比 RankMixer 的 per-token FFN 思路），又不像 HiFormer 那样全异质化导致参数爆炸。
3. **Cross-Request KV Caching 是工业 ranking 的独门红利**。LLM 的 KV cache 只能在单序列内复用，而 ranking 场景 "同请求内几百候选共享用户侧" 这个结构被作者抓得非常精准，S-side $O(C)\to O(1)$、跨请求增量 $O(L)\to O(\Delta L)$，两级 cache 直接让 330M 模型延迟比 10M baseline 还低——**这可能是这篇论文最大的工程贡献**。
4. **Pyramid Stack 是 causal + recency 结构的自然结果**。不像 NSA/BigBird 那样需要复杂的稀疏模式，这里直接"深层只保留尾部 query"，简单优雅、与 KV cache 完美兼容、还顺带解决了长序列的算力瓶颈。
5. **完整的证据链**：29B 曝光的工业数据 + 12 个强基线的参数-FLOPs 匹配对比 + 6 项核心消融 + 三轴 scaling + 两场景线上 A/B + 系统级对比（MFU、延迟、显存），很难找到漏洞。
6. **直接兑现 LLM 红利**：FlashAttention-2 / 混合精度 / recompute 等 LLM 工程栈在 OneTrans 上即插即用，这是 DLRM 架构无论如何也吃不到的。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **Pyramid schedule 是启发式的**（线性从 $L_S^{(0)}$ 缩到 $L_{NS}$，按 32 对齐）。更细致的 per-layer query budget 可能有更大潜力，或者学习式 pruning（类似 Native Sparse Attention 的 selection）可以拿到更好的 compute-quality Pareto。
2. **Causal 是为 KV cache 服务的选择**，full-attention 在离线精度上基本持平甚至略高——说明如果未来有更优秀的 full-attention 友好 cache 机制，OneTrans 还有上升空间。
3. **Mixed Parameterization 的参数分布极偏**：随着 $L_{NS}$ 增大，NS-side 的 token-specific QKV/FFN 会成为参数主力；这对小模型友好，但极端规模下会不会冲掉 S-side 的 scaling 收益，还需要更宽规模（>1B）的实验。
4. **序列融合规则**（timestamp-aware 交错）在多行为类型场景下仍是固定的，未来可以探索 learnable fusion 或者把 "行为类型" 本身作为 token 属性喂进去。
5. **KV Caching 依赖 ID/embedding 稳定**：用户行为中途变更、embedding 热更新都会使缓存失效；线上全量部署下的 cache hit 率、miss 惩罚没有详细披露。
6. **Scaling 到 1B 以上受限于 p99 延迟约束**（论文明确说"未来工作"）。与 sparse MoE、LoRA-style expert、量化等方向结合可能是下一步自然延伸。
7. **可能的改进方向**：
   - 把 OneTrans 的 Mixed Parameterization 与 **MixFormer / RankMixer 的 SMoE** 结合：S-tokens 共享 MoE expert pool，NS-tokens token-specific expert；
   - 把 **Gated Attention**（arXiv:2505.06708）的 SDPA-output sigmoid gate 插进 OneTrans Block，进一步稳住大规模训练；
   - 在 **生成式 Recommender (HSTU 等)** 的框架下复用 OneTrans 的 mixed parameterization，把 "candidate 作为一个 NS-token" 进一步统一；
   - 把 **长期行为** (SIM、ETA) 的聚合 token 作为 NS-token 并入 OneTrans 序列，让 cross-attention 自动完成目标聚合。
