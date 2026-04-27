# 字节四大工业 Ranking 架构对比：RankMixer / LONGER / OneTrans / MixFormer

> 四篇论文全部出自 ByteDance，时间跨度 2025.05 – 2026.02，构成了字节在工业级推荐 Ranking 架构上的一条完整演化线。本文把它们放在同一张坐标系里对比，厘清各自解决的核心问题、架构哲学、与彼此的继承/替代关系。

## 0. 四篇论文一图式定位

```
时间线 →
 2025.05                2025.05              2025.07             2025.10              2026.02
   │                       │                    │                   │                    │
 LONGER                    │               RankMixer             OneTrans            MixFormer
(RecSys'25)                │               (2507.15551)         (WWW'26)             (2602.14110)
长序列建模的终极版          │               特征交互的 scaling     统一 ranking          "统一架构"的精细版
                          │                                       backbone
                          ▼
                     GatedAttention (LLM 注意力机制改进，同月但非 RecSys 专属)

                       │        │            │           │
                       ▼        ▼            ▼           ▼
                    "我只做     "我只做      "我把两者     "我把两者合并，
                     序列"     特征交互"    合到一条栈里"  并且做得更精细"
```

一句话各自定位：

| | 一句话 | 革新对象 |
|---|---|---|
| **RankMixer** | 用 hardware-conscious 的 multi-head token mixing + Per-token FFN + Sparse-MoE 把**特征交互模块**扩到 1B 但 MFU 拉到 45% | 特征交互模块（DLRM-MLP / Wukong / HiFormer） |
| **LONGER** | 用 Token Merge + Hybrid Cross/Self Attention + KV Cache 把**序列建模模块**端到端吃到 10k | 序列建模模块（SIM / TWIN / UE / MIMN） |
| **OneTrans** | 用 Mixed Parameterization + Pyramid Stack + Cross-Request KV Cache 把 **"序列建模 + 特征交互"两段式整个合并**成一条 Transformer 栈 | DLRM 的 encode-then-interaction 范式 |
| **MixFormer** | 在 OneTrans 的"统一 backbone"思路上，用 Query Mixer + Cross-Attention + Output Fusion + User-Item Decoupling 做成**更精细、更省 FLOPs 的统一架构** | OneTrans 自己 + 其它两段式方案 |

---

## 1. 它们各自在做哪一段

传统 DLRM ranking 的结构：

```
┌─ 非序列特征 ────────────► [特征交互模块 FI] ──┐
│                          (DCNv2/Wukong/                │
│                           HiFormer/RankMixer)          ├─► Tower
└─ 序列特征 ──► [序列模块 SM] ─► 压缩向量 ────┘
              (DIN/SIM/TWIN/LONGER)
```

四篇分别攻击的位置：

| 论文 | 改进/替代位置 | 是否替代整体 backbone |
|------|--------------|---------------------|
| **RankMixer** | **FI 模块** 升级版，仍和 SM 串联 | ❌ 否 |
| **LONGER** | **SM 模块** 升级版，仍和 FI 串联 | ❌ 否 |
| **OneTrans** | **整个 DLRM** 替换成单栈 Transformer | ✅ 是 |
| **MixFormer** | **整个 DLRM** 替换成单栈 Transformer | ✅ 是 |

**关键观察**：
- RankMixer 和 LONGER 是"两段式里各自的终极版"——它们本身不冲突，可以拼起来用（OneTrans 论文的 baseline 里就有 `RankMixer+LONGER`、`RankMixer+Transformer` 等组合）。
- OneTrans 和 MixFormer 是"统一架构"的两代方案——MixFormer 论文直接把 OneTrans 作为 Parallel 范式的代表 baseline 进行比较，并声称 MixFormer 在"效果 × FLOPs"两个维度上都胜过它。

---

## 2. 架构哲学的两条对立轴

把四篇按"统一 vs 分离"和"序列 vs 特征"两条轴放到一张图上：

```
                       Feature-Interaction 为主
                               ▲
                               │
                         RankMixer (1B FI 模型)
                               │
                               │
Stacked/Parallel ─────┼────────┼──────── Unified / Single Stack
(保留两段式)           │                        (合并两段式)
                        │
                               │
               LONGER (10k 序列)     OneTrans (mixed param)
                                     MixFormer (query mixer + cross-attn)
                               │
                               ▼
                          Sequence-Modeling 为主
```

可以看到：
- **左上（RankMixer）**：只扩 FI，不管序列
- **左下（LONGER）**：只扩 SM，不管 FI
- **右下（OneTrans / MixFormer）**：合并，两者同时扩

这也是字节研究演化的时间顺序：**先分别把两段做到极致 → 再尝试把两段合一**。

---

## 3. 四大维度全景对比表

### 3.1 处理对象

| 维度 | RankMixer | LONGER | OneTrans | MixFormer |
|------|-----------|--------|----------|-----------|
| 首要输入 | 非序列特征（分组 tokenize） | 长行为序列（≤ 10k） | S-tokens + NS-tokens 拼成一条 | 非序列特征（N 个 head） + 行为序列 |
| 是否处理长序列 | ❌ 外挂 DIN 等做 | ✅ 最长 10k | ✅ 最长 ~1.5k（pyramid 裁剪） | ✅ 512~10k 都测 |
| 是否做 FI | ✅ 核心卖点 | ❌ 外挂 RankMixer 等做 | ✅ NS-tokens 间 self-attn 自动做 | ✅ Query Mixer + Output Fusion |
| 整体定位 | **FI 模块升级** | **SM 模块升级** | **整条 backbone 替换** | **整条 backbone 替换（更精细）** |

### 3.2 Token 化与参数化

| 维度 | RankMixer | LONGER | OneTrans | MixFormer |
|------|-----------|--------|----------|-----------|
| Token 来源 | 特征按语义分组，每组 MLP 投成一个 token ($T\times D$) | 行为序列每个 item 一个 token + 少量 global tokens | S-tokens（行为事件）+ NS-tokens（每个非序列 group 一个） | 非序列特征切成 $N$ 个 head；序列 token 按 head 切 KV |
| Token 粒度 | 十几到几十个特征 token | 2000–10000 行为 token | $L_S$（几百–1500）+ $L_{NS}$（几十–上百） | $N$（16）head + 序列 |
| 参数共享 | **Per-token FFN**（每 token 独享 FFN；MoE expert 共享） | 所有 token 共享一套 QKV/FFN（标准 Transformer） | **Mixed Parameterization**：S 共享 / 每个 NS 独享 | **Per-head** 的 SwiGLU FFN（在 Query Mixer 和 Output Fusion 两处） |
| 参数扩展手段 | Sparse MoE + DTSI | Token Merge 的 $K^2$ 倍放大 | NS token-specific QKV+FFN 本身的堆叠 | Per-head 独立 FFN + width $D$ |

→ **关键差异**：RankMixer 和 MixFormer 相信"**per-token / per-head** 独立 FFN"，LONGER 相信"**全共享**"，OneTrans 相信"**同质共享、异质独立**"的折中。

### 3.3 特征交互（跨 token/跨 head）机制

| 维度 | RankMixer | LONGER | OneTrans | MixFormer |
|------|-----------|--------|----------|-----------|
| 核心交互机制 | **Multi-head Token Mixing**（无参数 reshape+转置，替代 self-attn） | Cross + Self Causal Attention | 统一 Causal Self-Attention（Mixed QKV） | **HeadMixing**（无参数 reshape，替代 self-attn）+ 显式 Cross-Attention |
| 复杂度 | $O(TD)$ 无参数 | Cross $O(L\cdot k)$ + Self $O((m+k)^2)$ | $O((L_S+L_{NS})^2)$ 但配 pyramid 降阶 | HeadMixing $O(ND)$ + Cross-Attn $O(NT)$ |
| Self-Attention 位置 | ❌ 被 token mixing 替代 | 后续 N 层都是 self | 全程 self | ❌ 被 HeadMixing 替代 |
| Cross-Attention 位置 | ❌ | ✅ 第 1 层（query 压缩） | ❌ 统一成 self（NS 看 S 是 self-attn 的子图） | ✅ 每层都有（特征 query 读序列 KV） |

**逻辑**：
- RankMixer 和 MixFormer 是"**无参数混合 + 独立 FFN**"路线（来自 MLP-Mixer / FNet 思想），认为 attention 在 FI 场景里性价比不够；
- LONGER 是"**Perceiver / Q-Former 风格的 cross-compression**"路线，用显式 cross-attn 把长序列压短；
- OneTrans 是"**纯 Transformer 统一栈**"路线，所有 token 平等进一个 self-attn。

### 3.4 长度 / 算力瓶颈应对

| 维度 | RankMixer | LONGER | OneTrans | MixFormer |
|------|-----------|--------|----------|-----------|
| 序列长度瓶颈 | 不处理（让外挂的 DIN 处理） | **Token Merge (K=4~8) + Recent-k query (100)** | **Pyramid Stack** 每层裁剪 query 数 | **User-Item Decoupling** + 逐层独立 FFN |
| 参数扩展手段 | **Sparse-MoE + DTSI** 推理时稀疏 | 共享权重 + hidden dim 扩大 | Mixed Parameterization 的 NS 侧膨胀 | $D$ + $N$ + 每层独立 FFN |
| FLOPs 管理 | MFU 45% 是头号卖点 | 显式 FLOPs 公式与 −50% 的 pareto | FLOPs 扩 143× 延迟反降的 system-level 成就 | MixFormer-medium 2242 GFLOPs vs STCA→RankMixer 6736 GFLOPs (**−67%**) |

### 3.5 特殊 token 设计

| | 有无特殊 token | 作用 |
|---|---|---|
| **RankMixer** | ❌ 没有，所有 token 都是特征 token | — |
| **LONGER** | ✅ **Global Tokens**（候选表示/CLS/UID/交叉特征） | anchor，稳 attention、消 sink |
| **OneTrans** | ✅ **[SEP]** token + 把 NS-tokens 放序列尾 | 分隔多类行为；NS-tokens 当任务主体 |
| **MixFormer** | ❌ 没有 token 级锚点，但有 **N 个特征 head** 充当 query 集合 | N 个 head 对序列 KV 做 cross-attn |

### 3.6 Serving 优化

| 维度 | RankMixer | LONGER | OneTrans | MixFormer |
|------|-----------|--------|----------|-----------|
| KV Cache | 不涉及 | ✅ 单请求内跨候选 | ✅ 单请求内 + **跨请求增量** $O(\Delta L)$ | 不直接用 KV cache |
| User-Item 计算解耦 | 不涉及 | 靠 cross-attn 的 "候选对序列不可见" | 靠 "S 前 NS 后 + causal mask" 天然解耦 | ✅ **User-Item Decoupling**（方向性 mask，request-level batching） |
| FlashAttention | ❌ | 未强调 | ✅ FA-2 | 未强调 |
| 混合精度 + Recompute | ✅ | ✅ BF16/FP16 + 自研 custom_gradient | ✅ | ✅ Pre-RMSNorm |
| 分层 embedding 存储 | 未强调 | ✅ HBM/MEM/SSD 三级 | 未提 | 未提 |
| 训练框架 | 常规 | ✅ 全同步 GPU（去 PS） | 常规 DP all-reduce | 常规 |

→ **LONGER 在底层框架层最下沉**（自研栈 + HBM/MEM/SSD 分层）；**OneTrans 在"算法-KV"耦合最精致**；**MixFormer 在"算法-batching"耦合最精致**。

### 3.7 Scaling 的扫法

| | 扫哪些轴 | 斜率对比 |
|---|---|---|
| **RankMixer** | Params / FLOPs（100M → 1B） | 比 Wukong、HiFormer、DHEN 斜率更陡 |
| **LONGER** | Length / Params / FLOPs（各独立 power-law，$R^2>0.96$） | 未跟别的架构对比 |
| **OneTrans** | Length / Depth / Width | **斜率显著 > RankMixer** |
| **MixFormer** | Dense scaling (FLOPs) + Sequence length（512 → 10k） | 斜率与 SOTA 序列模型持平；截距更高 |

### 3.8 主要实验战场

| | 任务 | 数据规模 | 集群 | 最强 baseline 被打败幅度 |
|---|---|---|---|---|
| **RankMixer** | Finish/Skip AUC | 抖音 Feed | — | +0.16% AUC vs HiFormer (同参数) |
| **LONGER** | CVR | 5.2B 样本 / 130 天 | 48 A100 | +0.21% AUC vs Transformer |
| **OneTrans** | CTR+CVR | 29.1B 曝光 | 16 H100 | +0.87% CTR UAUC vs RankMixer+Transformer |
| **MixFormer** | Finish/Skip AUC | 抖音 | — | +0.16% Finish AUC vs STCA→RankMixer，同时 **FLOPs −67%** |

### 3.9 线上 A/B 收益

| | 场景 | 核心指标提升 |
|---|---|---|
| **RankMixer** | 抖音 Feed | 活跃天数 +0.29%、时长 +1.08%（低活用户 +1.74% / +3.64%）；广告 ADVV **+3.90%** |
| **LONGER** | 抖音广告 + 电商 | 广告 ADVV 最高 **+2.15%**；**电商直播 GMV/u +6.54%、Order/u +7.92%** |
| **OneTrans** | Feeds + Mall | Feeds GMV/u **+5.68%**、click/u +7.74%、延迟 −3.91%；冷启动 order/u +13.59% |
| **MixFormer** | 抖音 + 抖音极速版 | 时长 +0.28%、评论 **+0.70%**（极速版 +1.91%）；推理 **>30% 加速** |

→ **上线收益量级**：OneTrans > LONGER > RankMixer > MixFormer（**但注意 MixFormer 的 baseline 已经是 RankMixer**，增量是叠加的）。

---

## 4. 四两对两两的"真正分歧点"

### 4.1 RankMixer vs LONGER：一横一纵，互补不冲突

- **RankMixer**（横）：把**非序列特征交互模块**做到 1B 参数、MFU 45%；
- **LONGER**（纵）：把**用户行为序列模块**做到 10k 长度、FLOPs −50%；
- 两者都**保留了两段式架构**，事实上在字节内部是可以直接组合成 **RankMixer + LONGER** 一起上线的（OneTrans 论文正是把这个组合作为最强两段式 baseline）。
- **如果你问"该先上哪个"**：看系统短板。FI 端不够重就先上 RankMixer；序列端还在用 SIM/TWIN 就先上 LONGER。

### 4.2 LONGER vs OneTrans：序列模块终极版 vs 整个 backbone 重构

- LONGER 接受"序列模块独立存在"的假设，把这个模块做到 10k；
- OneTrans 否定这个假设，把序列 token 和非序列 token 扔进同一条 self-attn 栈。
- **LONGER 的 Global Tokens 像是 OneTrans 的 NS-tokens 的雏形**，但 LONGER 的 Global Tokens 是"锚点"、共享参数；OneTrans 的 NS-tokens 是"任务主体"、token-specific 参数。
- OneTrans 论文把 **RankMixer+LONGER** 列为 baseline，证明"统一架构的斜率更陡"。
- **如果你问"下一步往哪走"**：业界大方向是统一架构（OneTrans / MixFormer 路线），LONGER 的很多思想（Recent-k query、Token Merge、KV Cache）会被吸收进统一架构。

### 4.3 OneTrans vs MixFormer：同一"统一架构"哲学下的两代实现

**两者共识**：
- 一条 Transformer 搞定序列 + 特征交互；
- 用 causal/directional mask 实现 "user-side 跨候选复用"；
- Scaling law 完整、工程优化充分。

**两者分歧**（关键！）：

| | OneTrans | MixFormer |
|---|---|---|
| Token 数量 | 大（$L_S+L_{NS}$，通常几百到 1500） | 小（$N=16$ 个 head + 序列 T） |
| 跨 token 交互 | **Self-Attention**（标准 softmax attn） | **无参数 HeadMixing**（reshape + transpose） |
| 长序列处理 | **Pyramid Stack**（层间裁剪 query 数） | **显式 Cross-Attention**（逐层用特征 head 去查序列 KV） |
| 特征 head 数量 | = NS-tokens 数量，通常几十个 | 固定 $N=16$ |
| Per-layer 独立 FFN | ❌ 层间共享 | ✅ Cross-Attn 的序列 FFN 逐层独立 |
| User-Item 解耦方式 | causal mask（S 前 NS 后天然解耦） | **方向性 mask** 控制 head 间可见性 |
| FLOPs | OneTrans-L 8.62T | UI-MixFormer-medium 2.24T |
| 性能 | CTR UAUC +2.79% vs DCNv2+DIN | Finish AUC +1.28% vs STCA→RankMixer baseline |

**精髓差别**：
> OneTrans 走 "**纯 Transformer**" 路线（所有 token 经过 self-attn），靠 pyramid 降阶；
> MixFormer 走 "**MLP-Mixer 跨 head + Cross-Attention 读序列**" 的混合路线，用无参数 mixing 省 FLOPs、用 cross-attn 读长序列。

**MixFormer 论文直接把 OneTrans 作为 Parallel 范式的 baseline**，并声称 MixFormer 更省 FLOPs（2.24T vs 23T，相差 10×）——不过要注意 MixFormer 论文里 OneTrans 的配置是"parallel"实现，不是 OneTrans 自己论文里的最优实现，这个 10× 差距需要打折看。

### 4.4 RankMixer vs MixFormer：血缘最近的两个

这两者**同一个思路同一个人马**（RankMixer 和 MixFormer 都有 Zhifang Fan）。核心血缘：

| | RankMixer | MixFormer |
|---|---|---|
| 跨 token 交互 | Multi-head Token Mixing（无参数） | HeadMixing（同一个东西换名字） |
| 内部 FFN | Per-token FFN | Per-head SwiGLU FFN |
| 序列建模 | ❌ 外挂 DIN | ✅ 内建 Cross-Attention |
| MoE | ✅ Sparse-MoE + DTSI | ❌ 未用 MoE |

→ **MixFormer ≈ RankMixer + 内建 Cross-Attention 序列建模 + User-Item Decoupling**。可以理解成 "把 RankMixer 从 FI-only 升级成了统一 backbone"。

---

## 5. 选型指南（给工程师的决策树）

```
Q1. 你的 ranking 系统当前的瓶颈在哪？
    ├─ 特征交互不够强、模型参数卡在 100M 以下
    │      └─ 上 RankMixer（1B + MFU 45%，延迟零增）
    │
    ├─ 序列模块还在 SIM/TWIN，丢了大量长程信息
    │      └─ 上 LONGER（端到端 10k + KV Cache）
    │
    ├─ 两段式架构限制了统一 scaling
    │      │
    │      ├─ 希望"一条纯 Transformer 走到底" + 跨请求 KV cache
    │      │      └─ 上 OneTrans
    │      │
    │      └─ 希望省 FLOPs + 用无参数 mixing + 细粒度 head FFN
    │             └─ 上 MixFormer (尤其 UI-MixFormer)
    │
    └─ 不确定？
           └─ 按时间演化上：
              第一步 RankMixer（FI 侧）
              第二步 LONGER（SM 侧）
              第三步 OneTrans 或 MixFormer（统一 backbone）
```

---

## 6. 架构血缘关系图

```
                    DLRM (传统两段式)
                          │
           ┌──────────────┴──────────────┐
           │                             │
  [FI 模块演化]                     [SM 模块演化]
  DCNv2 → Wukong → HiFormer        DIN → SIM → TWIN → UE
           │                             │
      RankMixer (2025.07)           LONGER (2025.05)
           │                             │
           └──────────────┬──────────────┘
                          │
              "合并两段式" 的尝试
                          │
            ┌─────────────┴─────────────┐
            │                           │
      OneTrans (2025.10)         MixFormer (2026.02)
      纯 Transformer 路线         MLP-Mixer + Cross-Attn 混合路线
      (WWW'26)                   (2602.14110)
```

---

## 7. 一句话终极对比

> - **RankMixer**：非序列特征的千亿级表达力装进 1B 参数、MFU 45%。
> - **LONGER**：把 10k 长序列端到端送进模型，不再绕路。
> - **OneTrans**：序列和非序列本就不该分家，一条 Transformer 就够了。
> - **MixFormer**：一条 Transformer 可以更精细，无参数混合 + 显式 Cross-Attn + User-Item 解耦。

从趋势看，字节推荐 ranking 架构的演化路径是：

> **两段式各自做大 (RankMixer / LONGER) → 尝试合并 (OneTrans) → 精细合并 (MixFormer)**

下一步大概率是：**统一架构 + MoE + 生成式目标 + 超长序列** 的组合拳——RankMixer 的 SMoE + LONGER 的 10k 序列 + OneTrans 的 Cross-Request KV + MixFormer 的 HeadMixing 和 User-Item 解耦，全部装进同一条 backbone。
