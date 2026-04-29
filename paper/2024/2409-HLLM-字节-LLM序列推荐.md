# HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling

> **Authors**: Junyi Chen*, Lu Chi*, Bingyue Peng, Zehuan Yuan†
> **Affiliation**: ByteDance
> **Venue**: arXiv:2409.12740（2024.09，AAAI 2025 版本）
> **Links**: [arXiv](https://arxiv.org/abs/2409.12740) · [PDF](https://arxiv.org/pdf/2409.12740) · [Code (bytedance/HLLM)](https://github.com/bytedance/HLLM)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: LLM 应用到推荐系统已经有很多尝试，但业界面临三个没被解决的"元问题"：(1) **LLM 预训练权重的"世界知识"对推荐到底有没有价值？** 很多工作把行为历史 flatten 成长文本喂给 LLM，输入几千 token、生成又要多次 forward、quadratic attention 爆炸——效率极差、效果也只比传统模型好一点，让人怀疑"真的是 LLM 在起作用吗"；(2) **是否需要针对推荐任务做 fine-tune？** 通用 LLM 的 SFT 对对话能力好，但对推荐任务是不是也有帮助？还是反而伤害？(3) **推荐系统能不能复现 LLM 那种 scaling law？** 之前 HSTU / CLUE 在 100M–1B 级别验证过，但 **1B–7B 级别还是 open question**。
- **Method (方法)**: 提出 **Hierarchical Large Language Model (HLLM)**，一个**两层 LLM** 的干净架构：**Item LLM** 把每个物品的文本描述 + 一个特殊 `[ITEM]` token 压缩成 item embedding；**User LLM** 以用户历史的 item embedding 序列为输入，预测下一个 item 的 embedding。两个 LLM 都**保留预训练权重**（User LLM 丢掉 word embedding 但保留其它层），都用推荐目标 fine-tune。训练目标分两种：**Generative**（InfoNCE next-item prediction，可学习温度）；**Discriminative**（分类 loss，有 Early / Late Fusion 两种 target 融合方式）。
- **Results (效果)**: **学术数据集** PixelRec 和 Amazon Book Reviews 上全面 SOTA——Pixel8M 上 HLLM-1B 比最弱 baseline 平均 **+22.93%**，Books 上 **+108.68%**；加大 batch + 负样本后 HLLM-7B 在 Books 上比最强 baseline 再涨，达到 **+169.58%**（HR@10=17.65 vs baseline 的 7.54）。**Scaling**：Item LLM 从 BERT-Base 扩到 TinyLlama-1.1B，HR@10 从 4.02 → 5.24；User LLM 同步扩大也单调涨。**训练数据效率**：HLLM 只需 ID 模型 **1/6–1/4** 的数据即可追平。**线上 A/B**（抖音 ranking 场景）：核心指标 **+0.705%**。

## 2. Detailed Methodology (详细方法解读)

### 2.1 要解决的三个元问题

![HLLM 架构](https://github.com/bytedance/HLLM/raw/main/images/overall.png)

HLLM 通过实证回答三个 RQ：

1. **RQ1 · 预训练 + fine-tune 有价值吗**：Table 2 的 2×2 消融——Item LLM 和 User LLM 各自"从零 vs 预训练"四个组合，**两个都用预训练** R@10 最高（5.581），**两个都 scratch** 最低（5.063）。且 Table 3 显示随预训练 token 数（0T → 3T）增长，R@10 从 5.047 → 5.581 单调涨——**预训练权重的 scaling 也会传递到推荐任务**。
2. **RQ2 · 是否有 scaling law**：Item LLM 从 BERT-Base (110M) → BERT-Large (340M) → TinyLlama (1.1B)，R@10 从 4.02 → 4.64 → 5.24；User LLM 从 4M → 0.1B → 1.1B 同步提升；1B → 7B 在 Books 上仍有提升（R@10 从 17.34 → 17.65），**证明 7B 级别推荐 scaling law 成立**。
3. **RQ3 · 是否必须 fine-tune**：Table 4 四组对照——**Item LLM 冻结、User LLM 可学** R@10 只有 0.945（远低于 ID baseline SASRec-1B 的 2.868）；**Item 可学、User 冻结** 只有 2.47；**两者都可学**才有 5.581。→ **预训练 LLM 不能直接当推荐特征提取器，必须针对推荐目标 fine-tune**。

### 2.2 Item LLM：把文本描述压成一个 embedding

- 输入：物品的文本属性（title、tag、description 等）flatten 成 $T=\{t_1,\ldots,t_m\}$，**末尾加一个特殊 `[ITEM]` token**
- 整条序列用固定 prompt（"Compress the following sentence into embedding:"）前缀，过 LLM 后，**取最后一层 `[ITEM]` token 位置的 hidden state 作为 item embedding**
- 设计来源：BERT 的 `[CLS]` token + OpenAI text embedding 的 `[EOS]` 聚合思路
- 消融（Table 9）：文本越丰富越好（Tag+Title+Description），文本 token 截断长度 256 比 64 更好

### 2.3 User LLM：item embedding 序列的 next-item prediction

- 输入：$\{E_1, E_2, \ldots, E_n\}$——用户历史的 item embedding 序列
- 输出：$\{E'_2, E'_3, \ldots, E'_{n+1}\}$——每个位置预测下一个 item embedding
- 关键设计：**丢掉原 LLM 的 word embedding**（因为输入不再是 token id 而是 item embedding），但**保留 Transformer 主体所有预训练权重**
- 位置 embedding / attention mask / FFN / LayerNorm 全部复用
- 消融（Table 11）：user sequence 长度 10 → 50，HR@10 从 7.564 → 7.631（有提升但边际递减）

### 2.4 训练目标：Generative + Discriminative

**Generative（next-item prediction，用于学术评测）**：

$$\mathcal{L}_\text{gen} = -\sum_{j=1}^b \sum_{i=2}^n \log \frac{e^{s(E'_{j,i}, E_{j,i})}}{e^{s(E'_{j,i}, E_{j,i})} + \sum_k e^{s(E'_{j,i}, E_{j,i,k})}}$$

其中 $s(\cdot,\cdot)$ 是可学习温度的相似度函数，负样本 $E_{j,i,k}$ 从其它用户的序列里随机采。

**Discriminative（分类，用于工业上线）**：给定 $(u, i_\text{tgt})$，预测用户是否对目标 item 感兴趣。论文给出两个变体（Figure 2）：

- **Early Fusion**：$E_\text{tgt}$ 作为一个额外 token 拼到用户序列末尾，一起过 User LLM 生成 cross feature，送预测头 → **效果好但每个候选都要重算 User LLM**，不能跨候选复用
- **Late Fusion**：User LLM 先输出用户表征（用额外 `[USER]` token 聚合），再和 $E_\text{tgt}$ 一起送预测头 → **效果略弱但能跨候选共享**，适合线上部署

最终 loss：

$$\mathcal{L}_\text{dis} = \lambda \mathcal{L}_\text{gen} + \mathcal{L}_\text{cls}$$

$\mathcal{L}_\text{gen}$ 作为辅助 loss 进一步提点。

### 2.5 部署架构（Figure 4）

线上三阶段训练：

1. **Stage I**：端到端训练 Item LLM + User LLM（user 序列截断到 150，节省显存）
2. **Stage II**：**冻结 Item LLM**，离线预计算并存储所有 item 的 embedding；只训 User LLM，把用户序列扩到 **1000**
3. **Stage III**：冻结整个 HLLM，提取所有 user 的 embedding，和 item embedding 一起作为特征灌进**原有的在线 ranking 模型**

推理时：
- item embedding 在物品创建时一次性算好；user embedding 每天更新（只处理前一天活跃用户）
- 原 ranking 模型只多查两个 embedding 向量 → **在线推理时间几乎不变**

## 3. Experiment Analysis (实验结果解读)

### 3.1 实验设置

- **学术数据集**：
  - **PixelRec**（抖音图像推荐公开版）：200K / 1M / 8M 三档
  - **Amazon Book Reviews**：694K 用户 × 686K 物品 × 10M 交互
- **模型**：HLLM-1B（Item + User 均用 TinyLlama-1.1B）、HLLM-7B（均用 Baichuan2-7B）
- **Baselines**：SASRec、HSTU、LEARN；还自己实现了 **SASRec-1B**（把 SASRec 主干换成 TinyLlama 大小）和 **HSTU-1B**
- **训练**：lr=1e-4，HLLM 只训 5 epoch（其它 baseline 训 50/200 epoch），学术比较已经碾压

### 3.2 主实验（Table 7，Pixel8M + Books）

| 数据集 | 模型 | R@10 | R@200 | N@10 | Impv (avg) |
|-------|------|------|-------|------|-----------|
| **Pixel8M** | SASRecvit | - | - | 1.941 | −27.72% |
| | HSTU (baseline) | 10.315 | 18.327 | 3.939 | **0.0%** |
| | SASRec-1B | 10.899 | 19.044 | 4.166 | +4.83% |
| | HSTU-1B | 11.010 | 19.393 | 4.159 | +5.37% |
| | **HLLM-1B** | **12.475** | **21.179** | **4.919** | **+22.93%** |
| **Books** | SASRec (2018) | 7.54 | 14.31 | 2.60 | 0.0% |
| | HSTU-large | 10.82 | 19.08 | 3.93 | +47.80% |
| | SASRec∗ | 11.91 | 21.02 | 4.40 | +64.96% |
| | HSTU-1B | 12.03 | 21.60 | 4.36 | +64.37% |
| | **HLLM-1B** | **14.61** | **24.78** | **5.64** | **+108.68%** |
| | HLLM-1B† (bs↑ neg↑) | **17.34** | **27.22** | **7.41** | **+166.42%** |
| | **HLLM-7B†** | **17.65** | **27.59** | **7.50** | **+169.58%** |

→ **HLLM 的 scaling 曲线远比 ID 模型陡**：ID 模型 SASRec-1B 在 Books 上反而比 SASRec∗ 掉点，HLLM 却能吃 7B 参数继续涨。

### 3.3 Item Caching 的实用性（Table 8）

部署时冻结 Item LLM（预存 embedding），只训 User LLM：

| Method | R@5 | R@10 | N@5 |
|--------|-----|------|-----|
| HSTU-1B | 3.501 | 5.120 | 2.358 |
| **HLLM-1B-cache**（Item 冻结）| **3.585** | **5.218** | **2.432** |
| HLLM-1B（全 fine-tune）| 4.278 | 6.106 | 2.935 |

→ 冻结 Item LLM 后性能比 HSTU-1B 还好，**证明 item embedding caching 在工业上可行**；全 fine-tune 能再涨一截，但 cache 版本已经打赢 SOTA。

### 3.4 训练数据效率（Figure 3）

在 Pixel8M 上用不同数据量训练：
- HLLM-1B 用 **1M 数据**就达到 HSTU-1B 用 8M 数据的水平——**数据效率 ~6×**
- 模型越大，数据效率优势越明显

### 3.5 线上 A/B（抖音 ranking）

采用 Discriminative + Late Fusion 变体、三阶段训练流程部署，**核心指标 +0.705%**。在抖音这种体量下这是一个显著收益。

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **干净的两层 LLM 架构**：Item LLM 处理物品内容、User LLM 处理用户行为——把"内容理解"和"兴趣建模"彻底解耦。相比 text-in-text-out 的 LLM4Rec 流派（LLaRA、PALR 等）输入长度爆炸、推理成本高，HLLM **用 item embedding 而不是 text token 作为 User LLM 的输入**，序列长度直接回到传统 ID 模型的量级。
2. **严谨地回答了三个元问题**：预训练权重有用 ✓、fine-tune 必须 ✓、scaling law 到 7B 仍成立 ✓。这三条结论是 LLM4Rec 领域此前悬而未决的，HLLM 用 2×2 消融 × 参数扫描 × 数据扫描 全部给了答案。
3. **Item Caching 的工业实用性**：两层架构天然解耦 → 可以离线预计算 item embedding，在线只跑 User LLM → 推理延迟几乎不增。**这是 HLLM 能在抖音实际上线的工程关键**。
4. **三阶段训练策略**：Stage I 端到端学 + Stage II 冻 item 扩 user 序列 + Stage III 冻全栈当特征。这种"逐步冻结、逐步扩展"的策略值得所有 LLM4Rec 团队学习。
5. **代码全开源**（bytedance/HLLM），包含训练/评测完整 pipeline，复现门槛低。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **纯文本输入，未直接用多模态**：HLLM 的 Item LLM 只吃文本描述（title/tag/description），图像、视频、音频都需要先被文本描述化。结合 **多模态 LLM 做 Item Encoder** 是自然下一步（作者已在 appendix 讨论多模态扩展）。
2. **User LLM 的 word embedding 丢弃有点浪费**：pre-trained word embedding 里包含了大量 token 共现统计，完全丢掉等于浪费一部分预训练信号。或许可以尝试"用 item embedding 和 word embedding 共享投影空间"的做法。
3. **Discriminative 训练仍以 CTR 为主**：Late Fusion 为了部署简单牺牲了 Early Fusion 的精度，如何把两者的优势都拿到（类似 MixFormer 的 User-Item Decoupling 或 OneTrans 的 Cross-Request KV）是明显方向。
4. **7B 上限未探明**：论文只跑到 7B，**scaling 曲线尚未饱和**；但 7B 已经贴工业上线的算力天花板，继续扩到 13B/70B 需要新的工程优化（KV cache / MoE / 量化）。
5. **和 ID 模型结合的空间**：HLLM 现在是**纯内容派**（完全用 text content），但工业上 ID embedding 仍然非常强（尤其热门 item）。把 HLLM embedding 和 ID embedding **拼接或 gating 融合**，可能在头部场景再拿一截收益——这也是 LEARN（作者之前的工作）的思路。
6. **可能的改进方向**：
   - Item LLM 换成 **多模态 LLM**（LLaVA / Qwen-VL），直接吃图像/视频
   - 把 **HLLM 的 embedding 输出接到 OneTrans / MixFormer 的 NS-token 口**，形成 "LLM Encoder → 统一 Ranking Backbone" 的两阶段
   - 在 Stage II 期间做 **RLHF / DPO 式的偏好学习**，让 User LLM 不只预测 next item 还能学习用户长期满意度
