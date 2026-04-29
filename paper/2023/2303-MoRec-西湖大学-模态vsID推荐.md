# Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited

> **Authors**: Zheng Yuan*, Fajie Yuan*†, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, Yongxin Ni
> **Affiliation**: 西湖大学 (Westlake University) + 之江实验室 (Zhejiang Lab)
> **Venue**: **SIGIR 2023**，arXiv:2303.13835（2023.03）
> **Links**: [arXiv](https://arxiv.org/abs/2303.13835) · [PDF](https://arxiv.org/pdf/2303.13835) · [Code (IDvs.MoRec)](https://github.com/westlake-repl/IDvs.MoRec)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 过去十多年推荐系统被 **IDRec**（为每个 user/item 学一个 embedding）统治，因为 IDRec 在有足够交互的 warm item 上表现极强；而 **MoRec**（用 BERT / ViT 等模态编码器把文本/图像映射成 item 表征）十年前输给 IDRec，从此被归为"只适合冷启动、跨域"的窄用途方法。但 2023 年一切都变了——**BERT / ViT / GPT 这些基础模型已经极大地进化了**，模态编码的质量可能早已今非昔比。论文想重新严肃地问："**在 regular（包含冷热混合）和 warm item 场景下，现代 MoRec 能不能追平甚至超过 IDRec？如果能，它比 IDRec 多给了什么？**" 这关系到"下一代推荐系统应该押宝 ID 还是 modality"这种**方向性**的判断。
- **Method (方法)**: 在同等 backbone、同等训练 loss 下公平对比 IDRec 和 MoRec。控制三个变量：(1) **Backbone 架构**——两塔 DSSM（点对点）+ 序列 SASRec（seq2seq）两种代表性架构；(2) **Modality Encoder**——NLP 侧用 BERT-tiny/small/base/large + RoBERTa + OPT；CV 侧用 ResNet50 + Swin-T/B + MAE-base；(3) **训练范式**——TS (Two-Stage，冻结 ME 提特征) vs E2E (end-to-end 联合微调 ME)。在三个真实大规模数据集（MIND 文本、HM 视觉、Bili 视觉）上系统评测，并探讨四个 RQ：MoRec 能否打平 IDRec？NLP/CV 进展能否迁移？表示能否通用（pretrain once, reuse everywhere）？实际部署挑战（TS vs E2E 的 100× 效率 gap）。
- **Results (效果)**: 四大核心结论：(1) **MoRec 在 regular 和 warm-start 场景下已与 IDRec 打平、部分超越**——例如 MIND 上 SASRec+BERT-base 的 HR@10 为 18.68（IDRec 17.71，**+5.48%**），Bili 上 SASRec+Swin-T 为 3.28（IDRec 3.03，**+8.25%**），只有 HM 上略低（封面图信息不足以描述款式/价格）；(2) **NLP/CV 的 scaling 进展**（BERT-tiny → large、ResNet → Swin、MAE）**能正向迁移到推荐**，但并非简单线性——更大的 ME 通常给更好的 MoRec，但增益递减；(3) **E2E 训练是必须的**，TS 冻结 ME 提特征会严重落后，但 E2E 的算力 / 训练时间比 TS **高 100× 以上**；(4) **预训练 ME 直接当通用物品表征（"pretrain once, reuse everywhere"）目前仍不现实**——不同域的最优 hyperparameter 差异巨大，目前没有"推荐界的 BERT"。论文强调 **IDRec 在 RS 的霸主地位正被严重挑战**，是首次系统给出证据链的工作。

## 2. Detailed Methodology (详细方法解读)

### 2.1 IDRec vs MoRec：严格的 apple-to-apple 对照设置

![IDRec 和 MoRec 的架构对照（论文 Figure 1）](https://arxiv.org/html/2303.13835v4/x1.png)

对同一个 backbone（DSSM 或 SASRec），只替换 item encoder：

- **IDRec**：item 用可学习 embedding 表 $X^I \in \mathbb{R}^{|I|\times d}$，$i$-th item 直接查 $X^I_i$
- **MoRec**：item 用 **ME(text/image) + DT-Layer**（Dimension Transformation 把 ME 输出映射到推荐侧维度）

**两种 backbone**（Figure 1）：
1. **DSSM**（两塔）：user 塔 + item 塔，点积匹配，单 $\langle u, i \rangle$ pair 一个 loss
2. **SASRec**（seq2seq）：MHSA 把用户历史序列里每个位置的 item 作为 query，对下一个 item 做 next-item prediction，一条长度 $L$ 的序列生成 $L-1$ 个 loss

两种 loss 都用 BCE：

$$\min -\sum_{u\in\mathcal{U}}\sum_{i\in[2,\ldots,L]}\log\sigma(\hat y_{ui}) + \log(1-\sigma(\hat y_{uj}))$$

其中 $i$ 为正样本、$j$ 为负样本。所有其它超参（embedding dim、dropout、layer 数、优化器）**IDRec 先调到最优，MoRec 尽量复用然后小范围搜索**——这个设定对 MoRec 其实偏保守。

### 2.2 实验变量的"四个维度"

| 维度 | 选项 |
|------|------|
| **Backbone** | DSSM（双塔 CTR 风格）/ SASRec（seq2seq 序列推荐） |
| **Modality Encoder** (NLP) | BERT-tiny / BERT-small / BERT-base / BERT-large / RoBERTa-base / OPT-125M |
| **Modality Encoder** (CV) | ResNet50 / Swin-Tiny / Swin-Base / MAE-base |
| **训练范式** | TS（冻结 ME 预提特征）/ E2E（ME + 推荐主干联合 fine-tune） |

### 2.3 三个真实数据集，两种模态

- **MIND**（Microsoft 新闻，文本）：630K 用户 / 80K 物品 / 8.4M 交互——news title 作为 item 文本
- **HM**（H&M 电商，视觉）：500K / 87K / 5.5M——服装封面图作为 item 图像
- **Bili**（B 站视频，视觉）：400K / 128K / 4.4M——视频封面图作为 item 图像

数据预处理：图像统一 224×224，文本 title ≤ 30 tokens（覆盖 99%）。MIND 每用户取最近 23 条行为作序列，HM/Bili 取最近 13 条（视觉显存太大）。Leave-one-out 划分 train/valid/test。

### 2.4 四个核心研究问题

- **Q(i)**：现代 ME 下，MoRec 能否在 regular 和 warm-start 场景打平 IDRec？
- **Q(ii)**：NLP/CV 领域的技术进步能否迁移到 MoRec？
- **Q(iii)**：基础模型学到的表征是否像宣称的那样"通用"？能否从 NLP/CV 里直接拿来用？
- **Q(iv)**：如何有效使用 ME？TS vs E2E 各自的 pros/cons？

## 3. Experiment Analysis (实验结果解读)

### 3.1 Q(i)：MoRec 能打平 IDRec 吗？（Table 2）

**Regular setting**（完整测试集，含冷热混合）：

| 数据集 | 指标 | IDRec (DSSM) | IDRec (SASRec) | MoRec-BERT-base (SASRec) | MoRec-Swin-T (SASRec) | MoRec 增幅 |
|-------|------|--------------|----------------|--------------------------|------------------------|-----------|
| MIND | HR@10 | 3.58 | 17.71 | **18.68** | – | **+5.48%** |
| MIND | NDCG@10 | 1.69 | 9.52 | **10.02** | – | +5.25% |
| HM | HR@10 | 1.14 | 4.01 | – | 3.98 | −0.75% |
| HM | NDCG@10 | 0.56 | 1.63 | – | 1.66 | +1.84% |
| Bili | HR@10 | – | 3.03 | – | **3.28** | **+8.25%** |

→ **MIND 和 Bili 上 MoRec 已经领先 IDRec**，HM 上略输（论文解释：HM 服装的购买决策依赖价格/款式细节，**只用封面图**不足以捕捉）。

**Warm-20 setting**（Table 3，去掉交互数 < 20 的冷 item）：

| 数据集 | IDRec | MoRec |
|-------|-------|-------|
| MIND | 20.12 | **20.19** |
| HM | 7.89 | **8.05** |
| Bili | 3.48 | **3.57** |

→ **连 warm-20 都已追平/反超**——这是 MoRec 最震撼的结果。IDRec 历来引以为傲的"热门物品上的压倒性优势"已经不复存在。

### 3.2 Q(ii)：NLP/CV 的 scaling 能迁移吗？

- **NLP 侧**：BERT-tiny → small → base → large，MoRec 的 HR@10 **单调提升**（受益于更大 ME）；但 **RoBERTa-base 并未显著超过 BERT-base**——说明**预训练方法的差异比规模更关键**
- **CV 侧**：ResNet50 → Swin-T/B → MAE-base，Swin 显著强于 ResNet，**MAE（自监督）能进一步提升**
- **关键结论**：NLP/CV 的技术进展**能正向迁移**，但映射关系不是简单的"更大 ME = 更好 MoRec"，需要针对推荐任务再调

### 3.3 Q(iii)：表征是否真的"通用"？能否 pretrain once, reuse everywhere？

**不能**。论文的关键发现：
- 把冻结的 ME 直接当特征（TS 范式）在所有数据集上都输给 E2E 版本一大截
- 不同数据集的最优 ME 不同（BERT-base 在 MIND 最优，但在 Books 上未必）
- **没有"推荐界的 BERT"**——目前还不存在一个"预训练一次、所有推荐场景都能用"的通用物品表征

→ 推荐域的 foundation model **还没出现**，这是个明确的 open problem（**后来 HLLM / LEARN 等工作正是朝这个方向走**）。

### 3.4 Q(iv)：TS vs E2E 的效率-精度权衡（Table 6）

- **E2E 必须**：TS 性能远落后，但 E2E 训练时间和算力比 TS **高 100×+**（尤其视觉）
- MoRec 的 hyperparameter tuning 成本远高于 IDRec——**同样的调参预算下 MoRec 可能被严重低估**
- 论文明确喊话：**"如何高效地为 MoRec 调参，是一个重要但尚未被研究的问题"**

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **首次给 "ID vs Modality" 之争画上句号**：十年前的老结论"MoRec 只适合冷启动"在 2023 年已经**过时**。这篇论文用严谨的对照实验证明 **MoRec 在 regular + warm 场景下已经追平或超越 IDRec**，这是推荐系统领域方向性的结论。
2. **控制变量的实验设计堪称教科书级**：同 backbone、同 loss、同数据、同评测协议，只换 item encoder——对 MoRec 甚至还故意用 IDRec 的最优超参（对 MoRec 偏保守），结果仍然 MoRec 胜出，说服力极强。
3. **覆盖两种模态 + 两种 backbone + 多个 ME size**：MIND（文本）+ HM/Bili（视觉）× DSSM + SASRec × BERT 家族 + ResNet/Swin/MAE 家族，matrix 做得很完整。
4. **提出了四个明确的 open problem**：E2E 的效率、MoRec 的调参、"推荐界的 BERT"、跨域迁移——**HLLM (2024.09) / LEARN / IDProxy 等后续工作的研究议程基本都来自这里**。
5. **代码和数据集 Bili 全开源**（westlake-repl/IDvs.MoRec），可复现性强。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **只测了两塔和 SASRec**：没有测 DIN / DIEN / 更现代的 HSTU / RankMixer / LONGER 等工业架构。结论能否外推到这些架构需要更多验证——但**主要结论的方向性**（MoRec 追上 IDRec）应该不会变。
2. **HM 数据的 MoRec 劣势解释不充分**：论文承认"封面图不够"，但没给出"如果加上价格+款式文本能不能救"的对照——这恰好是后来多模态工作的突破口。
3. **User 侧仍是 ID**：论文只把 item 侧 ID 换成 modality，**user 侧还是纯 ID embedding**。一个更激进的方向是 **user 侧也 modality 化**（用用户行为文本化、简历、评论等），但这超出了本文范围。
4. **没有多模态融合实验**：文本 + 图像同时用会不会更好？（HM 上加上商品描述是否能补齐封面图的短板？）这篇论文每个数据集只用一种模态。
5. **Negative sampling 策略未充分探索**：MoRec 因为 item 表征来自 ME、相似物品的 embedding 会更聚拢，**负采样策略可能需要专门设计**（例如 hard negative mining），论文只用了简单的随机负采样。
6. **从这篇到 HLLM / LEARN 的演化**：这篇论文证明"ME 能打平 ID"后，下一个自然问题是"ME 能做多大？能不能把 LLM 当 ME？"——**HLLM（2024.09，字节）正是这条路线的最新答案**（7B 规模的 Item LLM + User LLM）。两篇一起读能看清推荐系统从"ID 霸权"到"LLM-based"的完整演化。
7. **可能的下一步**：
   - **多模态融合 ME**（CLIP / BLIP-2 同时吃图像+文本）替代单模态 ME
   - **E2E + LoRA / Adapter**（解决 100× 算力 gap）
   - **统一 user encoder + item encoder 的 LLM**（HLLM 的思路）
   - **RS-specific foundation model**（类似 NLP 的 BERT，一次预训练、多域微调）
