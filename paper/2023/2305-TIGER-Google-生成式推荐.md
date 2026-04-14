# Recommender Systems with Generative Retrieval

> **Authors**: Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
> **Affiliation**: Google DeepMind / Google / University of Wisconsin-Madison
> **Venue**: NeurIPS 2023
> **Links**: [arXiv](https://arxiv.org/abs/2305.05065)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 传统推荐系统的检索阶段依赖 dual-encoder（双塔模型）将 query 和 item 映射到同一向量空间，再通过 ANN（近似最近邻）搜索召回候选。这种范式存在几个问题：(1) 需要为每个 item 维护高维 embedding 索引，内存开销大；(2) item 表示依赖随机分配的 atomic ID，无法在语义相似的 item 间共享知识；(3) 新 item（cold-start）没有交互历史，难以被召回。
- **Method (方法)**: 提出 **TIGER**（Transformer Index for GEnerative Recommenders），用生成式检索（generative retrieval）替代传统的 embedding-based 检索。核心思想是为每个 item 生成语义有意义的 **Semantic ID**（由 RQ-VAE 对内容 embedding 进行层次化量化得到），然后训练 Transformer encoder-decoder 模型，给定用户交互历史的 Semantic ID 序列，自回归地预测下一个 item 的 Semantic ID。
- **Results (效果)**: 在 Amazon Beauty/Sports/Toys 三个数据集上全面超越 SASRec、S3-Rec、BERT4Rec 等 SOTA baseline，Beauty 上 NDCG@5 提升 29%、Recall@5 提升 17.3%；同时具备冷启动推荐和可控多样性的新能力。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

TIGER 框架分为两个阶段：

- **阶段一：Semantic ID 生成** — 使用预训练文本编码器（Sentence-T5）将 item 的内容特征（标题、价格、品牌、类目等）编码为 768 维的语义 embedding，再通过 RQ-VAE 量化为一组有序的离散 codeword（即 Semantic ID）
- **阶段二：生成式检索** — 将用户交互历史中的 item 替换为其 Semantic ID 序列，训练 Transformer encoder-decoder 模型，自回归地预测下一个 item 的 Semantic ID
- **推理时**: 使用 beam search 解码，生成候选 Semantic ID，再通过 lookup table 映射回实际 item

**关键范式转变**: 传统方法是"编码 query → 向量空间搜索 → 返回 item embedding 最近邻"，TIGER 是"编码历史序列 → 直接生成 item 的 Semantic ID → 查表返回 item"。Transformer 的参数本身就是索引（index），无需额外的 ANN 索引。

### 2.2 Core Modules (核心模块详解)

**模块一：RQ-VAE Semantic ID 生成**

- **功能**: 将 item 的连续语义 embedding 量化为层次化的离散 codeword 元组
- **输入**: item 的语义 embedding $x \in \mathbb{R}^d$（由 Sentence-T5 生成，$d=768$）
- **输出**: $m$-tuple Semantic ID $(c_0, c_1, \ldots, c_{m-1})$

- **RQ-VAE 编码过程**:

1. 编码器将输入 $x$ 映射为 latent representation: $z := E(x)$
2. 初始残差: $r_0 := z$
3. 在第 $d$ 层，从 codebook $C_d = \{e_k\}_{k=1}^{K}$ 中找到最近向量:

$$c_d = \arg\min_i \|r_d - e_i\|$$

4. 计算残差: $r_{d+1} := r_d - e_{c_d}$
5. 重复 $m$ 次，得到 Semantic ID $(c_0, c_1, \ldots, c_{m-1})$

- $K = 256$: 每层 codebook 大小
- $m = 3$: 量化层数（加上碰撞处理的第 4 位共 4 个 codeword）
- **直觉理解**: RQ-VAE 类似于"逐步逼近"——第一层找到最粗粒度的近似（如"这是护发产品"），第二层在残差上进一步细化（如"这是洗发水"），第三层继续精确（如"这是去屑洗发水"）。这使得 Semantic ID 天然具有层次结构，前缀相同的 item 在语义上更接近。

- **RQ-VAE Loss**:

$$\mathcal{L}(x) = \mathcal{L}_{recon} + \mathcal{L}_{rqvae}$$

$$\mathcal{L}_{recon} = \|x - \hat{x}\|^2$$

$$\mathcal{L}_{rqvae} = \sum_{d=0}^{m-1} \left(\|sg[r_d] - e_{c_d}\|^2 + \beta\|r_d - sg[e_{c_d}]\|^2\right)$$

- $\hat{x}$: 解码器从量化表示重建的输出
- $sg[\cdot]$: stop-gradient 操作
- $\beta = 0.25$: commitment loss 系数
- **直觉理解**: 第一项让解码器学会从量化表示中重建原始 embedding，第二项让 codebook 向量向数据靠拢（同时 commitment loss 让 encoder 输出靠近 codebook 向量），三者联合优化

- **碰撞处理**: 多个 item 可能映射到相同的 3-tuple Semantic ID，通过追加第 4 个唯一 token 消歧。如两个 item 共享 $(12, 24, 52)$，则分别表示为 $(12, 24, 52, 0)$ 和 $(12, 24, 52, 1)$

**RQ-VAE 实现细节**:
- 编码器: 3 层 MLP（512 → 256 → 128），ReLU 激活，latent 维度 32
- 3 层残差量化，每层 codebook 大小 256，向量维度 32
- 训练 20k epochs，Adagrad 优化器，lr=0.4，batch size=1024
- 使用 k-means 初始化 codebook 防止 codebook collapse

**模块二：Transformer Encoder-Decoder 生成式检索**

- **功能**: 给定用户历史交互的 Semantic ID 序列，自回归预测下一个 item 的 Semantic ID
- **输入构造**: 用户 token $t_u$ + 历史 item 的 Semantic ID 序列拼接:

$$\text{Input} = (t_u, c_{1,0}, c_{1,1}, c_{1,2}, c_{1,3}, c_{2,0}, \ldots, c_{n,3})$$

- **输出**: 下一个 item 的 Semantic ID $(c_{n+1,0}, c_{n+1,1}, c_{n+1,2}, c_{n+1,3})$
- **模型配置**:
  - Encoder: 4 层 Transformer，6 个 self-attention heads，head 维度 64
  - Decoder: 4 层 Transformer，自回归生成
  - MLP 维度 1024，input 维度 128
  - Dropout 0.1，总参数约 13M
- **词表**: 1024 个 semantic codeword tokens（256×4）+ 2000 个 user ID tokens（通过 hashing trick 映射）
- **直觉理解**: 与 NLP 中的 seq2seq 翻译类似——"翻译"用户的历史行为序列为下一个 item 的"语义地址"。encoder 理解用户意图，decoder 逐 token 生成目标 item 的 Semantic ID

**推理时**: 使用 beam search 解码多个候选 Semantic ID，通过 lookup table 映射回实际 item。无效 ID（不对应任何 item）的比例很低（top-10 时约 0.1%-1.6%）。

### 2.3 Training Strategy (训练策略)

- **训练数据**: 用户交互历史按时间排序，限制最近 20 个 item
- **评估协议**: Leave-one-out——最后一个 item 做测试，倒数第二个做验证，其余训练
- **训练步数**: Beauty/Sports 200k steps，Toys 100k steps（数据量较小）
- **Batch size**: 256
- **学习率**: 前 10k steps lr=0.01，之后 inverse square root decay
- **框架**: 基于 T5X 实现

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **数据集**: Amazon Product Reviews 三个类目

| 数据集 | 用户数 | 物品数 | 平均序列长度 |
|--------|--------|--------|-------------|
| Beauty | 22,363 | 12,101 | 8.87 |
| Sports and Outdoors | 35,598 | 18,357 | 8.32 |
| Toys and Games | 19,412 | 11,924 | 8.63 |

- **评价指标**: Recall@5/10, NDCG@5/10
- **主要 Baselines**: GRU4Rec, Caser, HGN, SASRec, BERT4Rec, FDSA, S3-Rec, P5

### 3.2 Main Results (主实验结果)

**Table 1: 序列推荐性能对比**:

| 方法 | Beauty R@5 | Beauty N@5 | Sports R@5 | Sports N@5 | Toys R@5 | Toys N@5 |
|------|-----------|-----------|-----------|-----------|---------|---------|
| SASRec | 0.0387 | 0.0249 | 0.0233 | 0.0154 | 0.0463 | 0.0306 |
| S3-Rec | 0.0387 | 0.0244 | 0.0251 | 0.0161 | 0.0443 | 0.0294 |
| **TIGER** | **0.0454** | **0.0321** | **0.0264** | **0.0181** | **0.0521** | **0.0371** |
| 提升 | +17.3% | +29.0% | +5.2% | +12.6% | +12.5% | +21.2% |

关键观察：
- TIGER 在所有三个数据集、所有指标上全面取得最优
- **Beauty 数据集提升最为显著**: NDCG@5 提升 29%（vs SASRec），Recall@5 提升 17.3%（vs S3-Rec）
- Toys 数据集 NDCG@5 提升 21.2%，NDCG@10 提升 15.0%
- 所有 baseline 都是基于 dual-encoder + ANN 的传统范式，TIGER 作为全新的生成式范式，取得了质的突破

### 3.3 Ablation Study (消融实验)

**Table 2: ID 生成方式对比**:

| ID 类型 | Beauty R@5 | Beauty N@5 | Sports R@5 | Sports N@5 | Toys R@5 | Toys N@5 |
|---------|-----------|-----------|-----------|-----------|---------|---------|
| Random ID | 0.0296 | 0.0205 | 0.007 | 0.005 | 0.0362 | 0.0270 |
| LSH SID | 0.0379 | 0.0259 | 0.0215 | 0.0146 | 0.0412 | 0.0299 |
| **RQ-VAE SID** | **0.0454** | **0.0321** | **0.0264** | **0.0181** | **0.0521** | **0.0371** |

关键发现：
- **Semantic ID vs Random ID**: RQ-VAE Semantic ID 全面碾压随机 ID，尤其在 Sports 数据集上 Recall@5 从 0.007 提升到 0.0264（**3.77 倍**），证明语义信息对生成式检索至关重要
- **RQ-VAE vs LSH**: RQ-VAE 一致优于 LSH，说明非线性 DNN 量化比随机投影哈希能学到更好的语义表示
- 这组消融实验是全文最关键的验证——Semantic ID 是 TIGER 成功的核心

**模型深度消融（Table 5, Beauty 数据集）**:

| 层数 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|------|----------|--------|-----------|---------|
| 3 | 0.04499 | 0.03062 | 0.06699 | 0.03768 |
| 4 | 0.04540 | 0.03210 | 0.06480 | 0.03840 |
| 5 | 0.04633 | 0.03206 | 0.06596 | 0.03834 |

→ 增加层数有轻微提升但不显著，说明 13M 参数的小模型已足够

**User ID 的作用（Table 8, Beauty 数据集）**:
- 加入 user ID token: Recall@5 从 0.04458 提升至 0.0454，NDCG@10 从 0.0367 提升至 0.0384
- 说明用户个性化信息有正面贡献

### 3.4 New Capabilities (新能力)

**冷启动推荐（Cold-Start）**:
- 模拟场景：从训练集中移除 5% 的 test items 作为"未见过的新 item"
- TIGER 可以通过 Semantic ID 的前缀匹配检索新 item——即使模型从未见过某个 item，只要它的 Semantic ID 与已知 item 共享前缀，就有机会被召回
- 在所有 Recall@K 指标上优于 Semantic_KNN baseline
- **直觉理解**: 传统模型用随机 atomic ID 表示 item，新 item 的 ID 从未在训练中出现过，完全无法被召回。TIGER 用语义 ID 表示 item，新 item 的语义 ID 可能与训练中见过的 item 共享 codeword（如同属"护发产品"），从而被泛化召回

**推荐多样性（Diversity）**:
- 通过解码时的 temperature sampling 控制多样性：$T=1.0$ 时 Entropy@10 = 0.76，$T=2.0$ 时提升至 1.38
- RQ-VAE 的层次结构使得可以在不同粒度上采样：对第一层 token 采样 = 在粗粒度类目间探索，对后续 token 采样 = 在类目内探索
- **直觉理解**: 传统模型的多样性需要额外的重排序策略，TIGER 天然支持通过温度参数调节——这是生成式模型的固有优势

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **范式级创新**: TIGER 开创了推荐系统中"生成式检索"的全新范式——从"搜索匹配"转变为"直接生成"，Transformer 的参数本身就是 item 索引，无需维护额外的 ANN 索引。这一思路后来深刻影响了整个生成式推荐（Generative Recommendation）方向
2. **Semantic ID 的设计精妙**: 利用 RQ-VAE 的层次化残差量化，天然赋予 Semantic ID 从粗到细的语义层次结构。前缀相同的 item 在语义上相近，这一性质同时支撑了冷启动、多样性和知识共享三大能力
3. **消融实验充分说服力强**: Random ID vs LSH vs RQ-VAE 的三级对比清晰展示了每个设计选择的贡献，尤其 Random ID 的巨大差距直接证明了语义信息的关键作用
4. **冷启动和多样性的"免费午餐"**: 这两个新能力并非额外设计，而是 Semantic ID + 生成式框架的自然衍生，体现了范式转变带来的本质优势
5. **Embedding table 大小不随 item 数线性增长**: 词表仅需 1024 个 semantic codeword embedding（vs 传统的 10K-20K item embedding），内存效率大幅提升

### 4.2 Limitations & Improvements (不足与改进方向)

1. **推理效率问题**: 自回归 beam search 解码比 ANN 检索慢得多，论文也坦诚这一点。后续工作可以探索非自回归解码、prefix-constrained decoding 等加速方案
2. **数据集规模偏小**: 三个 Amazon 数据集的 item 数仅 1-2 万，序列长度也很短（中位数 6）。在工业级百万/亿级 item corpus 上的 scaling 能力尚未验证
3. **Semantic ID 依赖内容特征质量**: 如果 item 的文本特征质量差或缺失，Sentence-T5 生成的 embedding 质量会下降，进而影响 Semantic ID 的语义性。对于纯行为驱动（无内容特征）的场景可能不适用
4. **无效 ID 问题**: 虽然比例较低（top-10 约 0.1%-1.6%），但在工业场景中仍需处理。论文提到的 prefix matching 是一个好方向但未实验验证
5. **RQ-VAE 与推荐模型分离训练**: 两阶段的训练方式可能导致 Semantic ID 不是推荐任务最优的表示。后续工作（如 MaskGR 等）已开始探索端到端联合训练
6. **可能的改进方向**:
   - 探索 RQ-VAE 与推荐模型的联合训练（end-to-end optimization）
   - 引入行为信号（collaborative signal）到 Semantic ID 的生成中，不仅依赖内容特征
   - 将 TIGER 范式与 LLM 结合，利用 LLM 的世界知识增强语义理解
   - 探索在工业级大规模 corpus 上的 scaling 方案
