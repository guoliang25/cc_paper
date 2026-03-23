# Efficient, Property-Aligned Fan-Out Retrieval via RL-Compiled Diffusion

> **Authors**: Pengcheng Jiang, Judith Yue Li, Moonkyung Ryu, R. Lily Hu, Kun Su, Zhong Yi Wan, Liam Hebert, Hao Peng, Jiawei Han, Dima Kuzmin, Craig Boutilier
> **Affiliation**: University of Illinois Urbana-Champaign (UIUC) / Google DeepMind
> **Venue**: arXiv preprint, March 2026
> **Links**: [arXiv](https://arxiv.org/abs/2603.06397)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 在推荐系统和检索场景中，用户经常有"集合级"需求——比如"帮我搭配一套波西米亚风的穿搭"或"推荐一组适合派对的歌单"。这要求系统一次性检索出一组满足多个属性（相关性、多样性、风格一致性）的结果，而非单个 item。传统方法要么用零样本 LLM 直接生成（质量不稳定），要么用 Best-of-N 采样（推理成本高），都不理想。
- **Method (方法)**: 提出 **R4T (RL as a Recipe for Training)** 三阶段框架：(1) 用 RL 训练 Fan-Out Language Model（FOLM），通过复合集级奖励优化查询生成；(2) 用训练好的 FOLM 合成高质量监督数据；(3) 将行为蒸馏到轻量级 diffusion model（扩散模型），实现非自回归的单步推理。
- **Results (效果)**: 在 Polyvore（时尚）和 Music（音乐）两个基准上，R4T 一致性超越零样本和 Best-of-N 基线，多样性评分从 ~56 提升至 ~76.8（Polyvore），同时 **推理延迟降低 12-20 倍**（从 50s 降至 4.21s，batch=1024）。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

![R4T 三阶段框架概览](https://arxiv.org/html/2603.06397/x1.png)

R4T 的整体流程分为三个阶段：

- **输入**: 用户查询 $q$（如 "bohemian festival style outfit"）
- **Stage 1 - RL 策略优化**: 使用 GRPO 算法训练一个 Fan-Out Language Model（FOLM），使其能根据 $q$ 生成一组子查询 $\{q_1, q_2, ..., q_K\}$，每个子查询用于检索一个具体 item。训练目标是最大化集合级复合奖励。
- **Stage 2 - 数据合成**: 用训练好的 FOLM 对大量查询生成高质量子查询，形成监督数据集 $\{(q, \{q_k\})\}$。
- **Stage 3 - 扩散蒸馏**: 将 FOLM 的行为蒸馏到一个轻量级 diffusion retriever，将子查询映射为 embedding，实现非自回归单步推理。
- **输出**: 一组 item embeddings，用于从 item 库中检索最终结果。

### 2.2 Core Modules (核心模块详解)

**模块一：Fan-Out Task Formulation（扇出任务定义）**

- **功能**: 将集合级检索问题形式化为"扇出"任务——给定一个查询，生成 $K$ 个子查询，每个子查询负责检索集合中的一个 item
- **关键概念**: 定义了两种集级检索任务：

  **(a) Open-ended Abstract Retrieval (OAR，开放式抽象检索)**：查询是抽象的（如"波西米亚风穿搭"），没有标准答案。奖励函数综合考量三个维度：

$$\mathcal{R}_{abs} = \lambda_g \cdot r_{ground} + \lambda_d \cdot r_{div} + \lambda_a \cdot r_{align}$$

- $r_{ground}$: groundedness（基础性），衡量每个子查询是否能检索到真实存在的 item
- $r_{div}$: diversity（多样性），衡量检索结果集的多样性
- $r_{align}$: alignment（对齐性），衡量整个集合是否与原始查询的语义对齐
- $\lambda_g = 0.6, \lambda_d = \lambda_a = 0.2$: 权重分配，基础性占主导
- **直觉理解**: 这个奖励同时要求生成的子查询"能找到真实商品"（不是幻觉）、"找到的商品足够多样"且"整体风格与用户需求一致"

  **(b) Weakly-Supervised Compositional Retrieval (WSCR，弱监督组合检索)**：有参考集合，奖励鼓励检索结果覆盖参考集中的 item：

$$\mathcal{R}_{comp} = r_{cover}$$

- $r_{cover}$: reference set coverage reward（参考集覆盖奖励），衡量子查询检索到的 item 与参考集的重合度

---

**模块二：Soft-GRPO 训练算法**

- **功能**: 在标准 GRPO（Group Relative Policy Optimization）基础上增加 KL 散度惩罚，稳定开放式生成的训练过程
- **输入**: 查询 $q$，当前策略 $\pi_\theta$，参考策略 $\pi_{ref}$
- **输出**: 更新后的策略参数 $\theta$

**核心公式 - Soft-GRPO 目标函数**:

$$\mathcal{L}_{SoftGRPO}(\theta) = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i \hat{A}_i, \text{clip}(\rho_i, 1\pm\epsilon) \hat{A}_i \right) \right] + \beta_1 D_{KL}(\pi_\theta \| \pi_{ref}) + \beta_2 D_{KL}(\pi_{ref} \| \pi_\theta)$$

- $\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{ref}(o_i|q)}$: importance sampling ratio（重要性采样比）
- $\hat{A}_i$: normalized advantage（归一化优势函数），通过组内奖励标准化计算
- $\epsilon$: clipping 参数，防止策略更新过大
- $\beta_1, \beta_2 = 0.05$: 前向和反向 KL 惩罚系数
- $G$: 每个查询的采样组大小
- **直觉理解**: 标准 GRPO 在开放式任务中容易 mode collapse（模式坍塌），双向 KL 惩罚迫使策略既不能偏离参考策略太远（前向 KL），又不能变得过于保守/确定（反向 KL），从而在探索和稳定之间取得平衡。

---

**模块三：Diffusion Retriever（扩散检索器）**

- **功能**: 将 FOLM 的自回归行为蒸馏到非自回归的扩散模型，实现高效推理
- **输入**: 查询 embedding $e_q$，噪声样本 $z \sim \mathcal{N}(0, \sigma^2 I)$
- **输出**: $K$ 个 item embeddings $\{e_1, ..., e_K\}$

**训练损失 (Variance Exploding 框架)**:

$$\mathcal{L}_{diff} = \mathbb{E}_{\sigma, e_q, \epsilon} \left[ \| D_\phi(\hat{e} + \epsilon; \sigma, e_q) - \hat{e} \|_2^2 \right]$$

- $D_\phi$: 扩散模型的去噪网络（denoiser），参数为 $\phi$
- $\hat{e}$: FOLM 合成的目标 item embedding
- $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$: 高斯噪声
- $\sigma$: 噪声水平，训练时从对数均匀分布采样
- **直觉理解**: 扩散模型学习从"纯噪声"一步去噪到"有意义的 item embedding"，相当于把 FOLM 多步自回归的"思考过程"压缩成了单步映射。推理时只需一次前向传播即可生成所有子查询 embedding。

**架构细节**: 采用 EDM（Elucidating the Design Space of Diffusion Models）预处理方案，包含 skip connection 和噪声水平条件注入。

### 2.3 Training Strategy (训练策略)

- **Stage 1 RL 训练**:
  - 基座模型: Gemma-2B / Qwen-2.5-7B
  - 优化器: Soft-GRPO
  - 采样组大小 $G = 16$
  - KL 系数 $\beta_1 = \beta_2 = 0.05$
  - 奖励: LLM-as-a-Judge（用 Gemini Flash 评估 groundedness, diversity, alignment）

- **Stage 2 数据合成**:
  - 对每个查询，用 FOLM 生成多组子查询
  - 子查询通过 embedding 模型映射到向量空间
  - 形成 $(e_q, \{\hat{e}_k\})$ 训练对

- **Stage 3 扩散训练**:
  - 框架: Variance Exploding (VE) diffusion
  - 采用 EDM preconditioning
  - 推理时使用单步去噪（single-step denoising）

- **Loss 设计直觉**: 三阶段 pipeline 的核心思想是"RL 负责探索，扩散负责高效部署"——RL 训练虽然昂贵但能发现最优扇出行为，扩散蒸馏虽然损失一些质量但大幅降低推理成本。

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **数据集**:
  - **Polyvore**（时尚穿搭）: 来自 Polyvore 平台的服装搭配数据
  - **Music**（音乐推荐）: 音乐歌单推荐数据
- **评价指标**:
  - OAR 任务: Groundedness（基础性）、Diversity（多样性）、Alignment（对齐性）、Average Score（综合分）
  - WSCR 任务: Recall@5K、Hit@5K、Vendi Score（多样性指标）
- **主要 Baselines**:
  - Zero-shot LLM（Gemma-2B, Qwen-7B, Gemini-2.5-Flash）
  - Best-of-N sampling（采样 N 次取最优）
  - R4T-FOLM（直接部署 RL 模型）
  - R4T-Diffusion（扩散蒸馏版本）

### 3.2 Main Results (主实验结果)

**OAR 任务（Table 1）**:

| 方法 | Polyvore Diversity | Polyvore Alignment | Polyvore Avg | Music Diversity | Music Avg |
|------|---|---|---|---|---|
| Zero-shot (Gemma) | 56.0 | 31.2 | 38.5 | 51.8 | 48.1 |
| Best-of-N (Gemma) | 61.0±2.1 | 32.7±1.7 | 40.9±1.6 | - | - |
| **R4T-FOLM** | **76.8±2.7** | **39.8±2.3** | **49.1±2.1** | **62.0±2.0** | **58.1±2.2** |
| R4T-Diffusion | 74.3±3.4 | 37.6±2.8 | 47.2 | 59.6±2.6 | 56.3 |

关键发现：
- R4T-FOLM 在所有指标上取得最优，Polyvore 多样性从 56.0 → **76.8**（+37% 相对提升）
- R4T-Diffusion 仅比 FOLM 略低（Diversity 74.3 vs 76.8），但推理速度快 12-20 倍
- Best-of-N 虽然比零样本好（61.0 vs 56.0），但远不如 R4T（61.0 vs 76.8）

**WSCR 任务（Table 2）**:

| 方法 | Recall@5K | Hit@5K | Vendi Score |
|------|---|---|---|
| Gemini-2.5-Flash | 15.7 | 52.1 | 33.4 |
| **R4T-FOLM (Qwen)** | **20.9** | **64.6** | 27.5 |
| R4T-Diffusion (Qwen) | 16.5 | 57.5 | **34.7** |

关键发现：
- R4T-FOLM 的 Recall@5K 达到 20.9，超越 Gemini-2.5-Flash 的 15.7（+33%）
- R4T-Diffusion 在 Vendi Score（多样性）上最优（34.7），说明扩散模型的随机性天然带来多样性优势

### 3.3 Ablation Study (消融实验)

![OAR 奖励分析](https://arxiv.org/html/2603.06397/x2.png)

**奖励组分消融**:
- 仅使用 groundedness 奖励时出现**病态解**：模型会生成重复的无意义查询（如反复输出 "line ending"），因为这些查询碰巧能匹配到一些 item
- 仅使用 diversity 奖励时，生成的子查询与原始查询语义偏离
- 三者平衡组合（$\lambda_g=0.6, \lambda_d=\lambda_a=0.2$）时性能最优且训练最稳定

**推理效率对比**:

![推理效率对比](https://arxiv.org/html/2603.06397/x3.png)

| Batch Size | LLM (FOLM) 延迟 | Diffusion 延迟 | 加速比 |
|---|---|---|---|
| 8 | 1.46s | 0.07s | **~20x** |
| 1024 | 50s | 4.21s | **~12x** |

扩散模型的优势在大 batch 下更明显——得益于非自回归架构的高并行度。

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **"RL 探索 + 扩散部署"的 pipeline 设计很巧妙**: 将 RL 的探索能力和扩散模型的推理效率完美结合。RL 解决了"如何发现好的扇出策略"的难题，扩散模型解决了"如何高效部署"的难题。这种"train expensive, deploy cheap"的思路值得借鉴。

2. **集合级奖励设计**: 将集合检索的多目标需求（相关性、多样性、对齐性）显式建模为复合奖励，比端到端黑箱训练更可控、更可解释。

3. **Soft-GRPO 的双向 KL 正则化**: 针对开放式生成任务中 RL 训练不稳定的问题，同时加入前向和反向 KL 惩罚，这个设计对其他需要 RL 微调 LLM 的任务也有参考价值。

4. **实际部署友好**: 最终部署的扩散模型是轻量级的，推理延迟在毫秒级别，非常适合在线推荐场景。

### 4.2 Limitations & Improvements (不足与改进方向)

1. **RL 训练成本高**: Stage 1 的 RL 训练仍然需要大量计算资源，尤其是需要 LLM-as-a-Judge 评估每个采样的奖励。可以考虑用更轻量的奖励模型替代 LLM 评委。

2. **奖励函数设计依赖人工**: 三个奖励维度的权重 $\lambda$ 是手动设定的（0.6/0.2/0.2），可以探索自适应权重调整策略（如 multi-objective RL）。

3. **评估范围有限**: 仅在时尚和音乐两个领域验证，缺乏对更广泛场景（如电商通用推荐、旅游行程规划）的实验。

4. **扩散蒸馏的质量损失**: R4T-Diffusion 相比 R4T-FOLM 在所有指标上都有一定下降（如 Diversity 74.3 vs 76.8），可以探索更好的蒸馏策略（如 progressive distillation 或 consistency model）来缩小差距。

5. **LLM-as-a-Judge 偏差**: 使用 Gemini Flash 作为评委打分，可能引入模型特有的偏差。未来可以结合人类评估或设计更客观的自动化指标。
