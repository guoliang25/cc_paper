# IDProxy: Cold-Start CTR Prediction for Ads and Recommendation at Xiaohongshu with Multimodal LLMs

> **Authors**: Yubin Zhang, Haiming Xu, Guillaume Salha-Galvan, Ruiyan Han, Feiyang Xiao, Yanhua Huang, Li Lin, Yang Luo, Yao Hu
> **Affiliation**: Xiaohongshu Inc.（小红书）/ Shanghai Jiao Tong University（上交）/ Fudan University（复旦）
> **Venue**: arXiv preprint, March 2026
> **Links**: [arXiv](https://arxiv.org/abs/2603.01590)

---

## 1. Quick Overview (快速概述)

- **Problem (问题)**: 推荐系统和广告中的 item cold-start（冷启动）问题——新内容缺乏历史交互数据，其 ID embedding 未经充分训练，导致 CTR 预估不准确、新内容曝光不足。传统方法要么直接用内容特征（语义与协同过滤空间存在 gap），要么用简单映射（无法捕捉工业级 embedding 的复杂分布）。小红书作为拥有 3 亿+ 用户的内容平台，新笔记/广告的冷启动问题尤为突出。
- **Method (方法)**: 提出 **IDProxy**，利用多模态大语言模型（MLLM, 具体使用 InternVL）从 item 的图文内容中生成**代理 embedding（proxy embedding）**，替代未充分训练的 ID embedding。采用两阶段 coarse-to-fine alignment（从粗到细对齐）：Stage 1 通过对比学习将 MLLM 的内容表示与 ID embedding 空间粗对齐；Stage 2 提取 MLLM 多层隐状态，通过轻量级 multi-granularity adaptor + residual gating 进行 CTR-aware 的细粒度对齐。
- **Results (效果)**: 离线 AUC 相比 baseline 提升 +0.14%；线上 A/B 测试中，内容 Feed 阅读量 +0.22%、互动 +0.5%，展示广告曝光 +1.28%、广告主价值 +1.93%、CTR +0.23%。新内容的提升幅度约为全量流量的 2 倍。已部署于小红书"发现"Feed 和展示广告。

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)

![IDProxy 两阶段框架概览](https://arxiv.org/html/2603.01590v1/x2.png)

IDProxy 的整体流程分为两个阶段：

- **输入**: item 的多模态内容信号（图片 + 文本，如小红书笔记的封面图和标题/正文）
- **Stage 1 — Coarse Proxy Generation（粗粒度代理生成）**: 用 MLLM 编码内容特征，通过 attention pooling 聚合为内容 embedding，再用对比学习与已有 ID embedding 空间对齐
- **Stage 2 — CTR-Aware Fine Refinement（CTR 感知的细粒度精调）**: 从 MLLM 多层隐状态中提取层次化表示，通过轻量级 adaptor + residual gating 精调，与 CTR ranking model 联合端到端训练
- **部署时**: proxy embedding 离线预计算并存储，serving 时通过 item ID 实时检索替换

### 2.2 Core Modules (核心模块详解)

**模块一：ID Embedding 预处理**

- **功能**: 为对比学习准备高质量的对齐目标
- **频率阈值过滤**: 仅保留交互次数 ≥ $\tau$ 的 item 的 ID embedding 作为对齐目标，过滤掉训练不充分的 embedding
- **L2 归一化**:

$$\mathbf{e}_i = \frac{\mathbf{e}_i^{raw}}{\|\mathbf{e}_i^{raw}\|_2} \in \mathbb{R}^d$$

- **直觉理解**: 工业级 ID embedding 的分布很不规则（不像学术数据集那样有清晰的聚类结构），归一化将所有 embedding 投影到单位超球面上，为对比学习提供更稳定的优化空间

![ID embedding 分布对比：学术数据集 vs 工业场景](https://arxiv.org/html/2603.01590v1/x1.png)

**模块二：Stage 1 — MLLM-based Coarse Proxy（基于 MLLM 的粗粒度代理）**

- **功能**: 将 item 的多模态内容映射到 ID embedding 空间的邻域
- **MLLM 编码**: 使用 InternVL 对 item 的图文内容进行多模态编码，得到 hidden states $\mathbf{H}_i$
- **Attention Pooling 聚合**: 通过注意力机制将多个 token 的 hidden states 聚合为单一向量 $\mathbf{z}_i = g(\mathbf{H}_i)$
- **K-means 层选择**: MLLM 有数十层，通过 k-means 聚类将层分为 3 组 $\{l_{n_1}, l_{n_2}, l_{n_3}\}$，分别代表浅层、中层、深层特征，从每组中选择代表层
- **投影与归一化**:

$$\tilde{h}_i = \frac{\phi(\mathbf{z}_i)}{\|\phi(\mathbf{z}_i)\|_2} \in \mathbb{R}^d$$

- $\phi$: 投影 MLP，将 MLLM 的表示维度映射到 ID embedding 维度 $d$

- **Proxy Alignment Loss（代理对齐损失）**:

$$\mathcal{L}_{PAL} = -\frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \log \frac{\exp(\tilde{h}_i^T \mathbf{e}_i / \tau_c)}{\sum_{j \in \mathcal{B}} \exp(\tilde{h}_i^T \mathbf{e}_j / \tau_c)}$$

- $\tilde{h}_i$: item $i$ 的 coarse proxy
- $\mathbf{e}_i$: item $i$ 的归一化 ID embedding（对齐目标）
- $\tau_c > 0$: 温度参数
- $\mathcal{B}$: mini-batch
- **直觉理解**: 标准的 InfoNCE 对比学习——拉近同一 item 的内容表示与 ID embedding，推远不同 item 的表示。这一步让 MLLM 的语义空间"大致对准"协同过滤的 ID 空间

**模块三：Stage 2 — Multi-granularity Adaptor + Residual Gating（多粒度适配器 + 残差门控）**

- **功能**: 在 Stage 1 粗对齐的基础上进行 CTR-aware 的细粒度精调
- **多粒度特征提取**: 从 MLLM 的三个层次子组提取 hidden states，通过 k-means 聚类选择的代表层捕获从浅层（低级特征）到深层（高级语义）的层次化信息

- **轻量级 Adaptor**:

$$\mathbf{p}_i^{raw\_fine} = \tilde{\phi}(\text{Concat}(\mathbf{z}_i^{(l_{n_1})}, \mathbf{z}_i^{(l_{n_2})}, \mathbf{z}_i^{(l_{n_3})})) \in \mathbb{R}^{\tilde{d}}$$

- $\tilde{\phi}$: 轻量级 MLP adaptor
- 拼接三个层次的表示，融合多粒度信息
- 参数量远小于 MLLM 和 ranking model

- **Residual Gating（残差门控）**:

$$\mathbf{r} = \sigma(\mathbf{W}_g [\mathbf{p}_i^{coarse}, \mathbf{p}_i^{raw\_fine}])$$

$$\mathbf{p}_i^{fine} = \mathbf{W}_c \mathbf{p}_i^{coarse} + \mathbf{r} \odot \mathbf{p}_i^{raw\_fine}$$

- $\sigma$: sigmoid 激活
- $\mathbf{r}$: 门控向量，控制 fine-grained 信号的贡献
- $\odot$: 逐元素乘法
- $\mathbf{W}_c$: coarse proxy 的线性变换
- **直觉理解**: 门控机制让模型自动决定"在 Stage 1 粗对齐的基础上，Stage 2 的细粒度信息应该补充多少"——如果粗对齐已经够好，门控会抑制细粒度信号（避免冗余）；如果粗对齐不够精确，门控会放大补充信号

- **CTR Loss 联合训练**:

$$\mathcal{L}_{CTR} = -\frac{1}{|\mathcal{D}|} \sum_{(u,i,x_{ui},y_{ui}) \in \mathcal{D}} \left[y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui})\right]$$

- Stage 2 将 adaptor 参数与 CTR ranking model 联合端到端训练，MLLM 参数冻结
- **直觉理解**: 让 proxy embedding 不仅在空间上与 ID embedding 对齐，更要在下游 CTR 预估任务上有效——"不只要像，还要有用"

**Structure Reuse（结构复用）**:
- Stage 2 复用 Stage 1 中 attention pooling 的结构和参数作为初始化
- 从 Table 1 看，structure reuse 带来 +0.06% AUC 的额外提升（v4 → v5），说明 Stage 1 学到的表示结构对 Stage 2 的精调很有价值

### 2.3 Deployment Strategy (部署策略)

![系统架构](https://arxiv.org/html/2603.01590v1/x3.png)

- **离线计算**: 新 item 上线时，通过 IDProxy 生成服务离线计算 proxy embedding 并存入在线存储
- **实时 serving**: ranking model serving 时通过 item ID 检索 proxy embedding，替代冷启动 item 未充分训练的 ID embedding
- **对生产架构零侵入**: 不修改 ranking model 的架构，仅替换输入的 ID embedding，降低部署风险
- **轻量级 adaptor 参数与 MLLM 打包为 IDProxy 生成服务**，与 ranking model 解耦

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)

- **平台**: 小红书（RedNote），3 亿+ 用户的内容社区 + 电商平台
- **场景**: 内容 Feed（发现页推荐）和展示广告
- **评价指标**: AUC（离线）；阅读量、互动量、时长、曝光、CTR、广告主价值（线上）
- **MLLM**: InternVL
- **Baselines**: 生产 baseline、Notellm2-Like Embed（类似文献中的方法）、Static Vector MLP Mapping

### 3.2 Offline Results (离线结果)

**Table 1: 离线 CTR 预估 AUC 增益**:

| 模型变体 | ΔAUC |
|---------|------|
| Base（生产 baseline） | 0 |
| + Notellm2-Like Embed（v1） | +0.015% |
| + Static Vector MLP Mapping（v2） | +0.02% |
| + **IDProxy Stage 1**（v3） | **+0.05%** |
| + **IDProxy Stage 1+2, w/o Structure Reuse**（v4） | **+0.08%** |
| + **IDProxy Stage 1+2, w/ Structure Reuse**（v5） | **+0.14%** |

关键观察：
- 简单的多模态 embedding 方法（v1, v2）提升有限（+0.015% ~ +0.02%），说明直接映射内容特征到 ID 空间是不够的
- Stage 1 的对比学习对齐带来 +0.05%，比简单方法高出 2-3 倍
- Stage 2 的 CTR-aware 精调在此基础上再翻倍（+0.08%），验证了"不只要像，还要有用"的设计理念
- Structure Reuse 再贡献 +0.06%（从 +0.08% 到 +0.14%），说明两阶段之间的知识传递很重要
- **总体 +0.14% AUC** 在工业级推荐系统中是显著的提升

### 3.3 Online Results (线上实验)

**Table 2: 线上 AUC 增益（按内容年龄分层）**:

| 指标 | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 |
|------|-------|-------|-------|-------|-------|
| 全量笔记 | +0.13% | +0.15% | +0.14% | +0.12% | +0.15% |
| **新笔记** | **+0.24%** | **+0.32%** | **+0.23%** | **+0.27%** | **+0.31%** |

→ 新内容的 AUC 提升约为全量流量的 **2 倍**，精确验证了 IDProxy 解决冷启动问题的有效性

**Table 3: 线上 A/B 测试业务指标**:

| 场景 | 指标 | 提升 |
|------|------|------|
| **内容 Feed** | 阅读量 (Reads) | +0.22% |
| | 互动量 (Engagements) | +0.39% ~ +0.5% |
| | 用户时长 (Time Spent) | 正向 |
| **展示广告** | 曝光量 (Impressions) | +1.28% |
| | 广告主价值 (ADVV) | **+1.93%** |
| | 花费 (COST) | +1.73% |
| | CTR | +0.23% |

关键观察：
- **广告场景收益更大**: 广告主价值 +1.93%、曝光 +1.28%，这是因为广告场景中新素材更新更频繁，冷启动问题更严重
- 内容 Feed 互动量 +0.5% 说明更准确的 CTR 预估让用户看到了更感兴趣的新内容
- 部署在小红书这样的大规模平台上，这些提升有巨大的商业价值

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)

1. **工业问题意识清晰**: 论文从工业实际出发，发现学术数据集上 ID embedding 分布有清晰聚类（t-SNE 可视化），但工业数据分布不规则——这直接驱动了 coarse-to-fine 两阶段设计，而非一步到位的简单映射
2. **MLLM 的务实使用方式**: 没有让 MLLM 参与线上推理（太慢），而是离线生成 proxy embedding 存储后在线检索，对生产架构零侵入。这种"MLLM 做离线预计算"的模式在工业界极具参考价值
3. **Residual Gating 的设计巧思**: 门控机制让 Stage 2 的细粒度信号自适应补充 Stage 1 的粗粒度对齐，避免了冗余信息干扰，实验证明了其有效性
4. **多粒度层次化特征提取**: 通过 k-means 聚类从 MLLM 的数十层中选择代表层，而非简单取最后一层，充分利用了 MLLM 不同层次的信息（浅层的视觉纹理 → 深层的语义理解）
5. **按内容年龄分层的评估设计**: Table 2 按 Day 1-5 分层展示 AUC 增益，清晰证明了"越新的内容受益越大"，这比仅报告全量指标更有说服力

### 4.2 Limitations & Improvements (不足与改进方向)

1. **仅关注 item 冷启动**: 论文只解决了 item 侧的冷启动，未涉及 user 冷启动（新用户同样缺乏交互历史），MLLM 可以从用户画像生成 user proxy 的方向值得探索
2. **Proxy 的时效性问题**: proxy embedding 离线预计算后是静态的，但 item 的 ID embedding 随着交互积累会持续更新。item 从冷启动过渡到热门后，何时从 proxy 切换到真实 ID embedding？论文未讨论这个过渡策略
3. **对 MLLM 的依赖**: 使用 InternVL 这样的重量级 MLLM 做内容编码，即使是离线计算，在 item 量级很大时（小红书日均新笔记可达百万级）计算成本也不容忽视
4. **消融实验可以更丰富**: 缺少对 k-means 层选择策略、频率阈值 $\tau$ 的敏感性分析，以及不同 MLLM backbone 的对比
5. **可能的改进方向**:
   - 设计动态 proxy——随着 item 积累交互数据，逐步从 proxy embedding 过渡到真实 ID embedding（如加权融合）
   - 探索更轻量的 MLLM（如 MiniCPM-V）替代 InternVL，在内容编码质量和计算成本间寻找更优平衡
   - 将 IDProxy 的思路扩展到检索阶段（而非仅在 ranking 阶段），用 proxy embedding 改善新 item 的召回率
   - 结合 TIGER 的 Semantic ID 思路，用 MLLM 生成的 proxy 来初始化新 item 的 Semantic ID
