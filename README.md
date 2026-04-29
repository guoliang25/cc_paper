# cc_paper

论文阅读笔记，涵盖推荐系统、大模型等 AI 领域的前沿工作。

## 论文列表

| 日期 | 方法 | 机构 | 领域 | 论文标题 | 笔记 |
|------|------|------|------|----------|------|
| 2019.02 | Adapter | Google Research | 参数高效微调 | [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) | [笔记](paper/2019/1902-Adapter-Google-参数高效微调.md) |
| 2023.03 | MoRec | 西湖大学 | 模态vsID推荐 | [Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited](https://arxiv.org/abs/2303.13835) | [笔记](paper/2023/2303-MoRec-西湖大学-模态vsID推荐.md) |
| 2023.05 | TIGER | Google DeepMind | 生成式推荐 | [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) | [笔记](paper/2023/2305-TIGER-Google-生成式推荐.md) |
| 2024.09 | HLLM | 字节跳动 | LLM序列推荐 | [HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling](https://arxiv.org/abs/2409.12740) | [笔记](paper/2024/2409-HLLM-字节-LLM序列推荐.md) |
| 2025.05 | GatedAttention | Qwen Team / 阿里 | 注意力机制 | [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) | [笔记](paper/2025/2505-GatedAttention-Qwen-注意力机制.md) |
| 2025.05 | LONGER | 字节跳动 | 长序列建模 | [LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://arxiv.org/abs/2505.04421) | [笔记](paper/2025/2505-LONGER-字节-长序列建模.md) |
| 2025.07 | RankMixer | 字节跳动 | 推荐系统Scaling | [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551) | [笔记](paper/2025/2507-RankMixer-字节-推荐系统Scaling.md) |
| 2025.10 | OneTrans | 字节跳动 / NTU | 统一排序架构 | [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104) | [笔记](paper/2025/2510-OneTrans-字节-统一排序架构.md) |
| 2025.11 | MaskGR | Snap / UT Austin | 生成式推荐 | [Masked Diffusion for Generative Recommendation](https://arxiv.org/abs/2511.23021) | [笔记](paper/2025/2511-MaskGR-Snap-生成式推荐.md) |
| 2025.12 | EnsRec | Snap | 序列推荐 | [Exploiting ID-Text Complementarity via Ensembling for Sequential Recommendation](https://arxiv.org/abs/2512.17820) | [笔记](paper/2025/2512-EnsRec-Snap-序列推荐.md) |
| 2026.02 | MixFormer | 字节跳动 | 推荐系统Scaling | [MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110) | [笔记](paper/2026/2602-MixFormer-字节-推荐系统Scaling.md) |
| 2026.03 | IDProxy | 小红书 | 冷启动CTR | [IDProxy: Cold-Start CTR Prediction for Ads and Recommendation at Xiaohongshu with Multimodal LLMs](https://arxiv.org/abs/2603.01590) | [笔记](paper/2026/2603-IDProxy-小红书-冷启动CTR.md) |
| 2026.03 | R4T | UIUC / Google DeepMind | 集合检索 | [Efficient, Property-Aligned Fan-Out Retrieval via RL-Compiled Diffusion](https://arxiv.org/abs/2603.06397) | [笔记](paper/2026/2603-R4T-UIUC&Google-集合检索.md) |
| 2026.04 | UniMixer | 快手 | 推荐系统Scaling | [UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems](https://arxiv.org/abs/2604.00590) | [笔记](paper/2026/2604-UniMixer-快手-推荐系统Scaling.md) |
| 2026.04 | SID-Coord | 快手 | 短视频搜索Ranking | [SID-Coord: Coordinating Semantic IDs for ID-based Ranking in Short-Video Search](https://arxiv.org/abs/2604.10471) | [笔记](paper/2026/2604-SIDCoord-快手-短视频搜索Ranking.md) |
| 2026.04 | R³-VAE | 字节跳动 | 生成式推荐SID | [R³-VAE: Reference Vector-Guided Rating Residual Quantization VAE for Generative Recommendation](https://arxiv.org/abs/2604.11440) | [笔记](paper/2026/2604-R3VAE-字节-生成式推荐SID.md) |

## 专题对比

| 主题 | 覆盖论文 | 笔记 |
|------|---------|------|
| 字节四大工业 Ranking 架构 | RankMixer / LONGER / OneTrans / MixFormer | [对比笔记](paper/对比/字节四大Ranking架构对比-RankMixer-LONGER-OneTrans-MixFormer.md) |
| RankMixer 的 Token 数 T 与 Scaling Law 的斜率/截距 | RankMixer / LONGER / OneTrans / MixFormer | [HTML 版（含数学公式 + 论文原图）](docs/rankmixer-T-and-scaling-slope-intercept.html) · [在线访问](https://guoliang25.github.io/cc_paper/rankmixer-T-and-scaling-slope-intercept.html) |

## 在线专题站

GitHub Pages 地址：<https://guoliang25.github.io/cc_paper/>
