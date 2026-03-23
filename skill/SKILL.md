---
name: paper-notes
description: >-
  Use this skill whenever the user wants to read, analyze, or take notes on an academic paper.
  This includes: providing an arXiv link (e.g., arxiv.org/abs/xxxx), a local PDF file path,
  or a paper title for lookup. Trigger when user mentions "读论文", "论文笔记", "paper notes",
  "读一下这篇", "summarize this paper", "read this paper", or provides any academic paper
  URL or PDF path and wants structured reading notes. Also trigger when the user asks to
  analyze methodology, experiments, or contributions of a paper. Especially useful for
  recommendation system and LLM papers, but works for any ML/AI domain.
---

# Paper Reading Notes (论文阅读笔记)

Generate structured reading notes from academic papers. Supports arXiv links, local PDFs, and paper title search.

## Input Processing

Determine the input type and gather paper content accordingly:

### Case 1: arXiv Link
If the user provides an arXiv URL (matching `arxiv.org/abs/` or `arxiv.org/pdf/`):
1. Extract the paper ID from the URL
2. Use `WebFetch` on `https://arxiv.org/abs/<id>` to get metadata (title, authors, abstract, venue)
3. Use `WebFetch` on `https://arxiv.org/html/<id>` to get the full paper text. **Also extract image URLs** from `<img>` tags for use in the note.
4. If HTML version fails, use `WebFetch` on `https://arxiv.org/pdf/<id>` as fallback
5. If both fail, ask the user to provide a local PDF file

### Case 2: Local PDF
If the user provides a file path ending in `.pdf`:
1. Use `Read` tool to read the PDF
2. For papers exceeding 20 pages, read in chunks: pages 1-20, then 21-40, etc.
3. Concatenate all chunks before proceeding

### Case 3: Paper Title
If the user provides a paper title (no URL, no file path):
1. Use `WebSearch` to search for `<title> arxiv`
2. Find the arXiv link from search results
3. Confirm with the user: "找到了这篇论文: <title>, <url>，是这篇吗？"
4. If user confirms, follow Case 1 flow
5. If user says no, ask for a more specific title or direct arXiv link

**IMPORTANT:** Gather ALL paper content before generating any notes. Do not write notes while still reading.

## Output

### Before Writing
After reading the paper, briefly inform the user in the terminal:
> "论文主题: <1-sentence summary>，正在生成阅读笔记..."

### File Path
- Base directory: `/Users/epoch/paper/`
- Structure: `/Users/epoch/paper/YYYY/YYMM-<方法名>-<机构>-<领域>.md`
- `YYYY` = 论文发表的年份（用于子目录）
- `YYMM` = 论文发表的年月（两位年份+两位月份），从论文中提取（arXiv 首次提交日期、会议/期刊发表日期）
- `<方法名>` = 论文提出的方法/模型名称，保留原始大小写或用小写，如 `DualGR`, `LLaMA`
- `<机构>` = 第一作者所属机构的中文简称，如 `快手`, `腾讯`, `字节`, `微软`, `清华`
- `<领域>` = 论文所属研究方向的中文简称，如 `生成式推荐`, `大模型`, `序列推荐`, `多模态`
- 示例: `/Users/epoch/paper/2025/2511-DualGR-快手-生成式推荐.md`
- Use `Bash` to run `mkdir -p /Users/epoch/paper/<YYYY>/` before writing

### After Writing
Display the file path to the user:
> "笔记已保存到: `/Users/epoch/paper/<YYYY>/<YYMM>-<方法名>-<机构>-<领域>.md`"

If paper content was incomplete (PDF truncation, missing sections), inform the user which parts may be missing.

## Note Template

Generate the note following this structure exactly — every section is required, but scale the depth of each section to the paper's actual content.

```markdown
# <Paper Title>

> **Authors**: <author list>
> **Affiliation**: <institutions> （注明所属公司/高校，如 Kuaishou / Tencent / ByteDance / Microsoft 等）
> **Venue**: <conference/journal, year> (or "arXiv preprint" if not published)
> **Links**: [arXiv](<url>) | [Code](<url, if available>)

---

## 1. Quick Overview (快速概述)

用 3-5 句通俗的话概括这篇论文：
- **Problem (问题)**: 这篇工作要解决什么问题？为什么这个问题重要？
- **Method (方法)**: 核心思路/方法是什么？（一句话概括）
- **Results (效果)**: 在什么数据集/任务上，相比 baseline 提升了多少？

## 2. Detailed Methodology (详细方法解读)

### 2.1 Overall Architecture (整体架构)
<!-- 描述模型/方法的整体流程：输入 → 主要模块 → 输出 -->
<!-- 在此处插入论文的模型 overview 图（通常是 Figure 1） -->
- 输入是什么（数据形式、维度）
- 经过哪些主要模块
- 最终输出是什么

### 2.2 Core Modules (核心模块详解)

<!-- 对每个关键模块，按以下格式说明 -->

**模块名称**

- **功能**: 这个模块做什么
- **输入**: 数据的形式和维度
- **输出**: 输出的形式和维度
- **关键公式**:

$$公式$$

- $符号_1$: 含义
- $符号_2$: 含义
- **直觉理解**: 用白话解释这个公式在做什么

<!-- 重复以上模块格式，直到所有核心模块都说明完毕 -->

### 2.3 Training Strategy (训练策略)
- **Loss Function**:

$$\mathcal{L} = ...$$

- 各项含义解释
- **直觉理解**: 这个 loss 为什么这样设计？优化目标是什么？
- 训练技巧（如有）: learning rate schedule, data augmentation, pre-training 策略等

## 3. Experiment Analysis (实验结果解读)

### 3.1 Experimental Setup (实验设置)
- **数据集**: 列出使用的数据集及其规模
- **评价指标**: 列出使用的 metrics
- **主要 Baselines**: 列出对比方法

### 3.2 Main Results (主实验结果)
逐表/逐图解读关键实验结果：
- 本方法在哪些指标上取得最优？
- 相比最强 baseline 提升幅度如何？
- 在不同数据集上表现是否一致？

### 3.3 Ablation Study (消融实验)
解读消融实验结果：
- 去掉/替换每个模块后性能变化如何？
- 哪个设计选择贡献最大？
- 有什么反直觉的发现？

## 4. Summary & Reflections (总结与思考)

### 4.1 Strengths (值得借鉴的地方)
<!-- 列出本文的亮点、创新点、好的设计思路 -->

### 4.2 Limitations & Improvements (不足与改进方向)
<!-- 指出局限性，并给出具体的改进思路 -->
```

## Behavior Rules

### Language Style
- **中英混合**: 中文叙述为主，专业术语保留英文
- 英文术语首次出现时附中文注释，如: "multi-head attention（多头注意力）"
- 后续出现可直接用英文术语
- 公式中的符号用英文，解释用中文

### Depth Strategy
- **快速概述**: 通俗易懂，假设读者有基本 ML 背景但不熟悉该特定领域
- **详细方法**: 专业深入，公式逐项拆解，每个符号都要解释
- **实验结果**: 数据驱动，引用表中具体数字，如 "在 MovieLens-1M 上 NDCG@10 提升了 3.2%"

### Formula Handling
- 所有关键公式用 LaTeX 格式（`$$...$$`）
- 列出每个符号的含义
- 附"直觉理解"用白话解释公式在做什么
- 示例:

$$\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in D} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) + \lambda \|\Theta\|^2$$

- $u, i, j$: 分别表示 user, positive item, negative item
- $\hat{r}_{ui}$: 模型预测的 user $u$ 对 item $i$ 的评分
- $\sigma$: sigmoid 函数
- $\lambda \|\Theta\|^2$: L2 正则化项，防止过拟合
- **直觉理解**: BPR loss 通过 pairwise comparison，鼓励模型对用户交互过的 item 给出比未交互 item 更高的分数

### Domain Adaptation
- **推荐系统论文**: 重点关注 user/item representation、交互建模方式、loss 设计、negative sampling 策略；实验部分重点解读对比表格和消融实验
- **大模型论文**: 重点关注 architecture 设计、scaling 特性、training recipe、data mixture；实验部分重点解读 benchmark 评测和 emergent ability
- **其他领域**: 使用通用模板，不做领域特定强调

### Content Rules
1. 忠实原文内容，不臆造或过度解读
2. 不确定的内容标注 `[待确认]`
3. 如论文有明显的 related work 创新对比，在"快速概述"中简要提及
4. 不要复制粘贴论文原文，用自己的话重新组织

### Figure References (论文图片引用)
在笔记中适当引用论文原文的图片，帮助读者理解：
- **整体架构图**: 在 "2.1 Overall Architecture" 中引用模型 overview 图（通常是 Figure 1）
- **模块细节图**: 在 "2.2 Core Modules" 中如有对应的模块结构图，一并引用
- **实验结果图**: 在 "3. Experiment Analysis" 中引用关键的 case study 图、可视化图、参数敏感性分析图等

引用方式：使用 arXiv HTML 版的图片 URL，格式为：
```
![<Figure caption 简要描述>](https://arxiv.org/html/<paper-id>/<image-filename>)
```
- 从 arXiv HTML 页面（`arxiv.org/html/<id>`）中获取图片的实际文件名（通常为 `x1.png`, `x2.png` 等，或 `extracted/figures/fig1.png` 等路径）
- 使用 `WebFetch` 抓取 HTML 页面时，注意提取 `<img>` 标签中的 `src` 属性以获取图片路径
- 如果 HTML 版不可用导致无法获取图片 URL，则用文字描述代替：`> 📊 **Figure X**: <对图片内容的文字描述>`
- 不要引用所有图片，只引用对理解方法和结果最有帮助的关键图
