---
title: 'Placeholder Note 1: \(L(N)=\alpha N^{-\beta}\)'
summary: 'A longer placeholder summary for checking how the blog card handles full-width prose, line wrapping, tag alignment, and inline math such as \(\mathcal{L}(\theta)=\mathbb{E}[\ell(f_\theta(x), y)]\). This should now fill most of the card width before wrapping.'
date: 2026-07-07
authors:
  - me
tags:
  - Placeholder
  - Notes
---

这是一篇用于测试博客正文排版的占位文章。它不是正式内容，主要用来观察标题层级、段落宽度、数学公式、代码块、表格、引用、列表、脚注和链接在当前主题下的显示效果。

正文里可以混合普通叙述和行内元素，比如行内代码 `policy.update()`、行内链接 [Hugo](https://gohugo.io/)，以及行内数学公式 \(L(N)=\alpha N^{-\beta}\)。如果这些元素在同一段里看起来不突兀，说明基础排版大体是稳的。

## 1. 段落与层级

这一节测试较长段落的换行效果。一个好的博客正文页面应该让读者能自然扫读：每行不能太长，段落之间要有明确间距，标题也应该能帮助读者快速定位内容。这里故意写得稍微长一点，用来观察中文、英文缩写、公式和标点混排时是否会出现奇怪的行高或间距问题。

### 小标题

小标题应该比二级标题更轻，不抢正文的层级。比如在一篇技术笔记里，二级标题可以表示一个大问题，三级标题可以表示某个局部观察。

## 2. 数学公式

行内公式测试：当模型规模 \(N\)、数据规模 \(D\)、训练计算量 \(C\) 同时变化时，我们希望公式和正文保持相近的视觉高度，而不是突然大一圈。

独立公式测试：

\[
\mathcal{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
\]

再来一个多行推导，看看上下间距是否合适：

\[
\begin{aligned}
C &\approx 6ND, \\
D_\star(N) &\propto N^{\alpha / \beta}, \\
\mathcal{L}_\star(C) &\propto C^{-\frac{\alpha\beta}{\alpha+\beta}}.
\end{aligned}
\]

## 3. 列表

无序列表测试：

- 先写一个核心问题。
- 再记录一个实验观察。
- 最后补一个后续行动。

有序列表测试：

1. 读论文，先看摘要和图。
2. 复现关键公式，检查变量定义。
3. 写一段自己的解释，避免只复制原文。

任务列表测试：

- [x] 检查行内公式大小。
- [x] 检查正文卡片样式。
- [ ] 替换占位内容为正式文章。

## 4. 表格

| Feature | Purpose | Current Check |
| --- | --- | --- |
| Math | Test inline and display equations | Looks close to body text |
| Code | Test monospace rendering | Should be compact and readable |
| Table | Test dense structured text | Should not feel cramped |
| Quote | Test highlighted prose | Should be visible but calm |

## 5. 代码块

下面是一段 Python 风格的伪代码，用来测试代码块的字体、背景和横向滚动。

```python
def estimate_loss(model_size, data_size, alpha=0.35, beta=0.28):
    irreducible_error = 1.69
    model_term = model_size ** (-alpha)
    data_term = data_size ** (-beta)
    return irreducible_error + model_term + data_term


loss = estimate_loss(model_size=1e9, data_size=3e11)
print(f"estimated loss: {loss:.4f}")
```

也可以测试一段 shell：

```bash
hugo --minify
curl -I http://127.0.0.1:1314/blog/
```

## 6. 引用与脚注

> A useful research note should be clear enough to reread, but compact enough to actually maintain.

上面这段引用用来测试 blockquote 的缩进、颜色和边距。脚注则适合放一些不想打断正文的小解释。[^note]

[^note]: 这是一个脚注测试，用来观察文章底部脚注区域的字号、间距和链接跳转。

## 7. 小结

如果这一页的标题、正文、公式、代码、表格、引用和脚注都看起来协调，那么之后写正式博客时就可以放心沿用这套样式。下一步可以再加一篇真正的短笔记，测试更接近实际写作的阅读体验。
