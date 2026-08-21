# 第三周新生入门任务学习记录

> 学习周期：2026.08.15 - 2026.08.21  
> 本 README 根据《2026级新生入门任务》第三周要求及当前 `git/week3` 目录整理。  
> 建议检查无误后，将本文件复制到 `git/week3/README.md`，再统一提交到远程仓库。

## 一、本周学习内容

本周围绕 AI 代码助手、Agent 自动化、NLP 基础、Transformer 与 Vision Transformer 展开，完成了文件批处理小程序、Word2Vec 实践、Transformer 核心模块笔记、GPT-2 在线可视化验证以及 ViT 论文精读笔记。


## 二、目录结构

```text
week3/
├── 文件批处理/
│   ├── file_organizer.py
│   ├── test_file_organizer.py
│   └── 文件批处理脚本说明.md
├── GPT2 demo.png
├── Transformer笔记.md
├── ViT论文精读学习笔记.md
├── Word2Vec.ipynb
└── 位置编码学习笔记.md
```

## 三、AI 辅助完成的文件批处理工具

### 1. 功能简介

`文件批处理/file_organizer.py` 根据扩展名将指定目录第一层的文件自动整理到不同分类目录中：

| 分类目录 | 文件类型示例 |
|---|---|
| `images` | `.jpg`、`.png`、`.gif`、`.webp` |
| `documents` | `.txt`、`.md`、`.pdf`、`.docx`、`.xlsx` |
| `code` | `.py`、`.ipynb`、`.js`、`.cpp` |
| `archives` | `.zip`、`.rar`、`.7z`、`.tar` |
| `media` | `.mp3`、`.wav`、`.mp4` |
| `others` | 未配置的其他扩展名 |

脚本默认使用安全的预览模式，不会立即移动文件；只有增加 `--execute` 参数才会正式执行。遇到同名文件时会自动添加序号，避免覆盖已有文件。

### 2. 使用方法

查看帮助：

```bash
python file_organizer.py --help
```

预览指定目录的整理结果：

```bash
python file_organizer.py "目标文件夹路径"
```

确认预览无误后正式整理：

```bash
python file_organizer.py "目标文件夹路径" --execute
```

运行自动化测试：

```bash
python -m unittest -v test_file_organizer.py
```

当前测试结果：4 项测试全部通过，包括扩展名大小写处理、预览模式、正式移动和同名文件防覆盖。

### 3. AI 助手的作用

AI 助手用于辅助完成需求拆分、代码结构设计、命令行参数设计、安全预览模式、异常处理和单元测试设计。使用 AI 生成代码后，仍需要人工检查分类规则、文件操作范围和测试结果，避免误移动或覆盖文件。

## 四、NLP 基础与 Word2Vec

`Word2Vec.ipynb` 用于学习和实践以下内容：

- 独热编码、Embedding 和词向量之间的关系；
- 为什么需要把离散词语映射成连续向量；
- Word2Vec 的 CBOW 与 Skip-gram 两种训练方式；
- 上下文窗口的作用；
- 词语相似度与语义关系；
- Word2Vec 的作用和局限。

当前 Notebook 包含 21 个代码单元，其中 20 个具有执行记录，并保留了相应输出结果。

## 五、Transformer 核心模块与 GPT-2 验证

### 1. Scaled Dot-Product Attention

Transformer 注意力的核心公式为：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- `Q`（Query）表示当前 token 想查找什么信息；
- `K`（Key）表示每个 token 可以用什么特征被匹配；
- `V`（Value）表示匹配后实际传递的内容；
- 除以 $\sqrt{d_k}$ 可以避免点积过大导致 Softmax 过于尖锐；
- Softmax 将相关性分数转换为权重，再对 `V` 进行加权求和。

### 2. Multi-Head Attention

多头注意力把输入映射到多个不同的表示子空间，让不同注意力头分别学习语法关系、位置关系、远距离依赖或其他模式。各个头的输出拼接后，再经过一次线性变换：

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(head_1,\ldots,head_h)W^O
$$

### 3. Position Encoding

Self-Attention 本身不包含顺序信息，因此需要加入位置编码，使模型能够区分不同 token 的先后位置。相关推导和 PyTorch 实现记录在：

- `Transformer笔记.md`
- `位置编码学习笔记.md`

### 4. GPT-2 在线可视化验证

使用 [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) 观察 GPT-2 如何完成：

```text
输入文本
→ Tokenization
→ Token Embedding + Position Embedding
→ 多层 Transformer Block
→ 下一个 token 的概率分布
→ 逐 token 生成文本
```

运行和学习结果截图保存在 `GPT2 demo.png`。

## 六、ViT 论文精读

阅读论文：[An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929)

### 1. 核心创新

ViT 将图像划分为固定大小的 Patch，把每个 Patch 映射为一个向量，然后像处理文本 token 一样，将 Patch token 序列输入标准 Transformer Encoder。

对于一张 `224×224` 图像，如果 Patch 大小为 `16×16`：

$$
N=\frac{224}{16}\times\frac{224}{16}=196
$$

与逐像素建模相比，Patch 显著缩短了 token 序列，使全局 Self-Attention 在图像任务中变得可行。

### 2. Patch Embedding

使用以下卷积可以一次完成“不重叠 Patch 提取 + 线性投影”：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

- `kernel_size=patch_size`：一次观察一个完整 Patch；
- `stride=patch_size`：相邻 Patch 不重叠；
- `out_channels=embed_dim`：把每个 Patch 映射为指定维度的 token。

### 3. Class Token 与 Position Embedding

- **Class Token**：放在 Patch 序列最前面，通过多层 Self-Attention 汇总整张图像的信息，最终用于分类。
- **Position Embedding**：标识不同 Patch 的空间位置；复杂的邻接关系、方向关系和物体结构仍需模型通过训练学习。

### 4. ViT 与 CNN 对比

| 对比项 | CNN | ViT |
|---|---|---|
| 核心操作 | 卷积 | Self-Attention |
| 感受野 | 从局部逐层扩大 | 可直接建立全局 Patch 关系 |
| 局部性 | 架构内置 | 主要从数据中学习 |
| 平移等变性 | 天然具备 | 不天然具备 |
| 小数据表现 | 通常样本效率更高 | 原始 ViT 更容易过拟合 |
| 大规模预训练 | 有效 | 尤其重要，能够弥补视觉先验较少的问题 |
| 图像细节 | 通过层次卷积逐步提取 | 受 Patch 大小影响，Patch 越小细节越丰富但计算越高 |

可以将二者的差异概括为：

> CNN 把局部性和平移等视觉规律写进架构；ViT 更多地依靠训练数据学习 Patch 之间的局部、全局和空间关系。

完整论文要点、模型架构图、核心公式和常见误区记录在 `ViT论文精读学习笔记.md`。

## 七、本周学习总结

通过本周任务，我完成了从传统词向量到 Transformer，再到 Vision Transformer 的学习链路：

```text
Word2Vec
→ Token Embedding
→ Position Encoding
→ Self-Attention
→ Multi-Head Attention
→ GPT-2 文本生成
→ ViT 图像 Patch 建模
```

Transformer 的关键思想是使用 Attention 动态建立 token 之间的关系。GPT-2 将文本表示为 token 序列并逐 token 生成回答；ViT 将图像表示为 Patch token 序列并利用 Class Token 完成图像分类。二者处理的数据不同，但都建立在 Embedding、Position Encoding 和 Self-Attention 之上。




