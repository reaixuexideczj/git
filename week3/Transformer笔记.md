# Transformer 核心模块学习笔记

## 1. 学习目标

本文整理 Transformer 的核心数据处理流程和主要模块，重点回答以下问题：

1. Transformer 如何处理文本数据？
2. Q、K、V 分别是什么？
3. Scaled Dot-Product Attention 如何计算？
4. Softmax 在注意力机制中有什么作用？
5. 多头注意力为什么能够提升模型能力？
6. 位置编码有什么作用，有哪些实现方式？
7. Transformer Block 包含哪些层，为什么需要这些层？

---

## 2. Transformer 的整体作用

Transformer 是一种处理序列数据的神经网络架构。它的核心能力是让序列中的每个 Token 根据任务需要，有选择地读取其他 Token 的信息，从而得到包含上下文的表示。

例如：

```text
小猫 喜欢 吃 鱼
```

初始 Embedding 只表示每个 Token 本身。经过 Transformer 后，“吃”的向量可以融合“小猫”“喜欢”和“鱼”的相关信息，从而成为一个包含当前句子上下文的向量。

整体流程可以概括为：

```text
文本
  ↓
Tokenization
  ↓
Token编号
  ↓
Token Embedding + Position Encoding
  ↓
多个Transformer Block
  ↓
上下文相关的Token表示
  ↓
输出层完成分类、生成或其他任务
```

Word2Vec 通常为一个词提供固定词向量，而 Transformer 会根据当前上下文继续更新 Token 表示。因此，同一个词出现在不同句子中，最终可以得到不同的上下文向量。

---

## 3. 输入表示：Token Embedding 与位置编码

### 3.1 Token Embedding

神经网络不能直接处理字符串，因此需要先将 Token 编号转换成连续向量：

```python
import torch
import torch.nn as nn

vocab_size = 100
d_model = 16

token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
)

token_ids = torch.tensor([
    [2, 5, 7, 9],
])

token_vectors = token_embedding(token_ids)

print("Token编号形状：", token_ids.shape)
print("Token向量形状：", token_vectors.shape)
```

输出形状为：

```text
Token编号：[batch_size, sequence_length]
Token向量：[batch_size, sequence_length, d_model]
```

### 3.2 为什么需要位置编码

Self-Attention 本身没有天然的顺序结构。下面两个句子包含相同的 Token，但顺序和含义不同：

```text
我 喜欢 你
你 喜欢 我
```

位置编码用于告诉模型每个 Token 在序列中的位置。Transformer 的输入通常是：

$$
X=E_{token}+P_{position}
$$

可以理解为：

```text
Token Embedding：这个Token是什么
Position Encoding：这个Token在哪里
```

两者相加后形状保持不变。

---

## 4. 正弦余弦位置编码

原始 Transformer 使用固定的正弦余弦位置编码：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中：

- $pos$ 表示 Token 的位置；
- $i$ 表示位置向量中的维度；
- $d_{model}$ 表示模型特征维度；
- 偶数维使用正弦函数；
- 奇数维使用余弦函数。

不同维度使用不同频率，使不同位置得到不同的波形组合。

```python
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=512):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("本示例要求d_model为偶数")

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度，形状变成[1, max_length, d_model]
        pe = pe.unsqueeze(0)

        # 位置编码随模型移动到CPU/GPU，但不参与梯度更新
        self.register_buffer("pe", pe)

    def forward(self, x):
        sequence_length = x.size(1)
        return x + self.pe[:, :sequence_length]
```

使用方法：

```python
position_encoding = SinusoidalPositionalEncoding(
    d_model=d_model,
    max_length=128,
)

x = position_encoding(token_vectors)

print("加入位置编码后的形状：", x.shape)
```

另一种常见方式是可学习位置嵌入：

```python
position_embedding = nn.Embedding(
    num_embeddings=128,
    embedding_dim=d_model,
)

sequence_length = token_ids.size(1)
position_ids = torch.arange(sequence_length)
position_vectors = position_embedding(position_ids)

x = token_vectors + position_vectors
```

正弦余弦位置编码由公式直接生成，不参与训练；可学习位置嵌入通常随机初始化，再通过训练调整。

---

## 5. Q、K、V 是什么

注意力机制中的三个核心向量是：

- Query（Q）：当前 Token 想寻找什么信息；
- Key（K）：每个 Token 提供的匹配特征；
- Value（V）：匹配成功后真正要读取的信息。

可以使用搜索引擎类比：

```text
Q：搜索框中输入的查询内容
K：每个搜索结果的标题和标签
V：搜索结果页面中的实际内容
```

Q 与 K 决定应该关注谁，V 决定关注后获取什么信息。

Q、K、V 都由同一个输入 $X$ 经过不同的可训练线性变换得到：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

```python
batch_size = 1
sequence_length = 4
d_model = 8
d_k = 4
d_v = 4

torch.manual_seed(42)

x = torch.randn(batch_size, sequence_length, d_model)

W_q = torch.randn(d_model, d_k)
W_k = torch.randn(d_model, d_k)
W_v = torch.randn(d_model, d_v)

query = x @ W_q
key = x @ W_k
value = x @ W_v

print("X形状：", x.shape)
print("Q形状：", query.shape)
print("K形状：", key.shape)
print("V形状：", value.shape)
```

---

## 6. Scaled Dot-Product Attention

核心公式为：

$$
A(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

它可以拆成四步。

### 6.1 Q 与 K 点积

$$
QK^T
$$

点积用于计算每个 Query 与所有 Key 的匹配程度。假设序列长度为 $T$，注意力分数矩阵形状就是：

```text
[batch_size, T, T]
```

矩阵的每一行表示一个 Query 对所有 Key 的关注分数。

### 6.2 缩放

$$
\frac{QK^T}{\sqrt{d_k}}
$$

当向量维度较高时，点积绝对值容易变大，使 Softmax 过度饱和，影响梯度传播。除以 $\sqrt{d_k}$ 可以让分数尺度更加稳定。

### 6.3 Softmax

Softmax 把任意实数分数转换成非负且总和为 1 的权重：

$$
softmax(z_i)
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

例如：

```text
原始分数：[0.2, 0.1, 0.9]
Softmax： [0.25, 0.23, 0.52]
```

注意力中的 Softmax 通常沿最后一个维度计算，因此每个 Query 对所有 Key 的权重之和等于 1。

### 6.4 对 V 加权求和

$$
AV
$$

例如：

$$
0.25V_1+0.23V_2+0.52V_3
$$

输出是所有 Value 向量按照注意力权重形成的加权组合。

### 6.5 PyTorch 实现

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf"),
            )

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(
            attention_weights,
            value,
        )

        return output, attention_weights
```

测试：

```python
attention = ScaledDotProductAttention()

output, weights = attention(query, key, value)

print("注意力权重形状：", weights.shape)
print("输出形状：", output.shape)
print("每一行权重之和：", weights.sum(dim=-1))
```

由于本示例的输入和投影矩阵是随机的，注意力权重暂时没有真实语义。模型经过训练后，Q、K、V 的投影矩阵才会逐渐学到对任务有用的关系。

---

## 7. Softmax 适合处理什么情况

Softmax 适合把一组互相竞争的分数转换成概率分布，常见场景包括：

1. 单标签多分类，例如一张图片只能属于猫、狗、汽车中的一个类别；
2. 语言模型预测词表中的下一个 Token；
3. 注意力机制中将匹配分数转换成注意力权重。

Softmax 不适合互相独立的多标签问题。例如一张图片可以同时包含人、汽车和道路，此时通常对每个类别分别使用 Sigmoid。

训练分类模型时，`nn.CrossEntropyLoss` 已经包含适合训练的 Softmax 相关计算，因此不要在交叉熵损失之前手动调用 Softmax。

---

## 8. 多头注意力

单头注意力对每个 Query 只生成一套注意力分布。多头注意力让模型同时生成多套注意力分布，从不同的表示子空间读取信息。

每个头的计算为：

$$
head_i
=
A
\left(
QW_i^Q,
KW_i^K,
VW_i^V
\right)
$$

所有头的输出拼接后再进行线性变换：

$$
MultiHead(Q,K,V)
=
Concat
(head_1,\ldots,head_h)W^O
$$

假设：

```text
d_model = 512
num_heads = 8
```

每个头的维度为：

$$
d_{head}=\frac{512}{8}=64
$$

因此必须满足：

$$
d_{model}\bmod h=0
$$

### 8.1 为什么分开计算反而有效

多头注意力不是直接把原始输入向量机械地切成几段，而是先通过可训练矩阵把全部输入特征投影成 Q、K、V，再拆分成多个头。

它有效的主要原因是：

1. 每个头拥有自己的投影子空间；
2. 每个头独立产生一套 Softmax 注意力权重；
3. 不同关系可以在拼接前分别保留；
4. 拼接后的输出经过 $W^O$ 再次融合；
5. 在总维度不变的情况下，模型获得多种并行的信息读取方式。

例如处理“吃”时：

```text
头1可能主要关注“小猫”
头2可能主要关注“鱼”
头3可能主要关注“喜欢”
```

这只是直观解释，实际训练中不保证每个头都有固定且清晰的人类语义。

### 8.2 PyTorch 实现

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model必须能被num_heads整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        self.key_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        self.value_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        self.output_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.attention = ScaledDotProductAttention()

    def split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape

        tensor = tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return tensor.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.shape

        query = self.split_heads(
            self.query_projection(x)
        )
        key = self.split_heads(
            self.key_projection(x)
        )
        value = self.split_heads(
            self.value_projection(x)
        )

        context, attention_weights = self.attention(
            query,
            key,
            value,
            mask=mask,
        )

        # [B, H, T, D_head] → [B, T, H, D_head]
        context = context.transpose(1, 2).contiguous()

        # 拼接所有注意力头
        context = context.view(
            batch_size,
            sequence_length,
            self.d_model,
        )

        output = self.output_projection(context)

        return output, attention_weights
```

测试：

```python
torch.manual_seed(42)

x = torch.randn(1, 4, 8)

multi_head_attention = MultiHeadSelfAttention(
    d_model=8,
    num_heads=2,
)

output, attention_weights = multi_head_attention(x)

print("输入形状：", x.shape)
print("注意力权重形状：", attention_weights.shape)
print("最终输出形状：", output.shape)
```

注意力权重形状 `[1, 2, 4, 4]` 分别表示：

```text
1：batch_size
2：注意力头数量
4：Query数量
4：Key数量
```

---

## 9. 因果掩码

用于文本生成的 Decoder 不能让当前位置查看未来 Token，否则训练时会发生答案泄露。

```text
        我  喜欢  吃  苹果
我      ✓   ×    ×    ×
喜欢    ✓   ✓    ×    ×
吃      ✓   ✓    ✓    ×
苹果    ✓   ✓    ✓    ✓
```

创建因果掩码：

```python
def create_causal_mask(sequence_length, device=None):
    mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        )
    )

    # 形状变为[1, 1, T, T]，便于广播到batch和head维度
    return mask.unsqueeze(0).unsqueeze(0)


causal_mask = create_causal_mask(
    sequence_length=x.size(1),
    device=x.device,
)

masked_output, masked_weights = multi_head_attention(
    x,
    mask=causal_mask,
)

print("因果掩码：")
print(causal_mask[0, 0])

print("第一个注意力头的权重：")
print(masked_weights[0, 0])
```

未来位置在 Softmax 后的权重会变成 0。

位置编码与因果掩码的作用不同：

```text
位置编码：告诉模型Token位于哪里
因果掩码：限制当前Token允许查看哪些位置
```

---

## 10. Feed Forward Network

注意力完成 Token 之间的信息交流后，还需要对每个 Token 的内部特征进行非线性加工，这由前馈神经网络完成。

典型结构为：

$$
FFN(x)
=
GELU(xW_1+b_1)W_2+b_2
$$

数据流通常是：

```text
d_model
  ↓
扩大到4 × d_model
  ↓
GELU
  ↓
恢复到d_model
```

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)
```

Attention 和 FFN 的区别可以概括为：

```text
Attention：在不同Token之间交换信息
FFN：对每个Token内部的特征进行加工
```

可以将 Attention 类比为“开会交流”，将 FFN 类比为“每个人会后独立整理和思考”。

---

## 11. Residual、LayerNorm 与 Dropout

### 11.1 Residual Connection

残差连接把模块输入直接加回输出：

$$
Y=X+F(X)
$$

它可以：

- 保留原始信息；
- 为梯度提供更直接的传播路径；
- 让深层网络更容易训练。

### 11.2 Layer Normalization

LayerNorm 对每个 Token 的特征进行标准化：

```python
normalization = nn.LayerNorm(d_model)
```

它可以稳定特征数值和训练过程，改善梯度传播。

### 11.3 Dropout

Dropout 在训练时随机将部分特征设置为 0：

```python
dropout = nn.Dropout(0.1)
```

它用于减少过拟合。推理时调用 `model.eval()`，Dropout 会自动关闭。

---

## 12. 完整 Transformer Block

下面实现一个 Pre-LN Transformer Block：

$$
X'=X+A(LN(X))
$$

$$
X_{next}=X'+FFN(LN(X'))
$$

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.attention_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            hidden_dim=4 * d_model,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        attention_output, attention_weights = self.attention(
            self.norm1(x),
            mask=mask,
        )

        x = x + self.attention_dropout(attention_output)

        feed_forward_output = self.feed_forward(
            self.norm2(x)
        )

        x = x + feed_forward_output

        return x, attention_weights
```

整体测试：

```python
torch.manual_seed(42)

batch_size = 2
sequence_length = 5
vocab_size = 100
d_model = 16
num_heads = 4

token_ids = torch.randint(
    low=0,
    high=vocab_size,
    size=(batch_size, sequence_length),
)

token_embedding = nn.Embedding(vocab_size, d_model)
position_encoding = SinusoidalPositionalEncoding(
    d_model=d_model,
    max_length=128,
)

block = TransformerBlock(
    d_model=d_model,
    num_heads=num_heads,
    dropout=0.1,
)

# Token Embedding + Position Encoding
x = token_embedding(token_ids)
x = position_encoding(x)

# 如果是Encoder式双向注意力，可以令mask=None
output, attention_weights = block(x, mask=None)

print("Token编号形状：", token_ids.shape)
print("Transformer输入形状：", x.shape)
print("Transformer输出形状：", output.shape)
print("注意力权重形状：", attention_weights.shape)
```

预期输出形状：

```text
Token编号形状：[2, 5]
Transformer输入形状：[2, 5, 16]
Transformer输出形状：[2, 5, 16]
注意力权重形状：[2, 4, 5, 5]
```

使用因果注意力：

```python
causal_mask = create_causal_mask(
    sequence_length=sequence_length,
    device=x.device,
)

causal_output, causal_weights = block(
    x,
    mask=causal_mask,
)

print("因果注意力输出形状：", causal_output.shape)
print("因果注意力权重形状：", causal_weights.shape)
```

---

## 13. Transformer Block 为什么包含这些层

| 模块 | 主要作用 |
|---|---|
| Token Embedding | 将 Token 编号转换成连续向量 |
| Position Encoding | 加入 Token 顺序和位置信息 |
| Q、K、V 投影 | 分别产生查询、匹配和内容表示 |
| Scaled Dot-Product Attention | 计算并汇总相关 Token 的信息 |
| Softmax | 将匹配分数转换成总和为 1 的注意力权重 |
| Multi-Head Attention | 使用多套注意力分布并行读取信息 |
| Causal Mask | 防止生成模型查看未来 Token |
| Feed Forward Network | 加工每个 Token 内部的特征 |
| GELU | 增加非线性表达能力 |
| Residual Connection | 保留原始信息并帮助梯度传播 |
| LayerNorm | 稳定特征数值和训练过程 |
| Dropout | 减少过拟合 |
| 多层堆叠 | 逐层形成更复杂的上下文表示 |

每个 Transformer Block 最核心的分工是：

```text
Attention：Token之间进行信息交流
FFN：Token内部进行特征加工
Residual：保留已有信息
LayerNorm：维持训练稳定
```

---

## 14. Encoder、Decoder 与 Decoder-only

原始 Transformer 包含 Encoder 和 Decoder。

### Encoder

Encoder 通常允许每个 Token 查看序列中的所有有效位置，适合理解输入文本。

```text
Embedding
→ Position Encoding
→ Multi-Head Self-Attention
→ FFN
→ 上下文表示
```

### Decoder

自回归 Decoder 使用因果掩码，只允许当前位置查看自己和左侧 Token，用于逐步生成文本。

原始 Encoder-Decoder Transformer 的 Decoder 还包含 Cross-Attention，用 Decoder 的 Q 查询 Encoder 提供的 K 和 V。

### Decoder-only

Decoder-only 模型只堆叠带因果掩码的 Transformer Block，根据前文预测下一个 Token。

---

## 15. 核心公式总结

### 输入表示

$$
X=E_{token}+P_{position}
$$

### Q、K、V

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

### Scaled Dot-Product Attention

$$
A(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

### Multi-Head Attention

$$
MultiHead(Q,K,V)
=
Concat
(head_1,\ldots,head_h)W^O
$$

### Feed Forward Network

$$
FFN(x)
=
GELU(xW_1+b_1)W_2+b_2
$$

### Pre-LN Transformer Block

$$
X'=X+A(LN(X))
$$

$$
X_{next}=X'+FFN(LN(X'))
$$

---

## 16. 最终理解

Transformer 首先把 Token 转换成向量，并加入位置编码。Self-Attention 使用 Q 和 K 计算 Token 之间的相关程度，通过 Softmax 得到注意力权重，再使用这些权重对 V 加权求和。

多头注意力在多个学习得到的子空间中独立执行注意力计算，使模型能够同时保留多套信息读取结果。多个头的输出被拼接，再经过输出矩阵融合。

注意力负责 Token 之间的信息交流，前馈网络负责每个 Token 内部的非线性特征加工。残差连接保留已有信息并帮助梯度传播，LayerNorm 稳定训练，Dropout 减少过拟合。多个 Transformer Block 堆叠后，模型能够逐层建立更复杂的上下文表示。

一句话概括：

> Transformer 通过位置编码保留顺序，通过多头注意力在 Token 之间传递信息，通过前馈网络加工特征，再利用残差连接和 LayerNorm 稳定地堆叠多层，最终获得包含上下文的 Token 表示。
