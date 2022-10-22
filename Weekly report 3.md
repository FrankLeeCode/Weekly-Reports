# Weekly Report 3

**Oct.21** 

## Paper Reading

### Attention is all you need

#### motivation

- 传统RNN方法需要考虑顺序，不能很好的并行计算
- 学习远距离之间的关系会很困难（参数量随着位置距离增加而增加）

#### Model Architecture

![The Transformer - model architecture](https://i.stack.imgur.com/eAKQu.png)

##### Dot-Product Attention

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}}）V
$$

Q，K代表queries，keys，其内容为一组$d_k$维向量堆叠的矩阵。V为values，是一组$d_v$维向量堆叠的矩阵。

##### Multi-Head Attention

$$
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O\\
where \ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

其中投影矩阵$W_i^Q \in \mathbb{R}^{d_{model}\times d_k}, W_i^K \in \mathbb{R}^{d_{model}\times d_k}, W_i^V \in \mathbb{R}^{d_{model}\times d_v}\ and W_i^Q \in \mathbb{R}^{hd_v\times d_{model}}$



##### Feed-Forward Networks

attention层后每个位置都添加有相同的全连接feed-forward网络。
$$
FFN(x)=\max(0, xW_1+b_1)W_2+b_2
$$

##### Positional Encoding

没有RNN，CNN所以失去了位置信息，通过添加positional encoding来给网络提供位置信息。
$$
PE_{(pos,2i)}=\sin(\frac{pos}{10000^{2i/d_{model}}}) \\
PE_{(pos,2i+1)}=\cos(\frac{pos}{10000^{2i/d_{model}}})
$$
$pos$为位置，$i$为维度。通过加和的方式直接加入input embedding（保证位置信息与其维度相同）



### ViT

> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

#### Motivation

- Transformer结构在NLP上面发展不错，但在计算机视觉还有很多局限。
- 没必要依赖CNN，纯transformer用于图像分类也应该性能优异。

#### Model Architecture

![ViT Model overview](https://github.com/lucidrains/vit-pytorch/raw/main/images/vit.gif)

##### input处理

标准transformer接受一维token，为了处理二维的图像信息，把图像reshape：$x\in \mathbb{R}^{H\times W\times C} \to x_p \in \mathbb{R}^{N\times(P^2 \cdot C)}$其中$P$为分割后图片尺寸；$N=HW/P^2$代表分割后图片分块数量，同时也是输入Tranformer的序列长度。由于Transformer使用潜在向量维度D，所以需要把图片拉平后投影为D维度向量再输入。

##### Patch + Position embedding

$$
\begin{align}
z_0 &= [x_{class};x^1_pE;x^2_pE;\cdots;x^N_pE]+E_{pos},  &&E\in\mathbb{R}^{(P^2\cdot C)\times D} ,E_{pos}\in\mathbb{R}^{(N+1)\times D} \\
z'_l &=MSA(LN(z_{l-1}))+z_{l-1}, &&l=1\dots L \\
z_l &=MLP(LN(z'_{l}))+z'_{l}, &&l=1\dots L \\
y &= LN(z^0_L)
\end{align}
$$

其中$LN$为Layernorm，

在所有分块序列embedding之前添加了一个可学习的embedding（$z^0_0=x_{class}$）,它通过transformer后的输出状态（$z^0_L$）用作图像表示$y$。

## Concepts

### Generative model & Discriminative model

![discriminative_vs_generative](https://duphan.files.wordpress.com/2016/09/discriminative_vs_generative.png?w=1100)

先定义几个符号意思：

- $P(C_i)$: 在数据集中$C_i$类数据所占的比率
- $P(x|C_i)$: 在数据集中$C_i$类中，某个数据x属于$C_i$的概率
- $P(C_i|x)$: 某个数据x，属于数据集中$C_i$类的概率（最终目的）

#### Discriminative model

Discriminative model不管其他分布，直接对$P(C_i|x)$建模、求解。以一个一层神经网络为例：

设模型参数为$w, b$，则$P(C_i|x)=sigmoid(wx+b)$，再对$C_i$打上1/0标签，通过损失函数训练得到模型参数$w, b$。

如果$P(C_i|x)=\max_{j=1}^{n}{P(C_j|x)}$，则判定x属于$C_i$类。

#### Generative model

Generative Model会先假定$C_i$分布模型，再计算x属于$C_i$的概率，从而确定x的类别概率。

##### 假定$C_i$分布

一般来说我们把它假定为高斯分布，因为高斯分布模型参数好计算，而且大自然中很常见。高斯分布为：
$$
f(x)=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\pi}{\sigma})^2}
$$

##### 计算模型参数

对于高斯分布而言，其均值和方差满足如下公式：
$$
\mu^* = \frac{1}{n}\sum_{i=1}^n x_i; \quad \Sigma^*=\frac{1}{n}\sum_{i=1}^n (x_i-\mu^*)(x_i-\mu^*)^T
$$
所以我们得到了$C_i$分布的参数$\mu_i\in C^n, \Sigma_i\in C^{n\times n}$，即$C_i \sim N(\mu_i,\Sigma_i)$（多元高斯分布）

##### 计算概率并分类

带入多元高斯分布函数中可得x属于$C_i$的概率分布:
$$
P(x|C_i)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma_i|^{1/2}}exp\{-\frac{1}{2}(x-\mu_i)^T\Sigma^{-1}(x-\mu_i)\}
$$
其中，D为向量x的维度，$\Sigma$代表变量 X 的协方差矩阵， i行j列的元素值表示$𝑥_i$与$𝑥_j$的协方差。

有了$P(C_i),P(x|C_i)$就可以计算$P(C_i|x)$:
$$
P(C_i|x)=\frac{P(C_i)P(x|C_i)}{\sum_{j=1}^{n}P(C_j)P(x|C_j)}
$$
这也是贝叶斯公式。

最后$P(C_i|x)=\max_{j=1}^{n}{P(C_j|x)}$，则判定x属于$C_i$类。

## torch

### einops包

> 数据处理维度处理包，分割，拉平等操作

安装

```sh
pip install einops
```

usage

```python
import torch
from einops import rearrange

y = x.transpose(0, 2, 3, 1)
# 改为
y = rearrange(x, 'b c h w -> b h w c')

# Flatten操作
x = torch.randn(64, 3, 16, 16)
y = rearrange(x, 'b c h w -> b (c h w)')
# out: y.shape: torch.Size([64, 768])

# 图片切分重拍操作
x = torch,randn(1, 3, 256, 256)
patch_height = 16
patch_width = 16
# 将图片拆分成16
# x:[b, c, h, w] -> y:[b, N, p^2c]
y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
```

