# Weekly Report

**Nov. 18**

## paper reading

### Zero-Shot Learning - The Good, the Bad and the Ugly

原来的数据集 SS(standard split) 分割中，有很多测试集的图像与ImageNet中的数据重合，而这些数据用来训练了ResNet。此时如果用ResNet提取图像特征用于Zero-shot测试，则结果通常会比那些未重叠的图像数据高，所以提出了一种新的数据分割方式PS( proposed split) 用于Zero-shot，保证了测试集部分与ImageNet没有重叠现象。

#### Datasets

| Dataset | Size   | Detail | Att  | Class | Class_train | Class_test |
| ------- | ------ | ------ | ---- | ----- | ----------- | ---------- |
| SUN     | medium | fine   | 102  | 717   | 580+65      | 72         |
| CUB     | medium | fine   | 312  | 200   | 100+50      | 50         |
| AWA     | medium | coarse | 85   | 50    | 27+13       | 10         |
| aPY     | small  | coarse | 64   | 32    | 15+5        | 12         |

> 100+50为一共150类用于测试，其中随机选择50类用于validation，其余同理

#### 分割方式对比

Training Time

| Dataset | Total_images | SS_train | SS_test | PS_train | PS_test |
| ------- | ------------ | -------- | ------- | -------- | ------- |
| SUN     | 14K          | 12900    | 0       | 10320    | 0       |
| CUB     | 11K          | 8855     | 0       | 7057     | 0       |
| AWA     | 30K          | 24295    | 0       | 19832    | 0       |
| aPY     | 15K          | 12695    | 0       | 5932     | 0       |

Evaluation  Time

| Dataset | Total_images | SS_train | SS_test | PS_train | PS_test |
| ------- | ------------ | -------- | ------- | -------- | ------- |
| SUN     | 14K          | 0        | 1440    | 2580     | 1440    |
| CUB     | 11K          | 0        | 2933    | 1764     | 2967    |
| AWA     | 30K          | 0        | 6180    | 4958     | 5685    |
| aPY     | 15K          | 0        | 2644    | 1483     | 7924    |

> 测试时，PS同时有训练集的类别和测试集的类别，因为在训练和测试课上评估准确性对于显示方法的泛化是至关重要的。

#### 实验结果

-  max-margin compatibility learning的方法结果好于attribute classifier learning 或者 hybrid methods。
- novelty detection scheme可以提高效果





### Distinguishing Unseen from Seen for Generalized Zero-shot Learning

#### motivation

- GZSL会把未见类识别为已见类，或者已见类识别为未见类，所以区分已见和未见领域是一种有效的解决方案

#### 常见方法

1. embedding methods
2. generative methods
3. domain-aware methods: 显式的区分可见域和不可见域



#### 文章方法

![distinguishing unsenn from seen network](assets/distinguishing unsenn from seen network.png)

##### Wasserstein distance

$$
\mathcal{L}_{W1}=\inf_{\gamma\in \prod(p_{z_x}p_{z_{a^s}})} \mathbb{E}_{(Z_x,Z_{a^s})} \sim[\|z_x-z_{a^s}\|]
$$

优点：即使两个分布不重叠也能使用

##### VAE训练损失

$$
\begin{align}
\mathcal{L}_{VAE}&=
\mathbb{E}_{q_{\phi1}(z_x|x)}[\log p_{\theta_1}(x|z_x)]-\lambda KL(q_{\phi1}(z_x|x)\|p(z_x)) 
\\ &+ 
\mathbb{E}_{q_{\phi2}(z_{a^s}|a^s)}[\log p_{\theta_2}(a^s|z_{a^s})]-\lambda KL(q_{\phi2}(z_{a^s}|a^s)\|p(z_{a^s}))
\\
\\
\mathcal{L}_{cls1} &= -\mathbb{E}[p_{z_x}\log q_{z_x}]-\mathbb{E}[p_{z_{a^x}}\log q_{z_{a^x}}]
\end{align}
$$

##### fictitious sample

通过未见类的特征最终生成虚假未见类的潜在表示
$$
z_{\tilde{x}}=E_v(\tilde{x}), \ \tilde{x}=D_v(E_s(a^u))
$$
理论上通过最小化下式来训练分类器
$$
-\mathbb{E}[p_{z_x}\log q_{z_x}]-\mathbb{E}[p_{z_{\tilde{x}}}\log q_{z_{\tilde{x}}}]
$$
但是生成的

#### Qustion

1. 方法为什么不算Generative?