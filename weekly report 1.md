# Weekly Report 1

**Oct. 6**


> 本周主要在推进pytorch tutorial的进度，目前找相关文献水平有限，没找到好的切入点所以效率不高，时间主要花在基础的pytorch学习上。

## pytorch tutorial
目前进度为[84/150](https://www.bilibili.com/video/BV1HY4y1T71A)，结束了resnet18的复现部分。pytorch tutorial比起一些直接的综述或者深度学习相关的书来说，以实际应用和代码的方式讲解，可以更直观的了解经典网络结构，会比只看理论￼介绍要深刻很多，好理解很多。

### tutorial中的一些知识点
overfiting处理方法
- Early Stopping：选取overfitting之前的epoch为最终参数。
- Dropout：舍去一些神经元以到达Learning less to learn better的目的。

数据增强方法：

- Flip
- Rotate
- Random Move & Crop
- GAN网络

## Paper reading

还没找到合适的切入点，面对玲琅满目的文章不知从何下手。本想以老师您的文章为切入点，但是似乎您的[Google Scholar主页](https://scholar.google.com/citations?user=LKaWa9gAAAAJ&hl=en)已经关闭。所以暂时从我知道的几个关键词：few shot， zero shot着手查看了一些相关综述进行了解。

### few-shot

样本少，以至于不能支撑训练传统的深度神经网络。few-shot通过训练“相似网络”将support set与query输入网络对比相似度，选出相似度最高的作为结果。

#### 一个简单有效的方法Fine tuning

在求出特征向量$x_i$后，用support set训练最终分类参数w，b，从而提高准确率。
$$
p_j= Softmax(W\cdot f(x_j)+b), \quad loss=\sum_jCrossEntropy(y_j, p_j),\ y_j为标签
$$

### zero shot

利用训练集数据训练模型，使得模型能够对测试集的对象进行分类，但是训练集类别和测试集类别之间没有交集；期间需要借助类别的描述，来建立训练集和测试集之间的联系，从而使得模型有效。相比起few-shot根据几张有标签的图片对输入进行识别（归类），zero shot只提供特征向量（语意描述）。



## Next Week Goal

1. 继续完成tutorial的学习
2. 选出一篇文献作为切入点开始阅读
3. 丰富周报内容
