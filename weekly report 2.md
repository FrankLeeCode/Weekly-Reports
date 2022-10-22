# Weekly Report 2

**Oct. 14**

- [x] 论文ppt

## torch tutorial 

> 原视频[torch tutorial](https://www.bilibili.com/video/BV1HY4y1T71A)已经被下架，暂时看到Gan实现部分，后续视频暂时没找到，只能看课件。

### 一些小工具/技巧
#### glob.glob()函数
>  收集图片时可以用到

功能：返回一个某一种文件夹下面的某一类型文件路径列表

`glob.glob(pathname, *, recursive=False)`

```python
import glob

# 返回img文件下所有png文件的路径
imgs_path = glob.glob('img/*.png')
```



#### PIL.Image

> 读取图片文件

功能：读取图片文件

`Image.open(pathname)`

```python
from PIL import Image

Image.open('test.png')

# 可以配合lambda函数 以及transforms快速处理图片为tensor
from torchvision import transforms

tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
  	transforms.Resize(24,24),
    transforms.ToTensor()
])
tf('test.png')

# lambda函数
# 简介版def函数，lambda arg1, arg2... : f(arg1, arg2...)
# 变量：arg1, arg2...， 返回值：f(arg1, arg2...)
f = lambda a, b: a+b
f(1, 2)
# out: 3
```



#### Trasnform.Normalize(mean, std)

> for `n` channels, this transform will normalize each channel of the input `torch.*Tensor` i.e., `output[channel] = (input[channel]-mean[channel]) / std[channel]`

功能：图像归一化，把图片从0～1数据移动到-1～1区间

```python
from torchvision import transforms

# mean均值， std方差。下数据为imagNet的统计归一化数据
transforms.Normalize(mean=[0.485, 0.256, 0.406],
                     std=[0.229, 0.224, 0.225])

# 归一化后图片显示会有问题，所以需要反归一化
# denormalize操作：
def denormalize(x_hat):  
  mean = [0.485, 0.256, 0.406]
	std = [0.229, 0.224, 0.225]
    
  # x_hat = (x-mean)/std
  # x = x_hat*std = mean
  # x:[c, h, w], mean:[3] => [3, 1, 1]
  mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
  std = torch.tensor(std).unsqueeze(1).unsqueeze(2)
  
  x = x_hat * std + mean
  
  return x
```



#### ImageFolder(root, transform)

功能：如果数据集是以文件夹为分割，文件夹名称为类别，内为类别对应所有图片数据。则可让imagefolder直接生成数据集。

```python
import torchvision
from torch.utils.data import DataLoader

# 文件类似 path/label1/0001.jpeg, path/label2/0003.png
db = torchvision.datasets.ImageFolder(root='path', transform=tf)
loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8) # num_workers表示多线程一次性取多少张图片。

db.class_to_idx # 返回字典，里面为label对应的编码
```



### 一些理论知识点

通过dropout或者给训练数据集增加noise可以一定增强网络robustness，使其能学到更高层次的特征信息，并且不只依赖于部分神经元，还能防止过拟合。



## Paper Reading

### Zero-Shot Brief Review 
[1]: https://ieeexplore.ieee.org/document/9895459	"G. -S. Xie, Z. Zhang, H. Xiong, L. Shao and X. Li, &quot;Towards Zero-Shot Learning: A Brief Review and an Attention-based Embedding Network,&quot; in IEEE Transactions on Circuits and Systems for Video Technology, 2022, doi: 10.1109/TCSVT.2022.3208071."

暂时无法很好的理解，还需要时间积累。



### Dual Progressive Prototype Network for Generalized Zero-Shot Learning

[2]: http://staff.ustc.edu.cn/~xjchen99/nips_web/nips.htm

详见 [DPPN论文阅读.pdf](DPPN论文阅读.pdf) 




## Next Week

1. 查询attention机制相关信息，transformer实现原理等内容。
2. 找寻zero-shot数据集，了解内容
3. 查看DPPN代码，了解具体实现方法。