# **SSD: Single Shot MultiBox Detector**

​	SSD网络是作者Wei Liu在ECCV 2016上发表的论文。对于输入尺寸**300x300**的网络使用Nvidia Titan X在VOC 2007测试集上达到**74.3%**mAP以及**59FPS**，对于512x512的网络，达到了76.9%mAP超越当时最强的Faster RCNN(73.2%mAP)。



# 1. Faster RCNN存在的问题

​	检测流程：将图片输入到backbone进行特征提取得到 feature maps -> 再通过Region Proposals Network 生成一系列 proposals -> 将 proposals 映射到 feature maps 中，得到每一个 proposals 对应的特征矩阵 -> 将每一个特征矩阵通过 ROI pooling层 ，统一缩放到指定的大小 -> 通过目标分类器以及边界框回归器来分别预测每一个 proposals 对应的类别以及边界框回归参数得到预测结果，最后再通过非极大值抑制滤除一些低概率的目标，得到最终的检测结果。

1. 对小目标检测效果很差（只是在一个特征层上预测，而这个特征层是经过了很多卷积层之后所得到的 feature maps ，这里的 feature map 已经被抽象到比较高的层次了。抽象层次越高，一些细节信息的保留就越少。）

2. 模型大，检测速度较慢

   <img src="https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804194303398.png" alt="image-20220804194303398" style="zoom: 80%;" />

# 2. SSD

## 2.1 在不同特征尺度上预测不同尺度的目标

模型框架图

![image-20220830191523174](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830191523174.png)

​	（简化了RCNN加一个特征金字塔）



![image-20220830191633851](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830191633851.png)

s2--padding=1

s1--padding=0

 	在第一个预测特征层中会去预测相对较小的目标，随着抽象程度加深，会去预测较大的目标，

![image-20220830195857043](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830195857043.png)

​	8 × 8的 feature map 抽象程度更低一些，图像的细节信息就会更多一些，就会在相对低层的特征矩阵上去预测较小的目标。

​	default box 与 faster rcnn 中提到的anchor box原理是一样的，只不过把 default box 分别放在不同特征层上。



## 2.2 Default Box的scale以及aspect设定

![image-20220830200617959](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830200617959.png)

源码不通过上面的公式进行计算。

scale--目标尺度、aspect--每个尺度对应的一系列比例

![image-20220830201044467](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830201044467.png)



![image-20220830201207798](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830201207798.png)

每个候选框有两种面积的框，第一种面积有3个比例，第二个只有1个比例。

![image-20220830202548299](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830202548299.png)

6个特征图上总共会生成 38x38x4+ 19x19x6 + 10x10x6+5x5x6+ 3x3x4＋1x1x4= 8732 个 default box.

![image-20220830203643058](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830203643058.png)



## 2.3 Predictor的实现

![image-20220830203922132](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830203922132.png)

​	对于大小为 p 通道的 m×n 特征层，预测潜在检测参数的基本元素是 3×3×p 小卷积核，它产生一个类别的分数，或相对于默认框坐标的形状偏移。

![image-20220830204745143](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830204745143.png)

​	具体来说，对于给定位置的k中的每个框，我们计算**c类分数**和**相对于原始默认框形状的4个偏移量**。这导致在特征图中的每个位置周围应用总共**(c+4)k**个过滤器，**为m×n特征图产生(c+4)kmn输出**。

（这里的c需要包括背景类）

![image-20220830205229229](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830205229229.png)

在feature map上的每个位置都会生成**k个default box**。

边界框回归参数预测的是 x、y、w、h

【注意】fast rcnn predict中输出是 4 × 21，一个anchor针对每个类别都会生成相应的边界框回归参数。



## 2.4 正负样本的选取

​	我们首先将每个真实框与具有最大iou的 default box 进行匹配(如MultiBox[7])。与MultiBox不同的是，我们将 **default box** 与任意一个 **ground truth** 匹配的 **iou** 阈值高于（0.5），则认为是正样本。

![image-20220830210634008](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830210634008.png)

​	因为 default box 与 gt 匹配后得到的正样本数量很少，如果把剩下的 default box 都当做是负样本，则大多数默认框都是负数的，特别是当可能的默认框的数量很大时。这就引入了正样本和负样本训练之间的显著不平衡。

![image-20220830211423692](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830211423692.png)

**难例挖掘：**我们没有使用所有负样本，而是<u>使用每个默认框的最高置信损失对它们进行排序</u>，并选择顶部的，使负和正之间的比率最多为 **3:1** 。我们发现，这能更快的优化和更稳定的训练。



## 2.5 损失计算



![image-20220830212439844](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830212439844.png)

其中N为匹配到的正样本个数，α为1.



## 2.6 类别损失

### 2.6.1 分类损失

![image-20220830212624073](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830212624073.png)

![image-20220830212711826](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830212711826.png)



### 2.6.2 定位损失

![image-20220830213304271](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830213304271.png)

![image-20220830213329755](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830213329755.png)

![image-20220830213507987](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220830213507987.png)

（这里和RCNN一样）



# 3. Nvidia SSD 源码

https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD

使用resnet 50 + fpn 为backbone

- The conv5_x, avgpool, fc and softmax layers were removed from the original classification model.
- All strides in conv4_x are set to 1x1.

![image-20220831095411355](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220831095411355.png)

Training of SSD requires computational costly augmentations. To fully utilize GPUs during training we are using the [NVIDIA DALI](https://github.com/NVIDIA/DALI) library to accelerate data preparation pipelines.

用[NVIDIA DALI](https://github.com/NVIDIA/DALI) 做了数据预处理的部分。



### 混合精度训练

float 32（一般） + float16

![image-20220831095750664](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220831095750664.png)



## 文件结构：

```
├── src: 实现SSD模型的相关模块    
│     ├── resnet50_backbone.py   使用resnet50网络作为SSD的backbone  
│     ├── ssd_model.py           SSD网络结构文件 
│     └── utils.py               训练过程中使用到的一些功能实现
├── train_utils: 训练验证相关模块（包括cocotools）  
├── my_dataset.py: 自定义dataset用于读取VOC数据集    
├── train_ssd300.py: 以resnet50做为backbone的SSD网络进行训练    
├── train_multi_GPU.py: 针对使用多GPU的用户使用    
├── predict_test.py: 简易的预测脚本，使用训练好的权重进行预测测试    
├── pascal_voc_classes.json: pascal_voc标签文件    
├── plot_curve.py: 用于绘制训练过程的损失以及验证集的mAP
└── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
```

## 预训练权重下载地址（下载后放入src文件夹中）：

- ResNet50+SSD: https://ngc.nvidia.com/catalog/models
  `搜索ssd -> 找到SSD for PyTorch(FP32) -> download FP32 -> 解压文件`
- 如果找不到可通过百度网盘下载，链接:https://pan.baidu.com/s/1byOnoNuqmBLZMDA0-lbCMQ 提取码:iggj



部署到生产环境：检测好的模型转化成TensorRT的格式，检测速度还能翻一番。





![image-20220831164355811](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220831164355811.png)



不同目标之间的目标边界框完全隔离开了。

![image-20220831172311426](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220831172311426.png)
