# 一、RCNN

2014年

作者Ross Girshick多次在PASCAL VOC的目标检测竞赛中折桂，曾在2010年带领团队获得终身成就奖。

## 1. RCNN算法流程可分为4个步骤

1. 一张图像生成**1K~2K**个候选区域(使用Selective Search方法)
2. 对每个候选区域，使用深度网络提取特征
3. 特征送入每一类的SVM分类器，判别是否属于该类
4. 使用回归器精细修正候选框位置

![image-20220804160459837](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804160459837.png)

​	第二步将每一个候选框送到网络里面取提取特征会有大量冗余，因为候选框很多框住的是同样的位置。

### 1.1 候选区域的生成

​	利用**selective Search算法**通过图像分割的方法得到一些原始区域，然后使用一些**合并策略**将这些区域合并，得到一个层次化的区域结构，而这些结构就包含着可能需要的物体。

<img src="https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804150713301.png" alt="image-20220804150713301" style="zoom:33%;" />

#### 1.1.1 Selective Search

​	Selective Search 算法，训练过程中用于从输入图像中搜索出2000个Region Proposal

- 使用一种过分割手段，将图像分割成小区域 (1k~2k 个)
- 计算所有邻近区域之间的相似性，包括颜色、纹理、尺度等
- 将相似度比较高的区域合并到一起
- 计算合并区域和临近区域的相似度
- 重复3、4过程，直到整个图片变成一个区域

​	在每次迭代中，形成更大的区域并将其添加到区域提议列表中。这种自下而上的方式可以创建从小到大的不同scale的Region Proposal



### 1.2 对每个候选区域，使用深度网络提取特征

​	将**2000个候选区域**缩放到**227x227**pixel，接着将候选区域输入事先训练好的AlexNet CNN网络获取4096维的特征得到**2000×4096**维矩阵。

![image-20220822160643808](images/image-20220822160643808.png)

​	由于物体标签训练数据少，如果要直接采用随机初始化CNN参数的方法是不足以从零开始训练出一个好的CNN模型。基于此，采用有监督的预训练，使用一个大的数据集（**ImageNet ILSVC 2012**）来训练AlexNet，得到一个1000分类的预训练（Pre-trained）模型。

​	这里的图像分类网络去掉了全连接层，得到4096维的特征向量，有2000个候选框，所以有2000*4096维的特征矩阵，每一行对应每一个候选区域的特征向量。



### 1.3 特征送入每一类的SVM分类器，判定类别

​	SVM分类器是二分类的分类器。

​	将2000×4096维特征候选框与20个SVM组成的**权值矩阵**4096×20相乘，获得2000×20维矩阵表示每个建议框是某个目标类别的**得分**（概率向量）。分别对上述2000×20维矩阵中每一列即每一类进行**非极大值抑制**剔除重叠建议框，得到该列即该类中得分最高的一些建议框。

​	<u>Pascal VOC当中有20个类别，所以有20个SVM分类器。</u>

![image-20220822160704431](images/image-20220822160704431.png)



#### 1.3.1 非极大值抑制剔除重叠建议框

​	如果计算IOU是大于给定的阈值，则认为这两个目标是同一个目标，然后就会把概率低的删掉。对每一列进行非极大值抑制处理。

![image-20220822160714141](images/image-20220822160714141.png)



### 1.4  使用回归器精细修正候选框位置

​	对NMS处理后剩余的建议框进一步筛选。接着分别**用20个回归器对上述20个类别中剩余的建议框进行回归操作**，最终得到每个类别的修正后的得分最高的bounding box。
​	如图，黄色框口P表示建议框Region Proposal，绿色窗口G表示实际框Ground Truth，红色窗口G^表示Region Proposal进行回归后的预测窗口，可以用**最小二乘法解决的线性回归问题**。

​	通过回归参数得到四个参数，<u>分别对应目标建议框中心点x的偏移量、y的偏移量、边界框高度缩放的因子、宽度缩放因子</u>，得到的四个值对建议框进行修正，得到红色的边界框。

![image-20220822160727433](images/image-20220822160727433.png)



## 2. RCNN框架

![image-20220804161055070](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804161055070.png)

### 2.1 存在的问题

#### 2.1.1 测试速度慢

​	测试一张图片约53s (CPU)。用Selective Search算法提取候选框用时约2秒，一张图像内候选框之间存在大量重叠，提取特征操作冗余。

#### 2.1.2 训练速度慢

​	过程及其繁琐

#### 2.1.3 训练所需空间大

​	对于SVM和bbox回归训练，需要从每个图像中的每个目标候选框提取特征，并写入磁盘。对于非常深的网络，如VGG16，从VOC2007训练集上的5k图像上提取的特征需要数百GB的存储空间。

#### 2.1.4 为什么在训练cnn的时候正例负例定义与训练SVM时不一致？

​	训练CNN的时候，IOU大于0.5标记为正样本，其他的标记为背景，而在训练SVM的时候，IOU小于0.3的标记为负样本，大于0.7是正样本，ground truth为正样本，其他的丢弃。**在训练CNN的时候对正例样本定义相对宽松，会在一定程度上加大正例的数据量**，防止网络的过拟合，**而SVM这种算法的机制，适合小样本的训练**，因此对正样本限制严格。

​	

#### 2.1.5 为什么不直接采用softmax而是采用SVM？

​	作者尝试了采用softmax直接进行训练，但是效果很差，作者认为当IOU大于0.5就认为是正样本会导致定位准确度的下降，而又需要采用IOU阈值0.5来训练CNN，因此采用CNN+SVM结合的方法来完成算法。



# 二、Fast RCNN

​	Fast R-CNN（**2015**年发表）是作者**Ross Girshick**继R-CNN后的又一力作。**同样**使用**VGG16**作为网络的backbone，<u>与R-CNN相比训练时间快9倍</u>，<u>测试推理时间快213倍</u>，<u>准确率从62%提升至66%</u>(在Pascal voc数据集上)。推理速度在GPU上达到5fps，每秒钟能检测5张图片，这个时间还包括了候选区域生成的部分。



## 2.1 Fast R-CNN算法流程可分为3个步骤

1. 一张图像生成**1K~2K**个候选区域(使用**Selective Search**方法)

2. **将图像输入网络**得到相应的特征图，将SS算法生成的<u>候选框投影到特征图上获得相应的特征矩阵</u>

3. <u>将每个特征矩阵通过ROI pooling层缩放到7x7大小的特征图</u>，接着将特征图展平通过一系列全连接层得到预测结果

   ROI——Region of Interest 感兴趣区域

![image-20220804162534045](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804162534045.png)

区别1：这里是将整个图像输入到CNN网络，得到相应的特征图。 而RCNN是将候选区域输入到网络中。

区别2：没有使用SVM分类器，和专门用于修正的回归器，而是直接使用一个网络。



## 2.2 一次性计算整张图的特征

![image-20220804163633190](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804163633190.png)

（右边的图参考SPPNet）

在训练过程中并不是使用SS算法所提供的所有候选区域，通过SS算法我们大约得到2000个候选框，但在训练过程中其实只需要一小部分，而且所采样的数据分正样本和负样本，正样本——候选框当中确实存在我们所需检测的样本，负样本可以简单理解为背景，里面没有我们所要检测的目标。

如何分正样本、负样本？例子：训练一个猫狗分类器，如果猫的数量远远大于狗的数量，网络不平衡的情况下，网络在预测过程当中会偏向于猫，这样是不对的。

对于每张图片，从2000个候选框当中，采集64个候选区域，64个候选区域当中一部分是正样本，一部分是负样本，正样本的定义是：候选框与真实目标边界框的IoU大于0.5，就认为是正样本，但并不是所有的正样本都会使用，我们只是随机采用一部分。候选框与我们真实目标边界框IoU值是在0.1-0.5之间最大的就认为是负样本。



### 2.2.1 ROI Pooling Layer

​	我们用于训练样本的候选框通过ROI Pooling Layer缩放到统一的尺寸，划分**7*7**=49等分，<u>划分之后，我们对每一个区域进行最大池化下采样（Max Pooling 操作）</u>，得到特征矩阵，这样**可以不限制输入图像的尺寸**。（在RCNN当中是要求图像输入是227 * 227大小的，下图忽略了深度channel）

![image-20220804183958398](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804183958398.png)



### 2.2.2 分类器

​	输出N+1个类别的概率(N为检测目标的种类,1为背景）共N+1个节点。

![image-20220804184636834](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804184636834.png)

​	将一张图输入到CNN网络当中得到特征图，根据映射关系得到每一个候选区域的特征矩阵。这个特征矩阵通过ROI Pooling层统一缩放到指定的尺寸，展平处理，再经过两个全连接层，再ROI feature vector的基础上并连两个全连接层，其中一个全连接层用于目标概率的预测，另一个全连接层用于边界框回归参数的预测。

​	

### 2.2.3 边界框回归器

​	输出对应N+1个类别的候选边界框回归参数(dx, dy, dw, dh)共(N+1)x4个节点

![image-20220804185525824](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804185525824.png)



### 2.3 损失计算

![image-20220804190136388](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804190136388.png)

![image-20220804190203963](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804190203963.png)

p0为候选区域为背景的概率

#### 2.3.1 分类损失

![image-20220804190556089](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804190556089.png)

这个log损失实际是交叉熵损失。 

![image-20220804190919218](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804190919218.png)

oi* 只有在正确标签的索引位置是为1的，其他位置是等于0的。假设当i等于u的时候是正确的标签值，ou*=1，则H = - log(ou)

![image-20220804191353324](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804191353324.png)



#### 2.3.2 边界框回归损失

![image-20220804191757099](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804191757099.png)

![image-20220824221416027](images/image-20220824221416027.png)



https://www.cnblogs.com/wangguchangqing/p/12021638.html

lamda是平衡分类损失和边界框回归损失的平衡系数。

[u>=1] 代表艾弗森括号，如果u>=1，则为1，去计算损失。如果为0，则不计算损失。



![image-20220804193041419](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804193041419.png)



# 三、Faster RCNN

​	2016

​	Faster R-CNN是作者Ross Girshick继Fast R-CNN后的又一力作。同样使用vGG16作为网络的backbone，推理速度在GPU上达到**5fps**(<u>包括候选区域的生成</u>)，准确率也有进一步的提升。在2015年的ILSVRC以及coco竞赛中获得多个项目的第一名。在VOC 2007测试集上达到**73.2%**mAP。

Faster R-CNN算法流程可分为3个步骤

1. 将图像输入网络得到相应的特征图

2. 使用RPN结构生成候选框，将RPN生成的候选框投影到特征图上获得相应的特征矩阵

3. 将每个特征矩阵通过ROI pooling层缩放到7x7大小的特征图，接着将特征图展平通过一系列全连接层得到预测结果

   Faster RCNN 可以看成 RPN + Fast R-CNN

<img src="https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804194303398.png" alt="image-20220804194303398" style="zoom:67%;" />

​	将图像输入到backbone，提取特征，得到特征图，使用RPN结构生成候选框，将生成的候选框投影到相应的特征矩阵，特征矩阵通过ROI Pooling层缩放到统一的大小（7 * 7），展平处理，通过一系列全连接层，得到预测概率和边界框回归参数。



## 3.1 RPN网络结构



![image-20220822161244894](images/image-20220822161244894.png)

​	这里的feature map是上一张图中红色字 feature map 的生成的特征图，使用一个3 * 3**滑动窗口**在feature map上进行滑动，每滑动到一个位置上就生成一个<u>一维的向量</u>，在这个向量的基础上<u>通过两个全连接层分别输出目标概率以及目标边界框回归参数</u>。（2k是针对这k个anchor生成的两个分类概率，一个是背景的概率，一个是前景的概率，所以k个anchors会生成2k个scores；4k是k个anchor生成4个回归参数）



256-d的由来取决于使用哪个backbone？ZF：256	VGG16：512

对于特征图上的每个3x3的滑动窗口，<u>计算出滑动窗口中心点对应原始图像上的中心点</u>（（原图/特征图）* （特征图坐标）），并计算出<u>k个anchor boxes</u>(anchor boxes都是固定的长宽比例，注意和proposal的差异)。



![image-20220825090603466](images/image-20220825090603466.png)

这里的cls只预测是前景还是背景。



![image-20220814152122527](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220814152122527.png)

​	每个位置（每个滑动窗口）在原图上都对应有3*3=9个anchor。

​	**三个尺度**（面积）{128 * 128，256 * 256，512 * 512}

​	**三种比例** {1：1，1：2，2：1}

​	每个位置在原图上都对应有3x3=9 anchor

​	对于 ZF感受野：171，对于VGG感受野：228。

​	通过一个小的感受野去预测一个比他大的目标边界框是有可能的。

​	对于一张1000x600x3的图像，大约有(60x40x9(20k)个anchor，忽略跨越边界的anchor以后，剩下约<u>6k个anchor</u>。对于<u>RPN生成的候选框</u>之间存在大量重叠，基于候选框的cls得分，**采用非极大值抑制**，IoU设为0.7，这样每张图片只剩2k个候选庭。



### 	3.1.1 PRN损失计算

#### 3.1.1.1 分类损失



![image-20220825094735971](images/image-20220825094735971.png)

## ![image-20220825095201810](images/image-20220825095201810.png)

 anchor位置个数，代表中心店个数。



##### 3.1.1.1.1 Softmax Cross Entropy（多分类交叉熵损失）

2k个预测

![image-20220825100017934](images/image-20220825100017934.png)

对于第一个anchor损失为：-log(0.9)

对于第二个anchor损失为：-log(0.2)



##### 3.1.1.1.2 Binary Cross Entropy（二分类交叉熵损失）

![image-20220825100645464](images/image-20220825100645464.png)

PyTorch中实现的Faster RCNN中使用的就是二值交叉熵损失，得到k个预测结果。



#### 3.1.1.2 边界框回归损失

​	与之前Fast rcnn中是一样的。

![image-20220825101531752](images/image-20220825101531752.png)



## 3.2 Faster R-CNN训练

现在使用Faster RCNN的方法是直接采用<u>RPN Loss+ Fast R-CNN Loss</u>的**联合训练方法**

原论文中采用分别训练RPN以及Fast R-CNN的方法

(1) 利用ImageNet预训练分类模型初始化前置卷积网络层参数，并开始单独训练RPN网络参数;
⑵ 固定RPN网络独有的卷积层以及全连接层参数，再利用
lmageNet预训练分类模型初始化前置卷积网络参数，并利用RPN网络生成的目标建议框去训练Fast RCNN网络参数。
⑶ 固定利用Fast RCNN训练好的前置卷积网络层参数，去微调RPN网络独有的卷积层以及全连接层参数。
(4) 同样保持固定前置卷积网络层参数，去微调Fast RCNN网络的全连接层参数。最后RPN网络与Fast RCNN网络共享前置卷积网络层参数，构成一个统一网络。



## 3.3 Faster RCNN框架

![image-20220825102607791](images/image-20220825102607791.png)



## CNN感受野

​		计算Faster RCNN中ZF网络feature map 中3x3滑动窗口在原图中感受野的大小。

![image-20220825091431771](images/image-20220825091431771.png)

​	

















# FPN（Feature Pyramid Networks for Object Detection）

https://arxiv.org/abs/1612.03144

![image-20220804200055496](images/image-20220804200055496.png)

针对目标检测任务cocoAP提升2.3个点pascalAP提升38个点





![image-20220804200136351](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804200136351.png)

(b) faster-rcnn	(c) ssd

（d）将不同的特征层进行融合

![image-20220804200118573](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804200118573.png)

1 * 1 conv —— 原论文中1x1的卷积核的个数为256，即最终得到的特征图的channel都等于256。

1 * 1 conv做的是backbone上，调整不同特征图上的channel。

邻近插值算法

![image-20220907091328098](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907091328.png)



## FPN完整结构



![image-20220804201247219](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220804201247219.png)



![image-20220907092359465](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907092400.png)



![image-20220907092658813](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907092659.png)



torchvision->ops->poolers

![image-20220907092827055](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907092827.png)
