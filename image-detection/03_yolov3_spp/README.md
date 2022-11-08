# YOLO

先看讲解 -> 读原文 -> 读代码（star多的） -> README.md 跑通代码 -> 结合原论文分析代码（网络搭建、数据预处理、损失计算）

# 1. YOLO v1

![image-20220901093851746](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901095522.png)

对比SSD，推理速度和map都不佳。

1）我们的系统将输入图像划分为 *S* × *S* 网格（grid cell）。如果某个对象的中心落入网格单元，则该网格单元负责检测该对象。

![image-20220901100813391](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901100814.png)

2）每个网格要预测B个bounding box，每个**bounding box**除了要预测位置之外，还要附带预测一个confidence（SSD网络和faster rcnn中没有）值。每个网格还要预测c个类别的分数。

![image-20220901102010657](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901102011.png)

![image-20220901102309644](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901102310.png)

​	这些置信度分数反映了模型对边框包含物体的置信度，也反映了它认为边框预测的准确性。在形式上，我们将置信度定义为 **Pr（Object）∗ IOUtruth pred** （预测目标和真实目标的交并比）

IOUtruth pred —— 预测目标和真实目标的交并比

Pr（Object）—— 0和1两种情况，如果有目标落在网格（等于1），置信度等于这个IOU值；如果没有目标落在网格（等于0），置信度等于0。

![image-20220901105623641](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901105624.png)



![image-20220901110014187](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901110014.png)

每个边界框由5个预测组成：x、y、w、h和置信度。（x、y）坐标表示**相对于网格单元格边界**的方框的中心（数值是0-1之间）。并将其宽度和高度**相对于整个图像**进行了预测（数值是0-1之间）。最后，置信度预测表示预测框和任意ground truth 框之间的IOU。

yolo 没有anchor的概念，SSD、faster rcnn 是预测anchor的 x、y、w、h 的回归参数，而 yolo 是直接得到 x、y、w、h 坐标值。



在测试时，我们将条件类概率和个体box置信度预测相乘，这给了我们每个盒子的特定类别的置信度分数。这些分数既包含了该类出现在框中的概率，也编码了预测的框与对象的匹配程度。

![image-20220901110902646](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901110903.png)



## 1.2 网络结构

![image-20220901111527436](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901111528.png)



![image-20220901142847858](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901142848.png)



## 1.3 损失函数



![image-20220901143329485](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901143330.png)

可以通过下图看到，相同的偏移量，对于小目标来讲预测很差，但是大目标来说预测很好。所以上面的公式中使用了开根号处理。

![image-20220901144054013](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901144054.png)

​	可以看到小目标相减的结果更大，大目标相减的结果差距更小。

![image-20220901151046617](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901151047.png)



##  1.4 存在的问题

1）当一些小目标聚集在一起的时候，检测效果就比较差。

2）很难推广到新的或不寻常的长宽比或配置的对象。

3）主要错误来源是预测不正确。（因为不是基于anchor的回归预测）

![image-20220901152657971](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901152658.png)

# 2. YOLO v2

2017 CVPR

![image-20220901154235930](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901154236.png)



![image-20220901154309578](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901154351.png)

​	通过YOLOv2 的模型框架，使用 PASCAL VOC 和 IMAGE NET数据集进行联合训练，最终能跟检测的种类个数能够超过9000，

![image-20220901154842112](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901154842.png)

横坐标——每秒钟能跟检测的图片帧数

纵坐标——在PSACAL VOC 2007 test上的map

![image-20220901155646988](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901155647.png)

## 2.1 YOLO v2 中的各种尝试 Better章节

1. Batch Normalization

   在yolo v1中作者在搭建网络过程中没有使用BN层，在yolo v2中作者在每个BN层后面加了BN层，加了BN层后对于训练收敛有非常大的帮助，同时也减少了一系列所需要的正则化处理。相比没有使用BN之前，能跟达到将近2%mAP的提升，使用了BN层后可以移除dropout层（防止过拟合）

   

   ![image-20220901160843715](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901160844.png)

   

2. High Resolution Classifier（更好分辨率的分类器）

   在 yolo v1 中训练 backbone 的时候，采用 224 × 224 的图像作为网络输入尺寸，yolo v1也是使用 224 × 224 在 Image Net 上进行预训练。在 yolo v2 网络当中采用更大的输入尺寸 448 × 448 的分辨率。

   采用更大分辨率的分类器有什么好处？

   采用更大分辨率的分类器能跟达到将近4%mAP的提升。

   ![image-20220901161410645](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901161411.png)

   

3. Convolutional With Anchor Boxes

   在yolo v1中，是直接预测目标高度、宽度、中心坐标，在yolo v1的论文之后，作者在文中说采用这种方式的效果是很差的，所以作者尝试**基于anchor的目标边界框进行预测**。

   **预测偏移量而不是坐标，简化了问题，并使网络更容易学习。**

   

   ![image-20220901162912887](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901162913.png)

   

   在没有锚框的情况下，我们的中间模型得到69.5 mAP，召回率为81%。

   使用锚框，我们的模型得到69.2 mAP，召回率为88%。

   虽然 mAP 减少了，但**召回率的增加意味着我们的模型有更大的改进空间。**

   ![image-20220901164533532](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901164534.png)

4. Dimension Clusters（anchor的聚类）

   之前在使用faster-rcnn和ssd的时候，其实作者并没有明确给出作者为什么要采用作者给定的那些预设anchor/default box，虽然在ssd中有给公式，但并不明确，在faster-rcnn中没有给出如何得到anchor的尺寸，只能是根据工程经验设定的。

   ![image-20220901164747482](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901164748.png)

   

   采用k-means聚类的方法，获取anchor，作者称其为anchor，这里的priors与faster-rcnn中的anchor、ssd中的default box是一样的，本篇论文中给出了生成priors的方法。也就是基于训练集中所有目标的边界框**采用 k-means 聚类**的方法去获得相应的priors。

   网络可以学会适当地调整 box ，但如果我们为网络选择更好的priors，我们可以使网络更容易学会预测良好的检测。在yolo v3中所有的priors也是通过聚类的方法得到的。

   ![image-20220901164916124](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901164916.png)

5. Direct location prediction

   

   当使用YOLO使用锚框时，我们会遇到第二个问题：**模型不稳定性，特别是在早期迭代中**。

   大多数的不稳定性来自于<u>预测目标边界框的中心坐标(x，y)导致的</u>。

   在faster-rcnn中关于预测目标边界框（x,y）坐标的公式，

   ![faster-rcnn论文中](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901204341.png)

   x_a —— anchor的中心点坐标

   w_a —— anchor的宽度

   t_x —— 中心坐标x的回归参数（关于anchor的偏移量）

   （下面的公式是错误的，应该是加号）

   ![image-20220901170603034](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901170603.png)

   

   上面**这个公式没有受到限制**，所以基于anchor预测的目标可能出现的图像中的任意一个地方，例如，tx=1的预测将使盒子向右移动锚盒的宽度，tx=−1的预测将其向左移动相同的数量。

   ![image-20220901204921030](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901204921.png)

   

   

   下面的例子设置比较极端，首先呢，我们假设对于每个网格 grid cell ，我们将 anchor 或者 prior 设置在每个 grid cell 的左上角，那么通过我们网络的预测之后呢，我们会得到一个关于X坐标和Y坐标的一个回归参数。由于甚至我们的公式并没有限制我们 t_x 和 t_y 的值，那么我们将 anchor 的中心坐标，加上我们预测的回归参数之后呢，它可能出现在我们图像的任意一个地方，那么就比如说，他可能加上偏移量之后就跑到右下角来了，那么很明显，这并不是我们想要看到的，因为在这个区域，如果他有目标，那么也是由最后的 grid cell 的 anchor 来进行预测的，也轮不到偏移后右下角的anchor来进行预测，所以这就会导致我们网络在训练过程中出现不稳定的一种情况。在yolo v2 论文当中，作者为了改善这种情况，作者就采用了另外一种预测的方法。

   ![image-20220901205421884](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901205422.png)

   

   假设将anchor设置在每个 grad cell 的左上角，假设坐标是(*c**x**, cy*) ，而且预测的边界框prior的宽度和高度为*pw*、*ph* 。作者在这里就将 t_x 放到一个sigmoid激活函数当中，所以通过这种方式之后，t_x这个偏移量只会在0到1之间，上图中左上角的anchor偏移之后，只会在一个grad cell之间，不会超过 grad cell ，所以对我们最终预测目标的中心点进行限制，每个anchor(prior)去负责预测目标中心落在某个grid cell区域内的目标。由于我们约束了位置预测，参数化更容易学习，使网络更加稳定。使用聚类得到的 边界框prior 比 直接预测边界框中心位置（faster rcnn），为YOLO提高了近5%。

   ![image-20220901221037616](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220901221037616.png)

   上面的 t_o 就是confidence，就是所想要检测的目标和真实目标之间的IOU值。

   

6. Fine-Grained Features

   更底层的特征信息，会包含更多图像细节。

   这些细节在检测小目标的时候所需要的，这也就是我们为什么我们直接拿高层的目标进行检测效果会很差，所以这里作者就**将高层的信息与相对低层的信息进行融合**。

   在 YOLO v2 中最终预测的特征图大小是 13×13 。作者去融合相对更低层一点的26×26分辨率的特征图。

   融合的方式：通过 **passthrough layer** 来进行实现的，通过这个 layer 能跟将相对底层的特征图与高层特征图进行融合，从而提升检测小目标的效果。

   ![image-20220901223723910](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901223724.png)

   

   

   ![image-20220901230116463](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901230117.png)!

   

   passthrough layer 是如何将两个不同尺度的特征图进行融合？

   （像 swin 中的 Patch Merging）

   ![image-20220901230247430](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901230248.png)

   模型提升效果大概有1个百分点

   

7. Multi-Scale Training

   YOLOv2将图片缩放到不同尺度，以提升YOLOV2的鲁棒性，来训练我们的模型。

   我们不是固定输入图像的大小，而不是每隔几次迭代就改变一次网络。

   我们的网络每10 batch 就随机选择一个新的图像尺寸。由于yolov2网络的缩放因子是32（416/13），所以我们输入网络的尺寸都是以32的倍数中提取：{320,352，...，608}。因此，输入网络最小的尺寸是320×320，最大的尺寸是608×608。也就是我们每迭代10个batch，就改变我们输入网络的图像大小，选择的尺寸是在上面的集合中选取，将网络的大小调整到这个维度，并继续进行训练。

​	![image-20220901231320579](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901231321.png)

## 2.2 Faster章节

​	Backbone: Darknet-19

​	

​	![image-20220901232500671](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901232501.png)

​	

![image-20220901232641104](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901232641.png)

表6.Darknet-19 只需要 55.8 亿次操作来处理一幅图像，但在 ImageNet 上达到了72.9%的 top-1 精度和91.2% 的 top-5 精度。（224 * 224，用这个尺度是因为前面的网络都用这个尺度，所以是为了对比。）

但在实际使用过程中，采用 448 * 448 作为高分辨率作为分类器进行训练，高分辨率的分类器肯定能达到更高的准确率。



## 2.3 YOLO v2模型框架

​	我们训练网络的标准ImageNet 1000类分类数据集160时代使用随机梯度下降开始学习率0.1，多项式率衰减为4，权重衰减0.0005和动量0.9使用暗网神经网络框架[13]。在训练过程中，我们使用标准的数据增强技巧，包括随机作物、旋转、色调、饱和度和曝光转移。

![image-20220901235833150](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901235833.png)

​	检测培训。我们修改了这个检测网络，**删除最后一个卷积层**（包括最后的Avgpool、Softmax），<u>添加三个3×3卷积层，每个层有1024个滤波器</u>，<u>然后是最后一个1×1卷积层，包含我们需要检测的输出数量</u>。对于VOC，我们预测有**5**个bounding box，每个bounding box有5个参数，每个bounding box有20个类，所以有125（（5+20） * **5**）个过滤器。从最后的**3×3×512** 层到第二个到最后一个卷积层之间，我们还添加了一个 passthrough layer ，以便我们的模型可以使用细粒度特征。

![image-20220901235929305](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220901235930.png)

## 2.4 网络训练细节

   论文没有描述。

​	如何匹配正负样本? 如何计算误差?

​	我们使用的权重衰减为0.0005，动量为0.9。我们使用**类似于YOLOV1和SSD的数据增强**，<u>使用随机裁剪，颜色变化</u>等。我们对COCO和VOC采用<u>相同的培训策略</u>。

![image-20220902000920836](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902000921.png)

# 3. YOLO v3

https://github.com/ultralytics/yolov3

2018年

![image-20220902084754524](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902084755.png)

![image-20220902084838457](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902084839.png)

## 3.1 COCO AP

Yolo v3 在当前的一系列检测网络的对比。

![image-20220902084945959](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902084946.png)



## 3.2 COCO AP IOU=0.5（PASCAL VOC的指标）

![image-20220902085210306](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902085211.png)

## 3.3 网络

修改了Backbone，没有maxpooling层，在Darknet-53中所有的下采样都是通过卷积层来实现的。采用的卷积核的个数比resnet少一些，所以参数就会少一些，运算量也会少一些。

https://blog.csdn.net/qq_37541097/article/details/81214953

![image-20220902085457986](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902085458.png)



![image-20220902085520474](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902085521.png)



![image-20220902093016883](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902093017.png)



![image-20220902093121167](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902093121.png)

​	我们仍然使用k-means聚类来确定我们的边界框先验（anchor）。我们只是任意地选择了9个聚类和3个尺度，然后按尺度均匀地划聚类。在COCO数据集上，9个聚类分别为：（10×13）、（16×30）、（33×23）、（30×61）、（62×45）、（59×119）、（116×90）、（156×198）、（373×326）



![image-20220902093313405](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902093314.png)

​	YOLOv3预测了3个不同尺度的边框。我们的系统使用类似的概念从金字塔网络[8]中提取特征。从我们的基本特征提取器中，我们添加了几个卷积层。最后一个预测了一个三维张量编码边界框、客观性和类预测。在我们的COCO[10]实验中，我们预测了每个尺度上的3个边框，所以张量是 N×N×[3∗（4+1+80）]，对于4个边界盒子偏移，1个客观预测和80个类预测。（N指的就是：13、26、52）

![image-20220902093705322](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902093706.png)



![image-20220902100903523](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902100904.png)

## 3.4 目标边界框的预测

在yolo v3中关目标中心点的回归参数并不是相对于anchor，而是**相对于cell的左上角点**。

在yolo v3中有三个预测特征层，每个预测特征层又采用了三个不同的模板，

虚线矩形框对应的就是anchor。

蓝色矩形框：网络最终预测的目标的位置以及大小。

o(x)=Sigmoid(x)

c_x —— 左上角点的x坐标

我们这里所计算得到的bx和by它的范围都是在这个grid cell之间的。

所以我们预测的边界框的中心点是被限制在当前这个grid cell之间的。能跟加快收敛

![image-20220902101030992](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902101031.png)

采用与YOLO v2当中一样的预测方式。

![image-20220902101125530](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902101126.png)



## 3.5正负样本匹配

​	

![image-20220902111802675](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902111803.png)

​	我们的系统**只为每个 ground truth 对象分配一个 bounding box prior** 。那么也就是说针对每一个gt而言都会分配一个正样本，也就是说一张图片它有几个gt目标，就有几个正样本。分配原则：就是将与我们gt重合程度最大的bounding box player作为正样本。

​	对于bounding box与ground truth也重合，但IOU不是最大，但确实超过某一个阈值，论文中说会直接丢弃这个预测结果。那么也就是说如果他不是和我们gt重合度最高的，但是它的重合度又大于指定某一阈值，比如说我们这里写的是0.5，那么这个bounding box prior既不是正样本也不是负样本，直接丢弃，剩下的样本认为是负样本。如果一个边界框先验没有分配给一个ground truth，也就是当前bounding box不是正样本的话，它既没有定位损失也没有类别损失，只有confidence score。



### 3.5.1 Ul 版源码当中正样本匹配准则

针对每一个预测特征层而言，都会采用三个不同的anchor模板。接下来我们就将每一个 gt 和每一个anchor模板去进行iou的计算，左上角重合去计算iou。选择iou>0.3，接下来我们再将gt，映射到我们的grid网格。

（如果三个iou都>0.3，那么这三个anchor都会变为正样本，以扩充正样本数量。）

![image-20220902114831288](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902115025.png)

## 3.6 U版损失计算



![image-20220902115821225](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902115822.png)

### 3.6.1 置信度损失

​	二值交叉熵损失

![image-20220902115937180](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902115937.png)

![image-20220902120016364](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902120017.png)

​	YOLOv3使用**逻辑回归**预测每个边界框的 objectness score （置信度）。

​	N——正负样本的总和



### 3.6.2 类别损失

​	采用二值交叉熵

![image-20220902122334183](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902122335.png)



![image-20220902122415928](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902122416.png)



![image-20220902144334099](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902144334.png)

​	能够发现这三个值的和并不是等于一的，预测是独立的，不是使用softmax cross entropy，也就是没有经过softmax处理。在我们采用二值交叉熵损失的情况时，每个类别的概率是相互独立的。



### 3.6.3 定位损失

t_x：它对应的就是网络预测的有关中心点的x方向的回归参数

g_x^：真实回归参数

t_y：对应的就是网络预测的有关中心点y方向的偏移量

g_y^：真实的中心点在y方向的偏移量

t_w：网络预测针对宽度的回归参数

g_w^：真实的宽度回归参数

t_h：网络预测的有关高度方向的回归参数

g_h^：真实的高度方向上的回归参数

对每一个正样本都会去计算定位损失，再求和，再除以正样本的总个数。

![image-20220902190514443](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902190515.png)



![image-20220902190454305](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902190455.png)



# 4. YOLO v3 SPP

YOLO SPP中使用的tricks：Mosaic图像增强、SPP模块、CIOU Loss

Focal loss：作者实现了，但是默认是没有使用的，因为使用的效果并不是很好，因此默认不启用。

![image-20220902193129541](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902193130.png)



## 4.1 Mosaic 图像增强

增加数据的多样性、增加目标个数、BN能一次性统计多张图片的参数（变相的增加了输入网络的Batchsize，比如输入一张由四张图像拼接的图像等效并行输入四张(batch_size=4原始图像））

多张网络拼接在一起输入网络的训练过程，源码中是4张图片拼接在一起进行预测。

## 4.2 SPP模块

![image-20220902194443199](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902194444.png)



<img src="https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902200324.png" alt="image-20220902200323386" style="zoom: 67%;" />

在每一个特征预测层前加一个SPP结构，在输入网络尺寸较小的时候只加一个SPP结构更好，在输入网络尺寸较大的时候，用三个SPP结构较好。

![image-20220902200402808](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902200403.png)

## 4.3 Loss

DIOU Loss 和 CIOU Loss 是在同一篇论文中讲到的。

![image-20220902202338034](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902202338.png)

### 4.3.1 IoU Loss

L2损失都是一样的，因此引入了IoU Loss。

![image-20220902203442686](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902203443.png)

### 4.3.2 GIoU Loss



![image-20220902205118635](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902205119.png)

缺点：

​	当两个框是垂直或者是水平的时候，GIoU退化成IoU。

![image-20220902205238491](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902205239.png)

### 4.3.3 DIoU Loss

![image-20220902213743977](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902213744.png)



### 4.3.4 CIoU Loss

​	现在有很多后处理算法当中将 IoU 替换成 DIoU。

​	![image-20220902215426279](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902215427.png)



![image-20220902215453061](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902215453.png)



### 4.3.5 Focal Loss

主要是针对 One-stage object detection model （SSD、YOLO）

One-stage中都会面临正负样本不平衡的问题。

一张图像中能够匹配到目标的候选框（正样本)个数一般只有十几个或几十个，而没匹配到的候选框(负样本)大概有**10^4-10^5**个。

在这10^4-10^5 个未匹配到目标的候选框中大部分都是简单易分的负样本(对训练网络起不到什么作用，但由于数量太多会淹没掉少量但有助于训练的样本)。

![image-20220902215641054](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902215641.png)

hard negative mining 负硬挖掘：选取损失比较大的负样本来训练网络。

OHEM——采用hard negative mining

![image-20220902221915442](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902221916.png)

Focal Loss 的设计是为了应对 one-stage 目标检测网络当中正负样本极度不平衡的情况，也就是前景和背景极度不平衡的状况（1:1000），对于普通的 cross entropy loss （二值交叉熵损失）：

log就是ln

![image-20220902222347670](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902222348.png)



![image-20220902222512034](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902222512.png)

进一步简化：

![image-20220902222548676](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902222549.png)



α 是针对正样本而言的，系数是α；对于负样本而言系数是1-α.

这里的α不是正负样本的比例，只是一个超参数，平衡正负样本的权重。

![image-20220902223210639](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902223211.png)

​	当引入α这个超参数的时候，α是平衡正负样本的权重。但他不能区分哪些是容易的样本，哪些是困难的样本。所以作者就在本文中提出了一个损失函数，这个损失函数能跟降低简单样本的权重，因此我们能跟聚焦于训练难分类的负样本。

​	![image-20220902223659476](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902223700.png)



![image-20220902223729579](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902223730.png)

![image-20220902224220487](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902224221.png)

最终Focal loss的表达式：

​	多一个超参数 α_t

![image-20220902224407704](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902224408.png)

更有助于学习难学习的样本，对于简单的样本就降低损失权重。

![image-20220902225518879](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220902225519.png)

可能要调参调的好才能更好匹配Focal loss。

缺点：易受噪声干扰。（标注错误的情况下）

# 5. YOLOv3 SPP 源碼

该项目源自[ultralytics/yolov3](https://github.com/ultralytics/yolov3)

## 5.1 文件结构

```
  ├── cfg: 配置文件目录
  │    ├── hyp.yaml: 训练网络的相关超参数
  │    └── yolov3-spp.cfg: yolov3-spp网络结构配置 
  │ 
  ├── data: 存储训练时数据集相关信息缓存
  │    └── pascal_voc_classes.json: pascal voc数据集标签
  │ 
  ├── runs: 保存训练过程中生成的所有tensorboard相关文件
  ├── build_utils: 搭建训练网络时使用到的工具
  │     ├── datasets.py: 数据读取以及预处理方法
  │     ├── img_utils.py: 部分图像处理方法
  │     ├── layers.py: 实现的一些基础层结构
  │     ├── parse_config.py: 解析yolov3-spp.cfg文件
  │     ├── torch_utils.py: 使用pytorch实现的一些工具
  │     └── utils.py: 训练网络过程中使用到的一些方法
  │
  ├── train_utils: 训练验证网络时使用到的工具(包括多GPU训练以及使用cocotools)
  ├── weights: 所有相关预训练权重(下面会给出百度云的下载地址)
  ├── model.py: 模型搭建文件
  ├── train.py: 针对单GPU或者CPU的用户使用
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── trans_voc2yolo.py: 将voc数据集标注信息(.xml)转为yolo标注格式(.txt)
  ├── calculate_dataset.py: 1)统计训练集和验证集的数据并生成相应.txt文件
  │                         2)创建data.data文件
  │                         3)根据yolov3-spp.cfg结合数据集类别数创建my_yolov3.cfg文件
  └── predict_test.py: 简易的预测脚本，使用训练好的权重进行预测测试
```

## 5.2 训练数据的准备以及目录结构

* 这里建议标注数据时直接生成yolo格式的标签文件`.txt`，推荐使用免费开源的标注软件(支持yolo格式)，[https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)
* 如果之前已经标注成pascal voc的`.xml`格式了也没关系，我写了个voc转yolo格式的转化脚本，4.1会讲怎么使用
* 测试图像时最好将图像缩放到32的倍数
* 标注好的数据集请按照以下目录结构进行摆放:

```
├── my_yolo_dataset 自定义数据集根目录
│         ├── train   训练集目录
│         │     ├── images  训练集图像目录
│         │     └── labels  训练集标签目录 
│         └── val    验证集目录
│               ├── images  验证集图像目录
│               └── labels  验证集标签目录            
```

## 5.3 利用标注好的数据集生成一系列相关准备文件，为了方便我写了个脚本，通过脚本可直接生成。也可参考原作者的[教程](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)

```
├── data 利用数据集生成的一系列相关准备文件目录
│    ├── my_train_data.txt:  该文件里存储的是所有训练图片的路径地址
│    ├── my_val_data.txt:  该文件里存储的是所有验证图片的路径地址
│    ├── my_data_label.names:  该文件里存储的是所有类别的名称，一个类别对应一行(这里会根据`.json`文件自动生成)
│    └── my_data.data:  该文件里记录的是类别数类别信息、train以及valid对应的txt文件
```

### 5.3.1 将VOC标注数据转为YOLO标注数据(如果你的数据已经是YOLO格式了，可跳过该步骤)

* 使用`trans_voc2yolo.py`脚本进行转换，并在`./data/`文件夹下生成`my_data_label.names`标签文件，
* 执行脚本前，需要根据自己的路径修改以下参数

```python
# voc数据集根目录以及版本
voc_root = "./VOCdevkit"
voc_version = "VOC2012"

# 转换的训练集以及验证集对应txt文件，对应VOCdevkit/VOC2012/ImageSets/Main文件夹下的txt文件
train_txt = "train.txt"
val_txt = "val.txt"

# 转换后的文件保存目录
save_file_root = "/home/wz/my_project/my_yolo_dataset"

# label标签对应json文件
label_json_path = './data/pascal_voc_classes.json'
```

* 生成的`my_data_label.names`标签文件格式如下

```text
aeroplane
bicycle
bird
boat
bottle
bus
...
```

### 5.3.2 根据摆放好的数据集信息生成一系列相关准备文件

* 使用`calculate_dataset.py`脚本生成`my_train_data.txt`文件、`my_val_data.txt`文件以及`my_data.data`文件，并生成新的`my_yolov3.cfg`文件
* 执行脚本前，需要根据自己的路径修改以下参数

```python
# 训练集的labels目录路径
train_annotation_dir = "/home/wz/my_project/my_yolo_dataset/train/labels"
# 验证集的labels目录路径
val_annotation_dir = "/home/wz/my_project/my_yolo_dataset/val/labels"
# 上一步生成的my_data_label.names文件路径(如果没有该文件，可以自己手动编辑一个txt文档，然后重命名为.names格式即可)
classes_label = "./data/my_data_label.names"
# 原始yolov3-spp.cfg网络结构配置文件
cfg_path = "./cfg/yolov3-spp.cfg"
```

## 5.4 预训练权重下载地址（下载后放入weights文件夹中）：

* `yolov3-spp-ultralytics-416.pt`: 链接: https://pan.baidu.com/s/1cK3USHKxDx-d5dONij52lA  密码: r3vm
* `yolov3-spp-ultralytics-512.pt`: 链接: https://pan.baidu.com/s/1k5yeTZZNv8Xqf0uBXnUK-g  密码: e3k1
* `yolov3-spp-ultralytics-608.pt`: 链接: https://pan.baidu.com/s/1GI8BA0wxeWMC0cjrC01G7Q  密码: ma3t
* `yolov3spp-voc-512.pt` **(这是我在视频演示训练中得到的权重)**: 链接: https://pan.baidu.com/s/1aFAtaHlge0ieFtQ9nhmj3w  密码: 8ph3

## 5.5 数据集，本例程使用的是PASCAL VOC2012数据集

* `Pascal VOC2012` train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* 如果不了解数据集或者想使用自己的数据集进行训练，请参考我的bilibili：https://b23.tv/F1kSCK

## 5.6 使用方法

* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要使用单GPU训练或者使用CPU训练，直接使用train.py训练脚本
* 若要使用多GPU训练，使用`python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 训练过程中保存的`results.txt`是每个epoch在验证集上的COCO指标，前12个值是COCO指标，后面两个值是训练平均损失以及学习率



# 6. YOLO v4 

2020CVPR

![image-20220910101712977](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910101713.png)



![image-20220910101729737](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910101730.png)



![image-20220910102416599](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910102417.png)



## 6.1 Backbone_CSPDarknet53

![image-20220910102504374](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910102505.png)



![image-20220910103040286](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910103041.png)



![image-20220910103349551](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910103350.png)



![image-20220910104121272](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910104122.png)



# YOLO v5



![image-20220910105136055](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910105136.png)



![image-20220910105148403](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910105149.png)



![image-20220910105405511](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910105406.png)



## 1.网络结构

- Backbone: New CSP-Darknet53
- Neck: SPPF, New CSP-PAN
- Head: YOLOv3 Head

![image-20220910111119912](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910111120.png)



## 2.数据增强



## 3.训练策略

32整数倍

差距大的话可以启用autoAnchor

![image-20220910111718491](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910111719.png)



## 4.其他





### 4.1损失计算

![image-20220910111942599](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910111943.png)



### 4.2平衡不同尺度损失4.3消除Grid敏感度

![image-20220910112618499](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220910112619.png)



### 4.4匹配正样本(Build Targets)



# 6. 代码中网络层结构



![image-20220903163157980](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220903163159.png)

捷径分支

![image-20220903163625656](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220903163626.png)



![image-20220903164722727](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220903164723.png)

不是预测层，而且预测层之后的的。

![image-20220903165223397](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220903165224.png)

