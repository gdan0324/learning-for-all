# EfficientNet V2

<u>2021 CVPR</u>



![image-20220907142222472](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907142223.png)



![image-20220907142240355](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220907142240355.png)

- 引入Fused-MBConv模块
- 引入渐进式学习策略(训练更快)



## 5 实验



![image-20220907144215480](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907144216.png)

​									*Table 7.* **EffificientNetV2 Performance Results on ImageNet** 

​	在EfficientNetV1中作者关注的是**准确率，参数数量以及FLOPs(理论计算量小不代表推理速度快）**，在<u>EfficientNetV2中作者进一步关注模型的训练速度</u>。

![image-20220907145314061](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907145314.png)



![image-20220907145329669](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907145330.png)



## 2. EfficientNet V1 中存在的问题

- 训练图像的尺寸很大时，训练速度非常慢
- 在网络浅层中使用Depthwise convolutions速度会很慢
- 同等的放大每个stage是次优的



### 2.1 训练图像的尺寸很大时，训练速度非常慢

​	针对这个问题一个比较好想到的办法就是降低训练图像的尺寸，之前也有一些文章这么干过。降低训练图像的尺寸不仅能够加快训练速度，还能使用更大的batch_sie.

![image-20220907151335114](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907151335.png)

### 2.2 在网络浅层中使用Depthwise convolutions速度会很慢

​	无法充分利用现有的一些加速器（虽然理论上计算量很小，但实际使用起来并没有想象中那么快）。故引入Fused-MBConv结构。

![image-20220907152009426](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907152010.png)

​	NAS技术进行网络搜索。

​	fused哪几个stage。



### 2.3 同等的放大每个stage是次优的

​	在EfficientNetV1中，每个stage的深度和宽度都是同等放大的。但每个stage对网络的训练速度以及参数数量的贡献并不相同，所以直接使用同等缩放的策略并不合理。在这篇文章中，作者采用了<u>非均匀的缩放策略来缩放模型。</u>



## 3. EfficientNet V2中做出的贡献

​	在之前的一些研究中，主要关注的是准确率以及参数数量(注意，参数数量少并不代表推理速度更快)。但在近些年的研究中，开始关注网络的训练速度以及推理速度(可能是准确率刷不动了)。

- 引入新的网络(EfficientNetv2)，该网络在训练速度以及参数数量上都优于先前的一些网络。
- 提出了改进的**渐进学习方法**，该方法会根据训练图像的尺寸动态调节正则方法(提升训练速度、准确率)
- 通过实验与先前的一些网络相比，训练速度提升11倍，参数数量减少为1/6.8

​	Dropout、Rand Augment、Mixup

![image-20220907153551722](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907153552.png)



## 4. 网络框架

![image-20220907153830012](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907153830.png)

与EfficientNetV1的不同点:

- 除了使用MBConv模块，还使用Fused-MBConv模块≥会
- 使用较小的expansion ratio
- ≥偏向使用更小的kernel_size(3x3)
- ≥移除了EfficientNetv1中最后一个步距为1的stage (v1'的stage8）



### Fused-MBConv模块



![image-20220907155551453](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907155552.png)



![image-20220907155750646](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907155751.png)



![image-20220907155831482](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907155832.png)



### EfficientNetV2其他训练参数

![image-20220907160224909](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907160225.png)



### 渐进式学习策略



![image-20220907160346134](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907160346.png)



​	训练早期使用较小的训练尺寸以及较弱的正则方法weak regularization，这样网络能够快速的学习到一些简单的表达能力。接着逐渐提升图像尺寸，同时增强正则方法adding stronger regularization。这里所说的regularization包括Dropout，RandAugment以及Mixup。

![image-20220907160435243](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907160436.png)



![image-20220907160606109](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907160606.png)



![image-20220907160638372](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907160639.png)



![image-20220907160908915](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907160909.png)
