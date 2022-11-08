# EfficientNet V1

Google发表

博文：https://blog.csdn.net/qq_37541097/article/details/114434046

![image-20220906083845335](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906083846.png)



![image-20220906083900893](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906083901.png)

​	在论文中提到，本文提出的EfficientNet-B7在lmagenet top-1上达到了当年最高准确率84.3%,与之前准确率最高的GPipe相比，参数数量仅为其1/8.4，推理速度提升了6.1倍。

- **同时**探索输入**分辨率，网络的深度、宽度**的影响



![image-20220906084956812](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906084957.png)



![image-20220906085025417](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906085026.png)

- 根据以往的经验，增加网络的深度depth能够得到更加丰富、复杂的特征并且能够很好的应用到其它任务中。但网络的深度过深会面临梯度消失，训练困难的问题。
- 增加网络的width能够获得更高细粒度的特征并且也更容易训练，但对于width很大而深度较浅的网络往往很难学习到更深层次的特征。
- 增加输入网络的图像分辨率能够潜在得获得更高细粒度的特征模板，但对于非常高的输入分辨率，准确率的增益也会减小。并且大分辨率图像会增加计算量。



## 1.1 EfficientNet-B0

![image-20220906085405853](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906085406.png)

​	Layer：重复堆叠次数



##  1.2 MBConv



![image-20220906085819907](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906085820.png)

​	这里DW卷积的卷积核可能是3x3或5x5，具体参数可以看刚刚给的表格。

注意，在源码中只有使用到shortcut连接的MBConv模块才有Dropout层

- 第一个升维的1x1卷积层，它的卷积核个数是输入特征矩阵channel的n倍

- 当n=1时，不要第一个升维的1x1卷积层，即Stage2中的MBConv结构都没有第一个升维的1x1卷积层（这和MobileNetV3网络类似)

- 关于shortcut连接，仅当输入MBConv结构的特征矩阵与输出的特征矩阵shape相同时才存在

  

## 1.3 SE模块

![image-20220906090738346](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906090739.png)

SE模块如图所示，由一个全局平均池化，两个全连接层组成。

第一个全连接层的节点个数是**输入该MBConv特征矩阵channels的1/4**，且使用Swish激活函数。

第二个全连接层的节点个数等于Depthwise Conv层输出的特征矩阵channels，且使用Sigmoid激活函数。



![image-20220906091105802](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906091106.png)

- width_coefficient代表channel维度上的倍率因子，比如在EfficientNetBO中Stagel的3x3卷积层所使用的卷积核个数是32，那么在B6中就是32×1.8=57.6接着取整到离它最近的8的整数倍即56，其它Stage同理。
- depth_coefficient代表depth维度上的倍率因子（仅针对Stage2到Stage8)，比如在EfficientNetBO中Stage7的L=4，那么在B6中就是4×2.6=10.4，接着向上取整即11.



![image-20220906092429074](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906092429.png)

缺点：占显存

