# ShuffleNet V1、V2

代码有点看不进去

**ShufflfleNet: An Extremely Effificient Convolutional Neural Network for Mobile** **Devices**

发表于2017

# 1 ShuffleNet  V1



![image-20220905142840836](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905142841.png)

网络中的亮点:

- 提出了 channel shuffle 的思想
- ShuffleNet Unit 中全是GConv和DWConv



## 1.1 实验

![image-20220905143254245](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905143255.png)

​	在移动设备上的实际推理时间（数量越小表示性能越好）。该**平台基于单一的高通骁龙820处理器**。所有的结果都用单线程来进行评估。



## 1.2 Channel ShuffleNet

GConv虽然能够减少参数与计算量，但GConv中不同组之间信息没有交流。

![image-20220905144010419](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905144011.png)

​	



![image-20220905145647016](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905145647.png)

​	例如，在ResNeXt[40]中，只有3个×3层配备了组卷积。因此，对于ResNeXt中的每个残差单元，1*1普通卷积占计算量的93.4%。



![image-20220905151348431](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905151349.png)

图2。ShuffleNet Unit. a)具有 dw 卷积的单元；b) 带GConv和 channel shuffle 的单元(stride=1)；c) 带 stride=2 的 shuffle Unit。



![image-20220905154134559](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905154135.png)

表1.ShuffleNet 的架构。复杂性通过FLOPs来评估，即浮点乘法-加法的数量。注意，对于阶段2，我们没有在第一个1*1卷积上应用组卷积，因为输入信道的数量相对较小。



![image-20220905154716333](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905154717.png)

​	每个阶段的第一个构建块应用于 **stride=2** 。

​	一个阶段内的其他超参数保持不变，**对于下一个阶段，输出通道将增加一倍**。

​	与 resnet[9] 类似，我们<u>将 bottleneck 的数量设置为输出通道的1/4</u>.



## FlOAP 计算

![image-20220905155903662](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905155904.png)

DW卷积是GConv的特殊情况

![image-20220905155929560](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905155930.png)



# 2 ShuffleNet V2

2018

![image-20220905164000028](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905164000.png)

**FLOPS:** 全大写，指每秒浮点运算次数，可以理解为计算的速度。是衡量硬件性能的一个指标。(硬件)
**FLOPs :** s小写，指**浮点运算数**，理解为<u>计算量</u>。可以用来衡量算法/模型的复杂度。（模型)在论文中常用GFLOPs ( 1 GFLOPs = 10^9 FLOPs )

论文：计算复杂度不能只看FLOPs -> 提出如何设计高校网络准则 -> 提出新的block设计

![image-20220905164908936](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905164909.png)

​	首先，其他对推理速度有相当大影响的重要因素没有被 **FLOPs** 考虑在内。比如，*memory access cost* (**MAC**)（内存访问时间成本），在诸如组卷积等某些操作中，这种成本构成了运行时的很大一部分。另一个是**并行度**。在相同的流量下，并行度高的模型可以比其他并行度低的模型快得多。

​	其次，根据平台的不同，具有相同流程的操作可能有不同的运行时间。

![image-20220905165745805](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905165746.png)

​	Conv 虽然这部分消耗了大部分时间，但其他操作包括数据I/O、数据 ShuffleNet 和 element-wise 操作(增加张量、ReLU等)也占用了相当多的时间。因此，flops对实际运行时的估计还不够足够准确。

![image-20220905165339651](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905165340.png)



## **2.1 several practical guidelines for efficient network architecture design**

**针对高效的网络架构设计的几个实用的指导方针**

**G1) Equal channel width minimizes memory access cost (MAC).**

**相等的信道宽度可尽量降低内存访问成本(MAC)。**

当卷积层的输入特征矩阵与输出特征矩阵channel相等时MAC最小(保持FLOPs不变时)

![image-20220905171040478](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905171041.png)



![image-20220905171141060](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905171141.png)



![image-20220905172204606](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905172205.png)



![image-20220905172333269](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905172334.png)

​	通过重复**堆叠**10个block来构建一个基准网络。每个块包含两个卷积层。第一个包含输入通道c1和输出通道c2，第二个则相反。<u>表1报告了在固定总流量的同时</u>，通过改变比率c1：c2来实现的运行速度。很明显，当c1：c2接近1：1时，MAC变小，网络评估速度更快。



**G2) Excessive group convolution increases MAC.**

当GConv的groups增大时(保持FLOPs不变时)，MAC也会增大

![image-20220905174004859](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905174005.png)

在 floap 不变的情况下，改变 g 的数值。

![image-20220905174807066](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905174807.png)



**G3) Network fragmentation reduces degree of parallelism.** 

​	网络设计的碎片化程度越高，速度越慢

![image-20220905180658938](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905180659.png)

​	虽然这种碎片化的结构已被证明有利于准确性，但它可能会降低效率，因为它<u>对像GPU这样具有强大并行计算能力的设备不友好</u>。它还引入了额外的开销，如内核启动和同步。

![image-20220905180434578](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905180435.png)



![image-20220905180919243](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905180920.png)



**G4) Element-wise operations are non-negligible.**

Element-wise 操作带来的影响是不可忽视的

![image-20220905192853357](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905192854.png)

​	如图2所示，在像[15,14]这样的轻量级模型中，元素级操作只占用了相当多的时间，特别是在GPU上。在这里，Element-wise 操作符包括ReLU、AddTensor、AddBias等。它们的 FLOPs 很小，但相对较大的MAC。特别地，我们可以把了DW卷积[12,13,14,15]看成一个 element-wise 操作，因为它也有一个很高的MAC/FLOPs比率。

![image-20220905192950832](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905192951.png)



![image-20220905193650509](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905193651.png)

我们观察到，在删除ReLU和short-cut后，在GPU和ARM上都获得了大约20%的加速。

![image-20220905193536717](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905193537.png)

可以说明，使用ReLU、short-cut更耗时。



![image-20220905194131728](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905194132.png)

我们得出结论，一个高效的网络架构应该：

1) 使用**“平衡”卷积**（等 channel 宽度）；

2) 了解使用组卷积的计算成本；（增加group数虽然可以减少参数、FLOPs、提高accuracy，但是会增加计算成本）

3) 降低碎片化程度；

4) 减少 element-wise operations。

   

## 2.2 设计ShuffleNet V2

​	

![image-20220905195001684](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905195002.png)

​	After convolution, the two branches are concatenated. So, the number of channels keeps the same (**G1**). The same “channel shuffle” operation as in [15]	

​	从每个block单元开始，将输入的 feature channel 分成两个 branch ，分别是 **c−c‘ 和 c’** channel。在G3之后，有一个分支减少碎片化程度。另一个分支由三个卷积组成，它们具有相同的输入和输出通道，以满足G1。这两个是1*1卷积<u>不再使用 group 卷积</u>，遵循G2。卷积之后，通过concatenated进行拼接，能跟满足G1准则。

![image-20220905195139587](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905195140.png)

a) ShuffleNet v1：stride=1

b) ShuffleNet v1：stride=2

c) ShuffleNet v2：stride=1

d) ShuffleNet v2：stride=2

![image-20220905200431947](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905200432.png)

​	请注意，ShuffleNet v1[15]中的“Add”操作已不再存在。像ReLU和dw卷积这样的 element-wise 只存在于一个分支中。此外，三个连续的元素操作“Concat”、“Channel Shuffle”和“Channel Split”合并为一个元素操作。根据G4的规定，这些变化是有益的。

![image-20220905201013130](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905201013.png)

​	对于空间向下采样，对该单元进行了轻微的修改，如图3(d).所示将移除channel split operator。因此，**输出通道的数量增加了一倍**。

​	为简单起见，我们设置了**c'=c/2**。整体网络结构类似于ShuffleNet v1[15]，总结如表5所示。

![image-20220905201557832](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905201558.png)

ShuffleNet v1 和 ShuffleNet v2的框架基本上都是一样的，只有一个区别：添加了一个额外的1×1卷积层（Conv5）

![image-20220905202456414](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905202457.png)

而且对于stage2的第一个block，它的两个分支中输出的channel并不等于输入的channel，而是直接设置为指定输出channel的一半，比如对于1x版本每分支的channel应该是58，不是24.

## 2.3 对比

![image-20220905203154589](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220905203155.png)

1.0 1.4 表示倍率因子
