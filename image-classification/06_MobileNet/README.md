# MobileNet V1、V2

**Effificient Convolutional Neural Networks for Mobile Vision**

发表于2017

传统卷积神经网络，内存需求大、运算量大导致无法在移动设备以及嵌入式设备上运行

# 作者



![image-20220818161328720](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818161328720.png)

MobileNet网络是由google团队在2017年提出的，专注于**移动端**或者**嵌入式设备**中的轻量级CNN网络。相比传统卷积神经网络，在准确率小幅降低的前提下大大减少模型参数与运算量。(相比VGG16准确率减少了0.9%,但模型参数只有vGG的1/32)



网络中的亮点:

- Depthwise Convolution(大大减少运算量和参数数量)
- 增加超参数α（卷积核个数倍率）、β（控制图像大小）

# 1 MobileNet V1

## 1.1 传统卷积

![image-20220818162301578](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818162301578.png)

​	

## 1.2 DW卷积



![image-20220818162746093](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818162746093.png)

- 卷积核channel=1
- 输入特征矩阵channel=卷积核个数=输出特征矩阵channel
  

## 1.3 Depthwise Separable Conv 深度可分卷积

​	![image-20220818162953345](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818162953345.png)

就是一个普通的卷积，只不过卷积核的大小为1.



## 1.4 普通卷积与 DW+PW 的计算量区别

![image-20220818163700111](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818163700111.png)



![image-20220818164003416](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818164003416.png)



超参数α（卷积核个数倍率）：Width Multiplier

![image-20220818163858979](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818163858979.png)

超参数β（控制图像大小）: Resolution Multiplier

![image-20220818163916828](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818163916828.png)



![image-20220818163934123](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818163934123.png)

​	DW卷积大部分卷积核的参数会变成0，没有起到作用。



# 2 MobileNet V2

​	MobileNet v2网络是由google团队在**2018**年提出的，相比MobileNet V1网络，准确率更高，模型更小。



![image-20220818172349670](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818172349670.png)

网络中的亮点:

- lnverted Residuals（倒残差结构)
- Linear Bottlenecks

## 2.1 倒残差结构

![image-20220818172531558](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818172531558.png)

Residuals blocks 使用 ReLU 激活函数

lnverted Residuals（倒残差结构) 使用 ReLU6 激活函数

### 2.1.1 ReLU6激活函数

![image-20220818173413887](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818173413887.png)

### 2.1.1 最后一个 1×1 卷积

​	使用线性激活函数，而不是使用ReLU激活函数。

![image-20220818192352890](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818192352890.png)

图1：嵌入在高维空间中的低维流形的RELU变换的例子。在这些例子中，初始螺旋用**随机矩阵T**和ReLU嵌入**到n维空间**中，然后**用T<sup>-1</sup>投影回2D空间**。<u>在n=2，3的例子中</u>，流形的某些点相互折叠成彼此的信息损失，而在n=15~30的情况下，变换是高度非凸的。

ReLU激活函数对低维特征信息照成大量损失，所以用线性激活函数。



## 2.2 模型

​	**当stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut（捷径分支）连接**

![image-20220818193727045](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818193727045.png)



![image-20220818193933642](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818193933642.png)



![image-20220818194728256](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818194728256.png)

- t是**扩展因子**
- c是输出特征矩阵深度channel
- n是bottleneck的重复次数
- s是步距（针对第一层，其他为1)

## 2.3 评估

### 2.3.1 Classification

![image-20220818195748933](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818195748933.png)



### 2.3.2 Object Detection

![](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220818195748933.png)



# 3 MobileNet V3

2019年

![image-20220904203131306](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904203132.png)



![image-20220904203225246](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904203226.png)

网络中的亮点:

- 更新Block（bneck）[在倒残差结构的基础上改动]
- 使用NAS搜索参数（Neural Architecture Search）
- 重新设计耗时层结构

## 3.1 对比

![image-20220904204006377](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904204007.png)

​	与MobileNet V2相比，MobileNet V3在ImigeNet分类任务中准确率高出3.2%，同时减少了20%的延迟。MobileNetV3-Small比MobileNetV2模型的准确率高出6.6%

![image-20220904204326715](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904204327.png)

## 3.2 更新Block

![image-20220904205210388](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904205211.png)

dw卷积中：stride==1且 input_c == output_c

- 加入SE模块（注意力机制）
- 更新了激活函数（NL-非线性激活函数）

![image-20220904205224900](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904205225.png)

​	池化操作（向量的长度就是这个特征矩阵的channel）-> 第一个全连接层channel的个数是前面特征矩阵channel个数的 1/4 -> 第二个全连接层channel个数与池化后的特征矩阵的channel保持一致 

​	（对于比较重要的 channel 赋予更大的权重，对于不太重要的 channel 赋予比较小的权重）

![image-20220904210257347](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904210258.png)

流程：

![image-20220904215631598](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220904215631598.png)



## 3.3 重新设计耗时层结构

​	1.减少第一个卷积层的卷积核个数 (32->16)

​	2.精简 Last Stage

​	我们将过滤器的数量从32减少到16个，与使用32个过滤器具有相同的精度。所以我们使用更少的卷积核的个数，这节省了2毫秒的时间。

![image-20220904220500575](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904220501.png)



​	使用NAS搜索出来的网络结构的最后一个部分是 Oriiginal Last Stage 。

![image-20220904220449576](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904220450.png)

​	Efficent Last Stage 减少了7毫秒的延迟，即运行时间的11%，而且几乎没有损失精度。

![image-20220904220531121](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904221721.png)



## 3.4 重新设计激活函数



![image-20220904222059100](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904222059.png)

​		swish *x* 激活函数计算、求导复杂，对量化过程不友好。

​		因此，作者提出了 h-swish 激活函数，

​	![image-20220904222647593](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904222648.png)

​	通过下图可以看出非常相似：

![image-20220904222027584](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904222028.png)



## MobileNet V3-Large

<img src="https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904224412.png" alt="image-20220904224411796" style="zoom: 67%;" />



## MobileNet V3-Small

![image-20220904224525851](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904224526.png)



## Pytorch官方实现



![image-20220904231158605](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220904231159.png)
