# 卷积神经网络 CNN （Convolutional Neural Network）

# 1. 历史

雏形：LeCun 的 LeNet（1998）网络结构

![image-20220815151722006](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815151722006.png)

1986 Rumelhart 和 Hinton 等人提出了反向传播（Back Propagation，BP）算法。

1998 LeCun 的利用 BP 算法训练 LeNet5 网络，标志着CNN的真正面世。(硬件跟不上)

2006 Hinton 在他们的 science Paper 中首次提出了Deep Learning的概念。

2012 Hinton 的学生 Alex Krizhevsky 在寝室用 GPU 死磕了一个Deep Learning 模型，一举摘下了视觉领域竞赛 ILSVRC 2012的桂冠，在百万量级的 lmageNet 数据集合上，效果大幅度超过传统的方法，从传统的70%多提升到80%多。

# 2. 全连接层

![image-20220815152800508](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815152800508.png)

​	BP(back propagation）算法包括<u>信号的前向传播</u>和<u>误差的反向传播</u>两个过程。即计算误差输出时按从输入到输出的方向进行，而调整权值和阈值则从输出到输入的方向进行。

## 2.1 反向传播

### 2.1.1 误差的计算



​	![image-20220815164659012](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815164659012.png)

layer 1 的第一个单元：

![image-20220815165129696](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815165129696.png)

求 y1、y2：

![image-20220815165229163](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815165229163.png)

使用softmax是因为想要将y1、y2处于同一个概率分布

![image-20220815165547640](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815165547640.png)

经过softmax处理后所有输出节点概率和为1。计算表达式如下：

![image-20220815165632629](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815165632629.png)

#### Cross Entropy Loss 交叉熵损失计算公式：

![image-20220815165905772](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815165905772.png)

接上面的例子：

![image-20220815170603458](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815170603458.png)



### 2.1.2 误差的反向传播

对 w11(2) 求偏导：

![image-20220815171058556](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815171058556.png)



![image-20220815171432362](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815171432362.png)



![image-20220815172037543](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815172037543.png)





### 2.1.3 权重的更新

![image-20220815172254828](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815172254828.png)



![image-20220815172540317](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815172540317.png)

### 优化器 optimazer

​	目的是为了使网络收敛更快

1. SGD

   ![image-20220815173127711](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815173127711.png)

   缺点：易受样本噪声影响、可能陷入局部最优解

2. SGD + Momentum

   多了一个动量部分，有效抑制噪声样本的干扰。

   ![image-20220815173504640](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815173504640.png)

   

3. Adagrad（自适应学习率）

   st是平方求和项，会越来越大，放到分母会使得学习率变小。

   缺点：学习率下降太快可能还没收敛就停止训练

   ![image-20220815173756980](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815173756980.png)

   

4. RMSProp（自适应学习率）

   引入了控制衰减速率的系数。

   ![image-20220815174239554](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815174239554.png)

   

5. Adam（自适应学习率）

   

   ![image-20220815174421553](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815174421553.png)

所有优化器演示动画：

![](https://gitee.com/shuangshuang853/picture-bed/raw/master/20180426130002689)



## 2.2 例子

​	计算白色像素占整个框像素的比例。

![image-20220815153328410](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815153328410.png)

​	得到 5 * 5 的矩阵之后就展开，拼接成一个行向量。可以将下面的行向量当成神经网络当中的输入层。

![image-20220815155505469](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815155505469.png)

​	独热编码：

![image-20220815155640475](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815155640475.png)

​	

![image-20220815155840108](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815155840108.png)

# 3. 卷积层

卷积操作

![image-20220815160004722](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815160004722.png)

​	目的：进行图像特征提取

​	卷积特性：拥有局部感知机制、权重共享

## 3.1 权重共享例子

​	对于一张1280 * 720 的图片

​	全连接层：假设 hidden layer 1 神经元的个数为1000， 1280 * 720 * 1000 = 9 2160 0000

​	卷积神经网络： 假设 hidden layer 1 采用 1000 个 5 * 5 的卷积核，5 * 5 * 1000 = 25000



## 3.2 卷积输入输出

![image-20220815161124871](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815161124871.png)

​	卷积核的 channel 与输入特征层的channel相同。

​	输出的特征矩阵channel与卷积核个数相同。

### 3.2.1 加上偏移量bias该如何计算?

​		<img src="https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815162031751.png" alt="image-20220815162031751" style="zoom:50%;" />

### 3.2.2 加上激活函数该如何计算?

​	为什么要引入激活函数？引入非线性因素，使其具备解决非线性问题的能力。

 1. sigmoid 函数

    ![image-20220815162309779](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815162309779.png)

    ![image-20220815162340287](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815162340287.png)

    Sigmoid激活函数饱和时梯度值非常小，故网络层数较深时易出现梯度消失。

    

 2. Relu 函数

    ![image-20220815162318547](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815162318547.png)

![image-20220815162348750](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815162348750.png)

​	缺点在于当反向传播过程中有一个非常大的梯度经过时，反向传播更新后可能导致权重分布中心小于零，导致该处的倒数始终为0，反向传播无法更新权重，即进入失活状态。失活之后无法再激活。



### 3.2.3 如果卷积过程中出现越界的情况该怎么办?

​	padding

![image-20220815162820060](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815162820060.png)

# 4. 池化层

​	目的：对特征图进行稀疏处理，减少数据运算量。

	1. 不需要学习参数。
	1. 只改变特征矩阵的w和h，不改变channel
	1. 一般 pool size 和stride相同

## 	4.1 MaxPooling下采样层

​	

![image-20220815163323000](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815163323000.png)

​	

## 	4.2 AveragePooling下采样层

​	

​				![image-20220815163613629](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220815163613629.png)

