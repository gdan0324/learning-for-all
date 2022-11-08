# Vision Transformer

2020 ICLR

![image-20220907220718082](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907220718.png)



![image-20220907225632229](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907225633.png)

​	ViT-L/16：16代表每个patch大小是16*16，将每一个 patch 输入到 **Embedding** 层（Linear Projection of Flattened Patches），通过 Embedding 得到一个个向量（token），每一个patch都会得到一个token。增加一个用于分类的**class token**，这里的class所对应的token的dimension与后面得到的向量是一样的，向量的长度也是相同的。还需要加一个**Position Embedding**. 输入到 Encoder，重复堆叠L次。提取calss token 所对应的输出，通过MLP head，得到最终的结果。

- Linear Projection of Flattened Patches(Embedding层)
- Transformer Encoder(图右侧有给出更加详细的结构)
- MLP Head(最终用于分类的层结构)



## 3. Embedding层

-  对于标准的Transformer模块，要求输入的是token(向量)序列，即二维矩阵[num_token, token_dim]
  在代码实现中，直接通过一个卷积层来实现以ViT-B/16为例，使用卷积核大小为16x16，stride为16，卷积核个数为768
- [224,224,3] ->[14,14,768] ->[196, 768]
- 在输入Transformer Encoder之前需要加上[class]token以及Position Embedding，都是可训练参数
  拼接[class]token: Cat([1,768],[196,768])->[197,768]
- 叠加Position Embedding: [197,768] ->[197,768]



#### 3.1. Position Embedding

不使用Embedding的区别

![image-20220908092019774](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908092021.png)

​	the difference in performance is fully explained by the requirement for a different learning-rate, see Figure 9.



![image-20220908092259010](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908092259.png)

余弦相似度位置编码



## 4. Encoder层

![image-20220908093336026](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908093336.png)



## 5. MLP Head层

- 注意，在Transformer Encoder前有个Dropout层，后有一个LayerNorm
- 训练lmageNet21K时是由Linear+tanh激活函数+Linear
- 但是迁移到ImageNet1K上或者你自己的数据上时，只有一个Linear



![image-20220908093623317](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908093624.png)



## 6. 模型参数



![image-20220908093951398](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908093952.png)

- Layers是Transformer Encoder中重复堆叠Encoder Block的次数
- Hidden size是通过Embedding层后每个token的dim（向量的长度)
- MLP size是Transformer Encoder中MLP Block第一个全连接的节点个数（是Hidden Size的四倍)
- Heads代表Transformer中Multi-Head Attention的heads数



## Hybrid混合模型

- R50的卷积层采用的StdConv2d不是传统的Conv2d
- 将所有的BatchNorm层替换成GroupNorm层
- 把stage4中的3个Block移至stage3中

​	https://blog.csdn.net/qq_37541097/article/details/118016048



## 4 实验

![image-20220907225803735](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907225804.png)



![image-20220908095722248](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908095723.png)



