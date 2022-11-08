# SuperGlue

![image-20221017194720920](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221017194720920.png)

​	超级胶水：学习特征匹配图神经网络



​	SuperGlue 网络是结合了<u>最优匹配层的图神经网络</u>，经过训练可以<u>对两组稀疏图像特征</u>进行匹配。

​	SuperGlue 作为“中端”运行，在单个端到端架构中执行上下文聚合、匹配和过滤。

- 作者：*Paul-Edouard Sarlin、Daniel DeTone、Tomasz Malisiewicz、Andrew Rabinovich*

# 摘要

​	本文介绍了一种通过联合寻找对应和拒绝非匹配点来匹配两组局部特征的神经网络。通过求解一个可微最优传输问题来估计分配，该问题的代价由图神经网络预测。我们引入了一种**基于注意力**的灵活的上下文聚合机制，**使SuperGlue能够共同推理底层的三维场景和特征分配**。与传统的、手工设计的启发式方法相比，我们的技术通过对图像对的端到端训练来学习三维世界的几何变换和规律。<u>在挑战现实世界的室内外环境中，超级胶优于其他学习方法</u>，**并在姿态估计任务中取得了最先进的结果**。该方法在现代GPU上进行<u>实时匹配</u>，并可以很容易地集成到现代SfM或**SLAM**系统中。这些代码和训练过的权重可以在github.com/magicleap/SuperGluePretrainedNetwork.上公开提供



# 引言















