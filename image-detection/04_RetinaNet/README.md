# **Focal Loss for Dense Object Detection**

2017

![image-20220907093957437](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907093958.png)

​	One-stage 网络首次超越了 Two-stage 网络，



## 5 实验

![image-20220907094156231](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907094156.png)

## 

## 3 结构

![image-20220907094528688](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907094529.png)



![image-20220907094835274](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907094836.png)

（为什么anchor比图还大？）

对每个预测特征层使用了使用了三个scale。



![image-20220907101909332](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907101910.png)



![image-20220907101945455](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907101946.png)

K——类别

A——每个特征层上预测的个数 3 * 3 =9

在FasterRCNN中是对于预测特征层上每一个anchor都会针对每个类别去生成一组边界框回归参数。



## 正负样本匹配

1. loU >=0.5,正样本
2. loU <0.4，负样本
3. lou E[0.4,0.5)，舍弃



## 损失计算

![image-20220907104705838](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220907104706.png)
