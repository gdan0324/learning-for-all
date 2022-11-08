

- 语义分割	(semantic segmentation)	FCN
- 实例分割	(Instance segmentation)	Mask R-CNN
- 全景分割	(Panoramic segmentation)	 Panoptic FPN



![image-20220908145717271](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908145718.png)



# PASCAL VOC

![image-20220908152224205](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908152225.png)

通道数是1，P模式

- 比如像素0对应的是(0,0,0)黑色
- 像素1对应的是127,0,0)深红
- 像素255对应的是(224,224,129）

忽略255的边缘

https://blog.csdn.net/qq_37541097/article/details/115787033





# MS COCO

针对图像中的每一个目标都记录了多边形坐标polygons

![image-20220908153430953](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908153432.png)



# 语义分割得到结果的具体形式



![image-20220908153717638](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908153718.png)

# 评价指标



![image-20220908154457296](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908154458.png)

FCN原论文



## mean IOU

![image-20220908164720522](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220908164721.png)



Labelme 

https://lgithub.com/wkentaro/labelme

https://blog.csdn.net/qq_37541097/article/details/120162702



半自动标注工具：https://github.com/PaddlePaddle/PaddleSeg





# 转置卷积

Transposed Convolution

作用: upsampling

- 转置卷积不是卷积的逆运算
- 转置卷积也是卷积

![image-20220909092043487](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909092044.png)

A guide to convolution arithmetic for deep learning (https://arxiv.org/abs/1603.07285)



转置卷积运算步骤:

- 在输入特征图元素间填充s-1行、列0
- 在输入特征图四周填充k-p-1行、列0
- 将卷积核参数上下、左右翻转
- 做正常卷积运算（填充0，步距1)

https://github.com/vdumoulin/conv_arithmetic
![image-20220909093621828](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909093622.png)

![image-20220909092820666](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909092821.png)



## 例子

