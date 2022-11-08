# COCO

论文地址：https://arxiv.org/pdf/1.405.0312.pdf

博文：https://blog.csdn.net/qq_37541097/article/details/113247318

![image-20220906151421087](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20220906151421087.png)



![image-20220906151440200](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906151441.png)



![image-20220906151455711](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906151456.png)

这里需要注意的一个点是<u>“什么是stuff类别”</u>，在官方的介绍论文中是这么定义的:
**where "stuff" categories include materials and objectswith no clear boundaries (sky, street, grass)** **stuff** 中包含没有明确边界的材料和对象。



​	object的80类与stuff中的91类的区别在哪?在官方的介绍论文中有如下说明:

![image-20220906153819878](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906153820.png)



## 与 PASCAL VOC 进行对比

预训练效果更好但更费时

6x2x5/24=2.5(单块GPU)

6x2x5x30/24=75(CPU)

![image-20220906154109083](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906154109.png)

https://cocodataset.org/



![image-20220906154723421](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906154724.png)



![image-20220906155446412](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220906155447.png)

 