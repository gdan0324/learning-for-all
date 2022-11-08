# DeepLab V1——SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS

2014 CVPR

![image-20220909175237145](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909175237.png)



## 语义分割任务中存在的问题

![image-20220909175415723](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909175416.png)

在图像标记任务中的应用存在两个技术障碍：下采样会导致图像分辨率降低和空间“不敏感”（不变性）。

解决方案：

- 'atrous'(with holes) algorithm(空洞卷积/膨胀卷积/扩张卷积)
- fully-connected CRF(Conditional Random Field) 【V3没用了】



## 网络优势

- 速度更快，论文中说是因为采用了膨胀卷积的原因，但fully-connected CRFs很耗时
- 准确率更高，相比之前最好的网络提升了7.2个点
- 模型结构简单，主要由DCNNs和CRFs联级构成

![image-20220909190659374](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909190700.png)



## LargeFOV

​	在将网络转换为完全卷积层后，第一个全连接层有4096个7×7空间大小的滤波器，成为我们密集分数图计算的计算瓶颈。我们通过对第一个FC层（通过简单的抽取）进行<u>空间子采样到4×4（或3×3）的空间大小</u>来解决这个实际问题。

![image-20220909190916613](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909190917.png)



## MSc(Multi-Scale)

![image-20220909193033413](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909193034.png)



## DeepLab-MSc-LargeFOV

![image-20220909193204914](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909193205.png)

https://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-MSc-LargeFOV/train.prototxt



# DeepLab V2

![image-20220909194250044](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909194250.png)



![image-20220909194325716](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909194326.png)



## DCNNs应用在语义分割任务中问题

- 分辨率被降低(主要由于下采样stride>1的层导致)
- 目标的多尺度问题
- DCNNs的不变性(invariance)会降低定位精度

![image-20220909194756649](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909194757.png)

## 对应解决方法

- **针对分辨率被降低的问题**，一般就是将最后的几个Maxpooling层的**stride设置成1**(如果是通过卷积下采样的，比如resnet，同样将stride设置成1即可)，**配合使用膨胀卷积**。
- **针对目标多尺度的问题**，最容易想到的就是将图像缩放到多个尺度分别通过网络进行推理，最后将多个结果进行融合即可。这样做虽然有用但是计算量太大了。为了解决这个问题，DeepLab V2中提出了ASPP模块( atrous spatial pyramid pooling)。
- **针对DCNNs不变性导致定位精度降低的问题**，和DeepLab V1差不多还是**通过CRFs解决**，不过这里用的是fully connected pairwise CRF，相比V1里的fully connected CRF要更高效点。

## 网络优势

- 速度更快
- 准确率更高（当时的state-of-art)
- 模型结构简单，还是DCNNs和CRFs联级



## ASPP



![image-20220909195912603](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909195913.png)



![image-20220909195954719](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909195955.png)



![image-20220909200243916](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909200244.png)



## 消融实验 

![image-20220909200340679](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909200341.png)



![image-20220909200419772](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909200420.png)



## 学习率变化策略

![image-20220909200803856](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909200804.png)



### Poly学习率

![image-20220909200920949](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909200921.png)



## DeepLab V2 网络架构

![image-20220909201050161](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909201050.png)





# DeepLab V3

2017

![image-20220909201621145](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909201621.png)



## 

![image-20220909202301299](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909202302.png)

博文推荐: https://blog.csdn.net/qq_37541097/article/details/121797301

- 引入了Multi-grid
- 改进ASPP
- 结构移除CRFs后处理



![image-20220909202549792](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909202550.png)



## DeepLabV3两种模型结构

联级模型

## ![image-20220909203034995](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909203035.png)



![image-20220909203315191](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909203315.png)



![image-20220909203305136](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909203305.png)



## Multi-grid



![image-20220909203525465](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909203526.png)



![image-20220909203553050](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909203553.png)



![image-20220909203641933](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909203642.png)



### 级联模型

![image-20220909203830766](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909203831.png)



### ASPP model消融实验

![image-20220909204004178](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909204005.png)



### 训练细节

![image-20220909204044656](https://gitee.com/shuangshuang853/picture-bed/raw/master/picture/20220909204045.png)

### Pytorch实现的DeepLabV3

- 没有使用Multi-Grid，有兴趣的同学可以自己动手加上试试。
- 多了一个FCNHead辅助训练分支，可以选择不使用。
- 无论是训练还是验证output_stride都使用的8。
- ASPP中三个膨胀卷积分支的膨胀系数是12,24,36









## 该项目主要是来自pytorch官方torchvision模块中的源码

* https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10
* Ubuntu或Centos(Windows暂不支持多GPU训练)
* 最好使用GPU训练
* 详细环境配置见```requirements.txt```

## 文件结构：
```
  ├── src: 模型的backbone以及DeepLabv3的搭建
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train.py: 以deeplabv3_resnet50为例进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的mIoU等指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重下载地址：
* 注意：官方提供的预训练权重是在COCO上预训练得到的，训练时只针对和PASCAL VOC相同的类别进行了训练，所以类别数是21(包括背景)
* deeplabv3_resnet50: https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth
* deeplabv3_resnet101: https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
* deeplabv3_mobilenetv3_large_coco: https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth
* 注意，下载的预训练权重记得要重命名，比如在train.py中读取的是```deeplabv3_resnet50_coco.pth```文件，
  不是```deeplabv3_resnet50_coco-cd0a2569.pth```


## 数据集，本例程使用的是PASCAL VOC2012数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* 如果不了解数据集或者想使用自己的数据集进行训练，请参考我的博文: https://blog.csdn.net/qq_37541097/article/details/115787033

## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要使用单GPU或者CPU训练，直接使用train.py训练脚本
* 若要使用多GPU训练，使用```torchrun --nproc_per_node=8 train_multi_GPU.py```指令,```nproc_per_node```参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上```CUDA_VISIBLE_DEVICES=0,3```(例如我只要使用设备中的第1块和第4块GPU设备)
* ```CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py```

## 注意事项
* 在使用训练脚本时，注意要将'--data-path'(VOC_root)设置为自己存放'VOCdevkit'文件夹所在的**根目录**
* 在使用预测脚本时，要将'weights_path'设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改'--num-classes'、'--aux'、'--data-path'和'--weights'即可，其他代码尽量不要改动

## 如果对DeepLabV3原理不是很理解可参考我的bilibili
* https://www.bilibili.com/video/BV1Jb4y1q7j7


## 进一步了解该项目，以及对DeepLabV3代码的分析可参考我的bilibili
* https://www.bilibili.com/video/BV1TD4y1c7Wx

## Pytorch官方实现的DeeplabV3网络框架图
![deeplabv3_resnet50_pytorch](./deeplabv3_resnet50.png)
