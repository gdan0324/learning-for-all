# 1. 决策树



## 1.1 西瓜书数据集

### 1.1.1 数据使用

![image-20220707143931638](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220707143931638.png)

## 1.2 生成决策树

json结构的树形结构

```json
{'feature:':'纹理','label:':None,'sons:':{'清晰':{'feature:':'敲声','label:':None,'sons:':{'浊响':{'feature:':'色泽','label:':None,'sons:':{'乌黑':{'feature:':'触感','label:':None,'sons:':{'硬滑':{'feature:':None,'label:':'好瓜','sons:':{}},'软粘':{'feature:':None,'label:':'坏瓜','sons:':{}}}},'青绿':{'feature:':None,'label:':'好瓜','sons:':{}},'浅白':{'feature:':None,'label:':'好瓜','sons:':{}}}},'沉默':{'feature:':None,'label:':'好瓜','sons:':{}},'清脆':{'feature:':None,'label:':'坏瓜','sons:':{}}}},'稍糊':{'feature:':'触感','label:':None,'sons:':{'硬滑':{'feature:':None,'label:':'坏瓜','sons:':{}},'软粘':{'feature:':None,'label:':'好瓜','sons:':{}}}},'模糊':{'feature:':None,'label:':'坏瓜','sons:':{}}}}
```

## 1.3 预测结果

​	一共有17条数据，使用了1-15条数据，构造决策树。去预测最后两条数据：

![image-20220707144056456](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220707144056456.png)

## 1.2 蘑菇分类数据集

### 1.2.1 数据集介绍

![image-20220707135753083](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220707135753083.png)

属性信息：（类：可食用=e，有毒=p）

- 帽形：钟形=b,圆锥形=c,凸形=x,扁平形=f,球形=k,凹陷=s
- 帽面：纤维=f，凹槽=g，鳞片=y，光滑=s
- cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
- 瘀伤：瘀伤=t，无=f
- 气味：杏仁=a,茴香=l,杂酚油=c,腥味=y,恶臭=f,霉味=m,无=n,辛辣=p,辣=s
- 鳃附件：attached=a,descending=d,free=f,notched=n
- 鳃间距：近=c，拥挤=w，远=d
- 鳃大小：宽=b，窄=n
- 鳃色: 黑=k,棕=n,浅黄=b,巧克力=h,灰=g,绿=r,橙=o,粉红=p,紫=u,红=e,白=w,黄=是的
- 茎形：放大=e，锥形=t
- 茎根: 球根=b,棒=c,杯=u,等=e,根茎=z,根=r,缺=?
- 茎表面上环：纤维=f，鳞状=y，丝状=k，光滑=s
- 茎表面下环：纤维=f，鳞状=y，丝状=k，光滑=s
- 茎色环：棕色=n,浅黄色=b,肉桂=c,灰色=g,橙色=o,粉色=p,红色=e,白色=w,黄色=y
- 茎色-下环：棕=n,浅黄=b,肉桂=c,灰=g,橙=o,粉=p,红=e,白=w,黄=y
- 面纱类型：部分=p，通用=u
- 面纱颜色：棕色=n，橙色=o，白色=w，黄色=y
- 环数：none=n,one=o,two=t
- 环型: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
- 孢子印色：黑色=k,棕色=n,浅黄色=b,巧克力=h,绿色=r,橙色=o,紫色=u,白色=w,黄色=y
- 人口：丰富=a，聚集=c，众多=n，分散=s，若干=v，孤立=y
- 栖息地：草=g，树叶=l，草地=m，路径=p，城市=u，废物=w，树林=d

8124 rows × 23 columns



### 1.2.2 生成决策树

对 8124 条数据，10%作为训练集，90%作为验证集

```json
{'feature:': 'spore-print-color', 'label:': None, 'sons:': {'w': {'feature:': 'ring-number', 'label:': None, 'sons:': {'o': {'feature:': 'cap-color', 'label:': None, 'sons:': {'e': {'feature:': None, 'label:': 'p', 'sons:': {}}, 'n': {'feature:': None, 'label:': 'p', 'sons:': {}}, 'c': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 't': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'n': {'feature:': None, 'label:': 'p', 'sons:': {}}}}, 'n': {'feature:': 'veil-color', 'label:': None, 'sons:': {'w': {'feature:': 'stalk-surface-below-ring', 'label:': None, 'sons:': {'s': {'feature:': 'stalk-color-above-ring', 'label:': None, 'sons:': {'w': {'feature:': 'stalk-color-below-ring', 'label:': None, 'sons:': {'w': {'feature:': 'stalk-surface-above-ring', 'label:': None, 'sons:': {'s': {'feature:': 'ring-type', 'label:': None, 'sons:': {'p': {'feature:': 'stalk-shape', 'label:': None, 'sons:': {'e': {'feature:': 'gill-spacing', 'label:': None, 'sons:': {'c': {'feature:': 'cap-shape', 'label:': None, 'sons:': {'b': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'x': {'feature:': 'population', 'label:': None, 'sons:': {'s': {'feature:': 'gill-size', 'label:': None, 'sons:': {'b': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'n': {'feature:': None, 'label:': 'p', 'sons:': {}}}}, 'n': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'y': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'v': {'feature:': None, 'label:': 'p', 'sons:': {}}}}, 'f': {'feature:': 'cap-surface', 'label:': None, 'sons:': {'y': {'feature:': None, 'label:': 'p', 'sons:': {}}, 's': {'feature:': None, 'label:': 'p', 'sons:': {}}, 'f': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 's': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'w': {'feature:': None, 'label:': 'p', 'sons:': {}}}}, 't': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'e': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'f': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'p': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'g': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'g': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'p': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'f': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'y': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'o': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'n': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'h': {'feature:': 'gill-size', 'label:': None, 'sons:': {'b': {'feature:': None, 'label:': 'p', 'sons:': {}}, 'n': {'feature:': None, 'label:': 'e', 'sons:': {}}}}, 'k': {'feature:': 'gill-size', 'label:': None, 'sons:': {'b': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'n': {'feature:': None, 'label:': 'p', 'sons:': {}}}}, 'r': {'feature:': None, 'label:': 'p', 'sons:': {}}, 'u': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'o': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'y': {'feature:': None, 'label:': 'e', 'sons:': {}}, 'b': {'feature:': None, 'label:': 'e', 'sons:': {}}}}
```

### 1.2.3 出现的问题

当采用前10%的数据构建决策树的时候，发现做预测的时候会报错，因为构建的决策树里面有些特征是没有划分到树里面的。

例如，只有三个特征划分到树里面，而他原本有9个特征

茎色环：棕色=n,浅黄色=b,肉桂=c,灰色=g,橙色=o,粉色=p,红色=e,白色=w,黄色=y

![image-20220707153511490](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220707153511490.png)

所以在预测下面这条数据的时候，就报错了。

![image-20220707153746013](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220707153746013.png)



### 1.2.4 解决方案

1. 对数据进行随机采样

2. 对于特征不在树上的，直接返回'未知'结果。

   

### 1.2.5 评估

![image-20220707160609515](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220707160609515.png)

混淆矩阵：

![image-20220707161045802](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220707161045802.png)

准确率为98%



# 2. ID3、C4.5、BIN决策树比较结果

## 2.1 不打乱数据集

​	为了保证采样得到的数据是**相同**的，取**前70%**的数据作为训练，**30%**的数据作为验证。

​	

|     指标     |   名称   |                            描述                            |                           计算公式                           |
| :----------: | :------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
|  macro avg   |  宏平均  |                      对所有类别的平均                      |                       (P_no+P_yes) / 2                       |
|  micro avg   |  微平均  |   对数据集中的每⼀个实例不分类别进⾏统计建⽴全局混淆矩阵   |                   (TP+TN) / (TP+FP+TN+FN)                    |
| weighted avg | 加权平均 | 是对宏平均的一种改进，考虑了每个类别样本数量在总样本中占比 | P_no * （support_no / support_all）+ P_yes * （support_yes / support_all） |



### 	2.1.1 ID3结果

![image-20220819115126558](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220819115126558.png)

​		精确度：0.80

### 	2.1.2 C4.5结果

![image-20220819120243815](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220819120243815.png)

​	精确度：0.87

### 	2.1.3 BIN结果

![image-20220819120131908](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220819120131908.png)

​	精确度：0.78



## 2.2 K折交叉验证

​	取 k = 5，固定随机种子 random_state = 100

### 	2.2.1 ID3结果

![image-20220819121942848](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220819121942848.png)

​	时间：0.8s



### 	2.2.2 C4.5结果

![image-20220819122110701](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220819122110701.png)

​	时间：1.6s



### 2.2.3 BIN结果



![image-20220819121908975](https://gitee.com/shuangshuang853/picture-bed/raw/master/image-20220819121908975.png)

有2个分类错误

时间：18.5s



**[疑问]** roc和auc曲线好像需要得到分类概率？























