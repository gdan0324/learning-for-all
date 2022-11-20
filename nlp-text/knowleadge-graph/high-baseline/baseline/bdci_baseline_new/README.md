# 目录树:
    ├── dataset                                             # 数据文件夹
    │──── bdci                                              # 数据集文件夹
    │     ├── train_bdci.json           # 比赛提供的原始数据
    │     ├── rel2id.json               # 关系到id的映射
    │     ├── evalA.json                # A榜评测集
    ├── data_generator.py                                   # 数据处理工具
    ├── output                                              # 模型输出保存路径 
    │   ├── config.txt                            # 模型训练参数 
    │   ├── evalResult.json                       # 模型预测结果
    │   ├── pytorch_model.bin                     # 训练好的模型
    │   └── log.txt                               # 模型训练日志
    │   └── dev_pred.json                         # 验证集预测结果
    │   └── test_pred.json                        # 测试集预测结果
    ├── predict.py                                          # 预测代码
    ├── pretrain_models                                     # 预训练模型
    ├── requirements.txt                                    # 依赖库
    ├── train.py                                            # 训练代码
    ├── data_utils.py                                       # 数据处理类
    ├── model.py                                            # 模型类
    └── util.py                                             # 工具类

# 一、预训练模型下载
    1、链接: https://pan.baidu.com/s/1COlBY1k9yHoGXAZdEpdbWA?pwd=dgre 提取码: dgre
    2、解压后，将预训练模型文件夹放到pretrain_models目录下
    
# 二、安装环境依赖
    安装requirement.txt文件中的依赖包
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --default-time=2000
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-time=2000 tensorflow
    
# 三、生成模型数据
    执行python data_generator.py生成train.json和test.json
    
    
# 四、模型训练
    执行 python train.py
    
# 五、模型预测
    执行 python predict.py

# 六、注意事项
    1.所用模型：GRTE；github：https://github.com/neukg/GRTE
    2.batch_size不可设置过大。3090卡，显存24G，batch_size为4
