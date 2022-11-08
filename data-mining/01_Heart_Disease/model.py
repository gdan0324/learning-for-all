import numpy as np
import pandas as pd
import matplotlib.pyplot as p1t
# 忽略警告
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('data/process_heart.csv')

# 去掉这一列  矩阵用X表示  input
X = df.drop('target', axis=1)

# y向量
y = df['target']

# 将数据划分为训练集和测试集,20%作为测试集，随机数种子
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 构建随机森林分类模型，在训练集上训练模型
from sklearn.ensemble import RandomForestClassifier
# 最大深度为5，决策树为100，随机种子数为5
model = RandomForestClassifier(max_depth=5,n_estimators=100,random_state=5)
# fit 拟合
model.fit(X_train,y_train)
# 可以查看第7个决策树
estimator = model.estimators_[7]

# 将输出特征值转为字符串
feature_names = X_train.columns
y_train_str = y_train.astype('str')
y_train_str[y_train == '0'] = 'no disease'
y_train_str[y_train == '1'] = 'disease'
y_train_str = y_train_str.values

# 将决策树可视化
from sklearn.tree import export_graphviz

export_graphviz(estimator, out_file='tree.dot',
                feature_names=feature_names,
                class_names=y_train_str,
                rounded=True, proportion=True,
                label='root',
                precision=2, filled=True)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'], shell=True)
from IPython.display import Image

Image(filename='tree.png')

