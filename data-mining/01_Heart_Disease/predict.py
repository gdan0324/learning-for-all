# 忽略警告
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集,划分特征和标签
df = pd.read_csv('data/process_heart.csv')
X = df.drop('target', axis=1)
y = df['target']
# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 构建随机森林模型
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=5, n_estimators=100)
model.fit(X_train, y_train)

## 对数据进行位置索引，从而在数据表中提取出相应的数据。
X_test.iloc
# 筛选出未知样本
test_sample = X_test.iloc[2]
# 变成二维
test_sample = np.array(test_sample).reshape(1, -1)

#  二分类定性分类结果
model.predict(test_sample)
# 二分类定量分类结果
model.predict_proba(test_sample)

y_pred = model.predict(X_test)
# 得到患心脏病和不患心脏病的置信度
y_pred_proba = model.predict_proba(X_test)
# 切片操作 只获得患心脏病的置信度
model.predict_proba(X_test)[:, 1]

# 混淆矩阵
from sklearn.metrics import confusion_matrix

confusion_matrix_model = confusion_matrix(y_test, y_pred)
# 将混淆矩阵绘制出来
import itertools


def cnf_matrix_plotter(cm, classes):
    '''
    传入混淆矩阵和标签名称列表，绘制混淆矩阵
    '''
    # plt.imshow (cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black", fontsize=25)
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel(' Predicted Label')
    plt.show()


cnf_matrix_plotter(confusion_matrix_model, ['Healthy', 'Disease'])

# 计算查全率、召回率、调和平均值
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease']))

# ROC曲线
y_pred_quant = model.predict_proba(X_test)[:, 1]
print(model.predict_proba(X_test))
print(y_pred_quant,len(y_pred_quant))
from sklearn.metrics import roc_curve, auc
print(y_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# 计算AUC曲线
auc(fpr, tpr)