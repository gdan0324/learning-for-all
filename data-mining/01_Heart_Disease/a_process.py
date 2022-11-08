import pandas as pd

df = pd.read_csv('data/heart.csv')

# 将定类特征由整数编码转为实际对应的字符串，还原为真实含义
df['sex'][df['sex'] == 0] = 'female'
df['sex'][df['sex'] == 1] = 'male'
df['cp'][df['cp'] == 0] = 'typical angina'
df['cp'][df['cp'] == 1] = 'atypical angina'
df['cp'][df['cp'] == 2] = 'non-anginal pain'
df['cp'][df['cp'] == 3] = 'asymptomatic'

df['fbs'][df['fbs'] == 0] = 'lower than 120mg/ml'
df['fbs'][df['fbs'] == 1] = 'greater than 120mg ml'

df['restecg'][df['restecg'] == 0] = 'normal'
df['restecg'][df['restecg'] == 1] = 'ST-T wave abnormality'
df['restecg'][df['restecg'] == 1] = 'left ventricular hyper trophy'

df['exang'][df['exang'] == 0] = 'no'
df['exang'][df['exang'] == 1] = 'yes'

df['slope'][df['slope'] == 0] = 'upsloping'
df['slope'][df['slope'] == 1] = 'flat'
df['slope'][df['slope'] == 1] = 'downsloping'

df['thal'][df['thal'] == 0] = 'unknown'
df['thal'][df['thal'] == 1] = 'normal'
df['thal'][df['thal'] == 1] = 'fixed defect'
df['thal'][df['thal'] == 1] = 'reversable defect'

# 将离散的定类和定序特征列转为One-Hot独热编码
# 将定类数据扩展为特征
df = pd.get_dummies(df)

# 导出预处理后的数据
df.to_csv('data/process_heart.csv', index=False)