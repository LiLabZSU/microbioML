'''
得到未添加任何背景信息的数据
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

shared = pd.read_table('E:/研究学习/杭州竞赛/2021.5.26/data/source_data/baxter.0.03.subsample.shared', engine='python')
# print(shared.shape)
meta = pd.read_table('E:/研究学习/杭州竞赛/2021.5.26/data/source_data/metadata.tsv', engine='python')
# print(meta[['Age', 'Smoke']])
meta = meta[['sample', 'dx']]
# print(shared.head())
shared = shared.rename(index=str, columns={'Group': 'sample'})  #将原列名Group变为sample
# print(shared.head())
data = pd.merge(shared, meta, on='sample')
# print(data.head())
data = data[data.dx.str.contains('adenoma') == False]
# print(data.shape)
# print(data.isnull().sum(axis = 0))    # 查看是否有缺失值以及缺失值的数目
# print(data['Smoke'])
# data['Smoke'] = data['Smoke'].fillna(1)   # 填充缺失值
# print(data.head())
data.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/origin_data', sep='\t', index=False)
X = data.drop(['sample', 'dx', 'numOtus', 'label'], axis=1)
# print(X.shape)
diagonisis = {'cancer': 1, 'normal': 0}
y = data['dx'].replace(diagonisis)
# print(X.shape)
# print(data.isnull().sum(axis = 0))
X.dropna()
y.dropna()
X_columns = X.columns
# print(X_columns)
# 数据归一化
# print(type(X))
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)
# print(type(X))
X = pd.DataFrame(X)
X.columns = X_columns
print(X.shape)
# print(y)
X.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/origin_X_data', sep='\t', index=False)
y.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/origin_y_data', sep='\t', index=False)