'''
数据处理，包括背景信息的融入，归一化操作,香农和辛普森多样性指数的融入
'''

'''
背景信息的融入
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


shared = pd.read_table('E:/研究学习/杭州竞赛/2021.5.26/data/source_data/baxter.0.03.subsample.shared', engine='python')
# print(shared.shape)
meta = pd.read_table('E:/研究学习/杭州竞赛/2021.5.26/data/source_data/metadata.tsv', engine='python')
shared = shared.rename(index=str, columns={'Group': 'sample'})  #将原列名Group变为sample
# print(meta.head())
# 处理meta信息
meta = meta[['sample', 'dx', 'Age', 'Smoke', 'Gender', 'Diabetic', 'Height', 'Weight', 'BMI']]
# print(meta.head())
# print(meta.isnull().sum(axis = 0))    # 查看缺失值的列数及个数
meta['Smoke'] = meta['Smoke'].fillna(1)   # 填充smoke缺失值
# print(meta['Diabetic'])
meta['Diabetic'] = meta['Diabetic'].fillna(0.0)
# print(meta['Diabetic'])
meta['Height'] = meta['Height'].fillna(meta['Height'].mean())
# print(meta['Height'])
meta['Weight'] = meta['Weight'].fillna(meta['Weight'].mean())
# print(meta['BMI'])
meta['BMI'] = meta['BMI'].fillna(meta['BMI'].mean())
# print(meta.isnull().sum(axis = 0))    # 查看缺失值的列数及个数
# 处理性别的编码
Gender_tmp = {'m': 1, 'f': 0}
meta['Gender'] = meta['Gender'].replace(Gender_tmp)
# 处理OTU数据以及合并数据
data_tem = pd.merge(shared, meta, on='sample')  # 一个中间的data，方便后面处理数据
data = data_tem[data_tem.dx.str.contains('adenoma') == False]
# print(data.shape) # 最终得到292个样本集合,整个二维结构化数据维度为[292,6931]
data = data.drop(['sample', 'numOtus', 'label'], axis=1) # 删除无用的信息——样本编号，OTU数目等
# print(data.shape)
# 最终得到292个样本集合,整个二维结构化数据维度为[292,6928]
data.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/deal_data_add_meta_info', sep='\t', index=False)

'''
数据归一化
'''
X = data.drop(['dx'], axis=1)
diagonisis = {'cancer': 1, 'normal': 0}
y = data['dx'].replace(diagonisis)
X_columns = X.columns
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)
# print(type(X))
X = pd.DataFrame(X)
X.columns = X_columns
columns = X.columns
# print(X.shape)
# print(y.shape)
X.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/normalization_X', sep='\t', index=False)
y.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/y', sep='\t', index=False)
# 得到加入背景信息的X和y，维度为（292,6927）和（292，）


# print(data_tem.shape)
data = data_tem
# print(data.shape)
shannon = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/shannon', sep='\t', engine='python')
# print(shannon.shape)
simpson = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/simpson', sep='\t', engine='python')
data['shannon'] = shannon
data['simpson'] = simpson
# print(data.shape)
data = data.drop(['sample', 'numOtus', 'label'], axis=1)
# print(data.shape)
data = data[data.dx.str.contains('adenoma') == False]
# print(data.shape)
# print(data)
data.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/all_info_data', sep='\t', index=False)
# 至此得到了包含所有背景信息及多样性指数的数据

X = data.drop(['dx'], axis=1)
diagonisis = {'cancer': 1, 'normal': 0}
y = data['dx'].replace(diagonisis)
X_columns = X.columns
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)
# print(type(X))
X = pd.DataFrame(X)
X.columns = X_columns
columns = X.columns
# print(X.shape)
# print(y.shape)
X.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/normalization_all_info_X', sep='\t', index=False)
y.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/all_info_y', sep='\t', index=False)
print(X.shape)
print(y.shape)