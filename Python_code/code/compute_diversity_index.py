# 香农、辛普森生物多样性指数的计算
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
shared = pd.read_table('E:/研究学习/杭州竞赛/2021.5.26/data/source_data/baxter.0.03.subsample.shared', engine='python')
data = shared.drop(['label', 'Group', 'numOtus'], axis=1)
# print(data)
# print(data.shape)
# 计算香农熵指数
# x = data.sum(axis=1)    # 计算每一行的reads数，结果相同都为10423
# print(type(data))
shanon = []
# print(data.iat[1, 1])
for i in range(data.shape[0]):
    TMP = []
    for j in range(data.shape[1]):
        a = float(data.iat[i, j]/(10423))   # 10423为reads（序列）数
        tmp = np.log10(a) * a * (-1)
        TMP.append(tmp)
    # print(TMP)
    shanon.append(np.nansum(TMP))
# print(shanon)
name1 = ['shanon']
Shannon = pd.DataFrame(columns=name1, data=shanon)
print(Shannon)
Shannon.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/shannon', sep='\t', index=False)
# 最终得到的香农多样性指数为（489，）
# Simpson多样性指数的计算
data1 = np.array(data/10423)
# print(data1)
data1 = data1**2
# print(data1)
Simpson = []
for i in range(data.shape[0]):
    simpson = 0.0
    for j in range(data.shape[1]):
        simpson += data1[i][j]
    simpson = 1 - simpson
    Simpson.append(simpson)
# print(Simpson)
name2 = ['simpson']
Simpson = pd.DataFrame(columns=name2, data=Simpson)
print(Simpson)
Simpson.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/simpson', sep='\t', index=False)
# 最终得到的辛普森多样性指数为（489，）