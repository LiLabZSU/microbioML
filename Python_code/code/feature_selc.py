# 数据降维即特征选择
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# 原始数据，未加入背景信息
X = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/origin_X_data', sep='\t', engine='python')
y = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/origin_y_data', sep='\t', engine='python')
# 加入背景信息
# X = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/normalization_all_info_X', sep='\t', engine='python')
# y = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/all_info_y', sep='\t', engine='python', header=None)
# 加入分组统计的数据
data = data = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/feature_construct', sep='\t', engine='python')


# print(X.shape)
# print(y.shape)

forest = RandomForestClassifier(n_estimators= 50, random_state= 10, n_jobs= -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 10, shuffle= True)
forest.fit(X_train, y_train.values.ravel())
importance = forest.feature_importances_
indices = np.argsort(importance)[::-1]

# 特征选择及数据保存
selc_num = 15
OTU_index = indices[0: selc_num]
print(OTU_index)
feature_selc_X = X.iloc[:, OTU_index]
# print(feature_selc_X)
# feature_selc_X.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/feature_selc_X', sep='\t', index=False)
feature_selc_X.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/origin_feature_selc_X', sep='\t', index=False)
# 得到降维后的数据（292,15）

# 可视化属性的重要性排名
plt.figure()
columns = X.columns
look_num = 15   # 输入想要查看的个数
# print(indices[0:look_num])
# print(columns[indices[0:look_num]])
plt.scatter(columns[indices[0:look_num]], importance[indices[0:look_num]], marker='o')
plt.title('OTU and importance')
plt.xlabel('OTU_name')
plt.ylabel('importance')
plt.show()
