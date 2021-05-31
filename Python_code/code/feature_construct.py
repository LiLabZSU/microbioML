# 新加入了背景信息的数据
# 利用已有的数据通过特征选择中的过滤法、包装法等构造新的特征，分组统计特征
import pandas as pd
import numpy as np
from scipy import stats

X = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/normalization_all_info_X', sep='\t', engine='python')
y = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/y', sep='\t', engine='python',header=None)
print(y)
# data = pd.concat([X, y], axis=1)
X['dx'] = y
data = X
# print(data.shape)
x1 = data[data.dx > 0]  # 取出标签为1的数据
x1_attribute = x1.drop(['dx'], axis=1)  # x1_attribute为不含标签1的属性
x1_label = x1['dx']
# print(x1_label)
x0 = data[data.dx < 1]  # 取出标签为0的数据
x0_attribute = x0.drop(['dx'], axis=1)  # x1_attribute为不含标签1的属性
x0_label = x0['dx']
# print(x0)

# 分组构造特征
# 取平均数构成新的特征
mean1 = np.mean(x1_attribute)
mean0 = np.mean(x0_attribute)
# print(mean1)
mean1['dx'] = '1'   # 为均值加上对应的标签1
mean0['dx'] = '0'   # 为均值加上对应的标签0
# print(mean1.shape)
# print(mean0)
# print(data)
# 将扩增的均值特征加入最初的数据中
new_data = data.append(mean1, ignore_index=True)
new_data = new_data.append(mean0, ignore_index=True)
# print(mean1)
# print(type(new_data))
new_data.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/add_mean', sep='\t', index=False)    # index设为False，不会保存索引，若不设置，默认保存索引

# 取众数构造新的特征 (可能存在问题，不同属性的众数，可以组合起来作为一个新的特征吗？)
columns = x1_attribute.columns
# print(columns.shape)
mode1 = stats.mode(x1_attribute)[0][0]
mode0 = stats.mode(x0_attribute)[0][0]
# print(mode1)
# print(mode0)
# print(type(mode1))
mode1 = pd.DataFrame(mode1, index=columns).T
mode0 = pd.DataFrame(mode0, index=columns).T
# print(mode1)
mode1['dx'] = '1'
mode0['dx'] = '0'
# print(mode1)
# print(mode0)
add_mode = data.append(mode1, ignore_index=True)
add_mode = add_mode.append(mode0, ignore_index=True)
# print(add_mode)
add_mode.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/add_mode', sep='\t', index=False)
final_data = new_data
final_data = final_data.append(mode1, ignore_index=True)
final_data = final_data.append(mode0, ignore_index=True)

# 取中位数构造新的特征
median1 = np.median(x1_attribute, axis=0)
median0 = np.median(x0_attribute, axis=0)
# print(type(median1))
median1 = pd.DataFrame(median1, index=columns).T
median0 = pd.DataFrame(median0, index=columns).T
median1['dx'] = '1'
median0['dx'] = '0'
# print(median1)
add_median = data.append(median1, ignore_index=True)
add_median = add_median.append(median0, ignore_index=True)
# print(add_median)
add_median.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/add_median', sep='\t', index=False)

final_data = final_data.append(median1, ignore_index=True)
final_data = final_data.append(median0, ignore_index=True)
print(final_data)
final_data.to_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/feature_construct', sep='\t', index=False)

