#   运用调好的参数模型，以及添加了新的特征的数据集对数据集做一个测试
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import random
import os
from sklearn.ensemble import RandomForestClassifier
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

data = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/feature_construct', sep='\t', engine='python')
print(data.tail(10))
X = data.drop(['dx'], axis=1)
# print(X)
y = data['dx']
# 利用X_add与y_add为了保证新构建的数据（均值，众数，中位数）只能在训练集中，不能进入测试集
num_adddata = 6 # num_adddata，取最后几行也即新构建的数据放入训练集
X_add = X.tail(num_adddata)
y_add = y.tail(num_adddata)
# print(X_add)
print(y_add)
X = X[0: -num_adddata]
y = y[0: -num_adddata]
# print(X)
# print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)
X_train = X_train.append(X_add)
y_train = y_train.append(y_add)
# print(X_train)
forest = RandomForestClassifier(n_estimators=50, random_state= 42)
forest.fit(X_train, y_train.values.ravel())
importance = forest.feature_importances_
indices = np.argsort(importance)[::-1]
# print(data.columns)
# 可视化属性的重要性排名
plt.figure()
columns = X.columns
look_num = 20   # 输入想要查看的个数
plt.scatter(columns[indices[0:look_num]], importance[indices[0:look_num]], marker='o')
plt.title('OTU and importance')
plt.xlabel('OTU_name')
plt.ylabel('importance')
plt.show()

find_num = 14  # 前15个重要的属性，也即模型输入的维度
OTU_xuhao = indices[0: find_num]
# print(OTU_xuhao[0: find_num])
#根据所选属性构建新的input_data
X_train = X_train.iloc[:, OTU_xuhao]
X_test = X_test.iloc[:, OTU_xuhao]
# print(X_train.columns)

#将输入数据转换为Tensor，默认为FloatTensor
X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values)
y_train = y_train.view(-1, 1)
#print(y_train.shape)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values)
y_test = y_test.view(-1, 1)

learning_rate = 1e-4
epochs = 10000
# model_nodenum = []  # 节点数集合
import basic_NN

# class mboi_test_1(nn.Module):
#     def __init__(self, input_channeel, model_nodenum, out_channel=1):
#         super().__init__()
#         self.hidden1 = nn.Linear(input_channeel, model_nodenum[0])
#         self.hidden2 = nn.Linear(model_nodenum[0], model_nodenum[1])
#         self.hidden3 = nn.Linear(model_nodenum[1], out_channel)
#
#     def forward(self, x):
#         x = F.relu(self.hidden1(x))
#         x = F.relu(self.hidden2(x))
#         x = torch.sigmoid(self.hidden3(x))
#         return x
model = basic_NN.mboi_test_1(input_channeel=find_num, model_nodenum=[3, 14])
# print(model)
optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
Loss_function = nn.BCELoss(reduction='sum')
Loss = []
for epoch in range(epochs):
    y_hat = model(X_train)
    loss = Loss_function(y_hat, y_train)
    optimzer.zero_grad()
    Loss.append(loss)
    loss.backward()
    optimzer.step()
    if(epoch%2000 == 0):
        print('迭代次数：', epoch, loss)


from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix

def deal_output(outdata, class_yuzhi):
    '''
    # 数据格式转换，将输出数据转换为sklearn中可用的格式
    # :param outdata: 经模型处理后的输出数据
    # :param class_yuzhi: 阈值，低于阈值的置为0
    # :return: 返回可用格式
    '''
    outdata_list = outdata.detach().numpy()
    outdata_list = outdata_list.flatten()
    outdata_list = np.where(outdata_list > class_yuzhi, 1, 0)
    outdata_list = outdata_list.reshape(-1, 1) # 只想将其变为n列，用这个reshape（-1,1）
    return outdata_list

def AUC_plot(input_data,input_data_y,isplot = False):
    pre_id = model(input_data)
    list_pre = pre_id.detach().numpy()
    list_input_data_y = input_data_y.detach().numpy()
    fpr, tpr, thresholds = roc_curve(list_input_data_y, list_pre, pos_label= 1)
    auc_num = auc(fpr, tpr)
    if (isplot):
        plt.figure()
        plt.xlabel('Fpr')
        plt.ylabel('Tpr')
        plt.title('ROC Curve and AUC')
        plt.plot(fpr, tpr, label='ROC Curve (area = %0.4f)' % auc_num, color='red')
        plt.legend(loc='lower right')
        plt.show()
    print('测试集AUC= ',auc_num)
    return auc_num

# X_train_pre = model(X_train)
# y_train_list = y_train.detach().numpy()
# X_train_pre = deal_output(X_train_pre, 0.5)
# train_auc_score = accuracy_score(y_train_list, X_train_pre)
# print('训练集准确率：', train_auc_score)
AUC_plot(X_test, y_test, isplot= True)