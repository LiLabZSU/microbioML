#  基础神经网络结构的搭建
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import os
import random

# 固定随机数，避免结果不可复现
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
X = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/feature_selc_X', sep='\t', engine='python')
y = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/y', sep='\t', engine='python', header=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)
# print(type(X_train))
#将输入数据转换为Tensor，默认为FloatTensor
X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values)
y_train = y_train.view(-1, 1)
#print(y_train.shape)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values)
y_test = y_test.view(-1, 1)
# print(type(X_train))

# 模型搭建

learning_rate = 1e-4
epochs = 10000
find_num = 15
class mboi_test_1(nn.Module):
    def __init__(self, input_channeel, model_nodenum, out_channel=1):
        super().__init__()
        self.hidden1 = nn.Linear(input_channeel, model_nodenum[0])
        self.hidden2 = nn.Linear(model_nodenum[0], model_nodenum[1])
        self.hidden3 = nn.Linear(model_nodenum[1], out_channel)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))
        return x
# 网络训练
# model = mboi_test_1(input_channeel=15, model_nodenum=[7, 5])
# # print(model)
# optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# Loss_function = nn.BCELoss(reduction='sum')
# Loss = []
# for epoch in range(epochs):
#     y_hat = model(X_train)
#     loss = Loss_function(y_hat, y_train)
#     optimzer.zero_grad()
#     Loss.append(loss)
#     loss.backward()
#     optimzer.step()
#     if(epoch%2000 == 0):
#         print('迭代次数：', epoch, loss)


from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix
#
# def AUC_plot(input_data,input_data_y,isplot = False):
#     pre_id = model(input_data)
#     list_pre = pre_id.detach().numpy()
#     list_input_data_y = input_data_y.detach().numpy()
#     fpr, tpr, thresholds = roc_curve(list_input_data_y, list_pre, pos_label= 1)
#     auc_num = auc(fpr, tpr)
#     if (isplot):
#         plt.figure()
#         plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_num))
#         plt.show()
#     print('测试集AUC= ', auc_num)
#     return auc_num
# AUC_plot(X_test, y_test, isplot= False)