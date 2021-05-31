# 测试背景信息的添加对于模型准确率的变化
import basic_NN
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

X = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/feature_selc_X', sep='\t', engine='python')
y = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/y', sep='\t', engine='python', header=None)
# X = pd.read_csv('E:/研究学习/杭州竞赛/2021.5.26/data/result_data/origin_feature_selc_X', sep='\t', engine='python')
# 分组统计的数据

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)
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

model = basic_NN.mboi_test_1(input_channeel=15, model_nodenum=[14, 3])
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决中文乱码
def AUC_plot(input_data,input_data_y,isplot = False):
    pre_id = model(input_data)
    list_pre = pre_id.detach().numpy()
    list_input_data_y = input_data_y.detach().numpy()
    fpr, tpr, thresholds = roc_curve(list_input_data_y, list_pre, pos_label= 1)
    auc_num = auc(fpr, tpr)
    # print(list_input_data_y)
    # print(list_pre)
    if (isplot):
        plt.figure()
        plt.xlabel('Fpr')
        plt.ylabel('Tpr')
        plt.title('ROC Curve and AUC')
        plt.plot(fpr, tpr, label='ROC Curve (area = %0.4f)' % auc_num, color='red')
        plt.legend(loc='lower right')
        plt.show()
    print('测试集AUC= ', auc_num)
    return auc_num
AUC_plot(X_test, y_test, isplot= True)

