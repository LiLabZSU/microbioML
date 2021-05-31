import numpy as np
import random
import math
import matplotlib.pyplot as plt
import basic_NN
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score

class GA(object):
    # 初始化
    def __init__(self, population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm):
        '''
        初始化参数
        :param population_size:种群数
        :param chromosome_num:染色体数，对应寻优参数的个数
        :param chromosome_length:染色体基因长度
        :param max_value:二进制基因转换为十进制的最大值
        :param iter_num:迭代次数
        :param pc:交叉率
        :param pm:变异率
        '''
        self.population_size = population_size
        self.chromosome_num = chromosome_num
        self.chromosome_length = chromosome_length
        self.max_value = max_value
        self.iter_num = iter_num
        self.pc = pc
        self.pm = pm
    def specia_origin(self):
        '''
        input:定义的类参数
        :return:populati（list）种群
        '''
        population = []
        for i in range(self.chromosome_num):
            tmp1 = []   # 存一条染色体中的全部可能的二进制基因
            for j in range(self.population_size):
                tmp2 = []   # 存染色体基因中的每一位二进制基因
                for l in range(self.chromosome_length):
                    tmp2.append(random.randint(0, 1))
                tmp1.append(tmp2)
            population.append(tmp1)
        return population
    def translation(self, population):
        '''
        input: 将染色体的二进制基因转换为十进制
        :param population:种群
        :return:种群的十进制数
        '''
        population_decimalism = []
        for i in range(len(population)):
            tmp = []
            for j in range(len(population[0])):
                total = 0
                for l in range(len(population[0][0])):
                    total += population[i][j][l] * (math.pow(2, l)) # 可转换为位运算
                tmp.append(total)
            population_decimalism.append(tmp)
        return population_decimalism

    def fitness(self, population):
        '''
        input: self定义的类参数
        :param population:种群
        :return:每一组染色体对应的适应度值
        '''
        fitness = []
        epochs = 10000
        model_nodenum_1 = self.translation(population)
        population_decimalis = self.translation(population)
        # 利用GA染色体数值迭代至神经网络模型
        for i in range(len(model_nodenum_1[0])):
            tmp = []    # 暂存每组染色体的十进制数
            # for j in range(len(population)):
            #     value = population_decimalis[j][i] * self.max_value / (math.pow(2, self.chromosome_length) - 10)
            #     tmp.append(value)
            # if tmp[0] == 0.0:
            #     tmp[0] = 1
            # if tmp[1] == 0.0:
            #     tmp[1] = 1


            for n in range(len(model_nodenum_1)):
                tmp.append(model_nodenum_1[n][i])
            if tmp[0] == 0.0:
                tmp[0] = 1
            if tmp[1] == 0.0:
                tmp[1] = 1
            input_tmp = [int(x) for x in tmp]
            model = basic_NN.mboi_test_1(input_channeel=basic_NN.find_num, model_nodenum=input_tmp)
            # print(model)
            optimzer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))
            Loss_function = nn.BCELoss(reduction='sum')
            for epoch in range(epochs):
                y_hat = model(basic_NN.X_train)
                loss = Loss_function(y_hat, basic_NN.y_train)
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()

            y_predict = model(basic_NN.X_test)
            y_predict_list = y_predict.detach().numpy()
            y_test_list = basic_NN.y_test.detach().numpy()  # 将tensor转换为numpy

            fpr, tpr, thresholds = roc_curve(y_test_list, y_predict_list, pos_label=1)
            auc_num = auc(fpr, tpr)
            # basic_NN.AUC_plot(basic_NN.X_test, basic_NN.y_test, isplot=True)
            fitness.append(auc_num)


        fitness_value = []
        num = len(fitness)
        for l in range(num):
            if(fitness[l]>0):
                tmp1 = fitness[l]
            else:
                tmp1 = 0
            fitness_value.append(tmp1)
        return fitness_value

        return fitness_value
    def sum_value(self, fitness_value):
        '''
        适应度求和
        :param fitness_value: 每组染色体的适应度值
        :return:适应度函数值之和
        '''
        total = 0.0
        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total
    def comsum(self, fitness1):
        '''
        计算适应度函数累加列表
        :param fitness1:适应度函数值列表
        :return:适应度函数累加列表
        '''
        for i in range(len(fitness1)-1, -1, -1):    #range(start,stop,step) 倒计数
            total = 0.0
            j = 0
            while(j<=i):
                total += fitness1[j]
                j+=1
            fitness1[i] = total

    def selection(self, population, fitness_value):
        '''
        选择操作
        :param population: 种群
        :param fitness_value: 适应度值
        :return:
        '''
        new_fitness = []    # 存储归一化适应度函数值
        total_fitness = self.sum_value(fitness_value)
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i] / total_fitness)
        self.comsum(new_fitness)

        ms = [] #存储随机数
        pop_len = len(population[0])
        for i in range(pop_len):
            ms.append(random.randint(0, 1))
        ms.sort()
        # 存储每个染色体取值的指针
        fitin = 0
        newin = 0

        new_population = population
        # 赌轮盘选择染色体
        while((newin < pop_len)&(fitin < pop_len)):
            if(ms[newin]<new_fitness[fitin]):
                for j in range(len(population)):
                    new_population[j][newin] = population[j][fitin]
                newin += 1
            else:
                fitin += 1
        population = new_population

    def crossover(self, population):
        '''
        交叉操作
        :param population: 种群
        :return:
        '''
        pop_len = len(population[0])

        for i in range(len(population)):
            for j in range(pop_len - 1):
                if (random.random()< self.pc):
                    cpoint = random.randint(0, len(population[i][j]))   # 随机选择基因中的交叉点
                    # 实现相邻的染色体基因取值的交叉
                    tmp1 = []
                    tmp2 = []
                    # 将tmp1作为暂存器，暂时存放第i个染色体第j个取值中的前0到cpoint个基因，
                    # 然后再把第i个染色体第j+1个取值中的后面的基因，补充到tem1后面
                    tmp1.extend(population[i][j][0: cpoint])
                    tmp1.extend(population[i][j+1][cpoint: len(population[i][j])])
                    # 将tmp2作为暂存器，暂时存放第i个染色体第j+1个取值中的前0到cpoint个基因，
                    # 然后再把第i个染色体第j个取值中的后面的基因，补充到tem2后面
                    tmp2.extend(population[i][j+1][0: cpoint])
                    tmp2.extend(population[i][j][cpoint: len(population[i][j])])
                    population[i][j] = tmp1
                    population[i][j+1] = tmp2

    def mutation(self, population):
        '''
        变异操作
        :param population:种群
        :return:
        '''
        pop_len = len(population[0])    # 种群数
        Gene_len = len(population[0][0])    # 基因长度
        for i in range(len(population)):
            for j in range(pop_len):
                if(random.random()<self.pm):
                    mpoint = random.randint(0, Gene_len-1)  #基因变位点
                    if(population[i][j][mpoint]==1):
                        population[i][j][mpoint] = 0
                    else:
                        population[i][i][mpoint] = 1

    def best(self, population_decimalism, fitness_value):
        '''
        找出最好的参数
        :param popilation_decimalism: 种群
        :param fitness_value:函数值列表
        :return:最优参数
        '''
        pop_len = len(population_decimalism[0])
        bestparameters = []
        bestfitness = 0.0
        for i in range(0, pop_len):
            tmp = []
            if(fitness_value[i]>bestfitness):
                bestfitness = fitness_value[i]
                for j in range(len(population_decimalism)):
                    tmp.append(population_decimalism[j][i])
                    if(population_decimalism[0] == 0.0):
                        population_decimalism[0] = 1
                    if(population_decimalism[1] == 0.0):
                        population_decimalism[1] = 1
                    bestparameters = tmp
        return bestparameters, bestfitness

    def main(self):
        results = []
        parameters = []
        best_fitness = 0.0
        best_parameters = []
        population = self.specia_origin()

        for i in range(self.iter_num):
            fitness_value = self.fitness(population)
            population_decimalis = self.translation(population)
            current_parameter, current_fitness = self.best(population_decimalis, fitness_value)
            if(current_fitness>best_fitness):
                best_fitness = current_fitness
                best_parameters = current_parameter
            print('best_parameters', best_parameters, 'best_fitness', best_fitness)
            results.append(best_fitness)
            parameters.append(best_parameters)
            self.selection(population, fitness_value)
            self.crossover(population)
            self.mutation(population)
        results.sort()
        print('Final parameters:', results[-1])


test_GA = GA(population_size=8, chromosome_num=2, chromosome_length=4, max_value=17, iter_num=5, pc=0.2, pm= 0.1)
test_GA.main()
'''
可能存在的问题：在两次随机染色体生成过程中，若节点数（所生成的染色体数值）相同或相近，则类似神经网络多训练了一个大迭代，故效果会好一些
'''