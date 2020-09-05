# 微生物组数据的机器学习代码

#安装相关R包
install.packages("tidyverse")
install.packages("randomForest")
install.packages("pROC")


#载入数据
setwd("D:\\LILAB\\microbiomeML") #数据所在路径,按数据实际路径修改
data <- read.csv("S1_table.csv", header=TRUE, row.names=1,sep=",")  #文献附件1的OTU表数据
head(data) #显示部分数据
#data$Malodour <- factor(ifelse(data$Malodour == "N", 1, 2))

#数据分折(k-fold)
CVgroup <- function(k, datasize, seed){
  cvlist <- data.frame()
  set.seed(seed)
  n <- rep(1:k, ceiling(datasize/k))[1:datasize] #将数据分成k份，并生成的完整数据集n
  temp <- sample(n, datasize) #随机化
  x <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(x, function(x) dataseq[temp == x])#dataseq中随机生成k个随机有序数据列
  return(cvlist)
}#定义分折函数

k <- 10    # k-fold
datasize <- 90	#样本数
cvlist <- CVgroup(k = k, datasize = datasize, seed = 123) #代入参数

#载入模型库
library(randomForest)

pred <- data.frame()#存储预测结果

for (i in 1:10) {
  train <- data[-cvlist[[i]], ] #训练集
  test <- data[cvlist[[i]], ] #测试集
   
  #随机森林
  rf.model <- randomForest(Malodour ~ ., data = train, ntree=100) #建立randomforest模型，ntree指定树数
  summary(rf.model)
  rf.pred <- predict(rf.model, test, type = "prob")[,2] #预测
  kcross <- rep(i, length(rf.pred))	#i第几次循环交叉，共K次
  temp <- cbind(Malodour = test$Malodour, Predict = as.data.frame(rf.pred)[,1], kcross)
  pred <- rbind(pred, temp)

}#循环计算K次模型

head(pred)


#绘制ROC曲线
library(pROC)
library(ggplot2)
rf.roc <- roc(pred$Malodour, as.numeric(pred$rf.pred))
rf.roc$auc

#plot(roc2)
ggroc(rf.roc, alpha=0.5, colour="red", linetype=1, size=2 ,legacy.axes = TRUE) +
	annotate("text", x = .75, y = .15, label = paste("AUC of RandomForest =", round(rf.roc$auc,2)))



   


