# ΢���������ݵĻ���ѧϰ����

#��װ���R��
install.packages("tidyverse")
install.packages("randomForest")
install.packages("pROC")


#��������
setwd("D:\\LILAB\\microbiomeML") #��������·��,������ʵ��·���޸�
data <- read.csv("S1_table.csv", header=TRUE, row.names=1,sep=",")  #���׸���1��OTU������
head(data) #��ʾ��������
#data$Malodour <- factor(ifelse(data$Malodour == "N", 1, 2))

#���ݷ���(k-fold)
CVgroup <- function(k, datasize, seed){
  cvlist <- data.frame()
  set.seed(seed)
  n <- rep(1:k, ceiling(datasize/k))[1:datasize] #�����ݷֳ�k�ݣ������ɵ��������ݼ�n
  temp <- sample(n, datasize) #�����
  x <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(x, function(x) dataseq[temp == x])#dataseq���������k���������������
  return(cvlist)
}#������ۺ���

k <- 10    # k-fold
datasize <- 90	#������
cvlist <- CVgroup(k = k, datasize = datasize, seed = 123) #�������

#����ģ�Ϳ�
library(randomForest)

pred <- data.frame()#�洢Ԥ����

for (i in 1:10) {
  train <- data[-cvlist[[i]], ] #ѵ����
  test <- data[cvlist[[i]], ] #���Լ�
   
  #���ɭ��
  rf.model <- randomForest(Malodour ~ ., data = train, ntree=100) #����randomforestģ�ͣ�ntreeָ������
  summary(rf.model)
  rf.pred <- predict(rf.model, test, type = "prob")[,2] #Ԥ��
  kcross <- rep(i, length(rf.pred))	#i�ڼ���ѭ�����棬��K��
  temp <- cbind(Malodour = test$Malodour, Predict = as.data.frame(rf.pred)[,1], kcross)
  pred <- rbind(pred, temp)

}#ѭ������K��ģ��

head(pred)


#����ROC����
library(pROC)
library(ggplot2)
rf.roc <- roc(pred$Malodour, as.numeric(pred$rf.pred))
rf.roc$auc

#plot(roc2)
ggroc(rf.roc, alpha=0.5, colour="red", linetype=1, size=2 ,legacy.axes = TRUE) +
	annotate("text", x = .75, y = .15, label = paste("AUC of RandomForest =", round(rf.roc$auc,2)))



   

