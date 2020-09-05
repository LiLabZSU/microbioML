# machine learning exercise
install.packages("tidyverse")
install.packages("randomForest")
install.packages("e1071")
install.packages("neuralnet")
install.packages("pROC")
install.packages("Metrics")

library(tidyverse)
library(randomForest)
library(e1071)
library(neuralnet)
library(pROC)
#载入数据
setwd("D:\\LILAB\\microbiomeML")
data <- read.csv("S1_table.csv")
head(data) #display data
data <- data[,-1]
data$Malodour <- factor(ifelse(data$Malodour == "N", 1, 2))

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

k <- 10
datasize <- 90
cvlist <- CVgroup(k = k, datasize = datasize, seed = 123) #代入参数

set.seed(2020)
svm.p2<-data.frame()
rf.p2<-data.frame()
ne.p2<-data.frame()

for (i in 1:10) {
  train<-data[-cvlist[[i]],]
  test<-data[cvlist[[i]],]
  #SVM
  svm.train<-svm(Malodour~.,train,probability=T)
  svm.pred <- predict(svm.train,test,probability=T)
  svm.pred<-attr(svm.pred,"probabilities")[,2]
  svm.p1<-data.frame(cbind(subset(test,select=Malodour),matrix(svm.pred)))
  svm.p2<-rbind(svm.p2,svm.p1)
  #randomforest
  rf.train<-randomForest(Malodour~.,train,ntree=500)
  rf.pred<-predict(rf.train,test,type="prob")[,2]
  rf.p1<-data.frame(cbind(subset(test,select=Malodour),matrix(rf.pred)))
  rf.p2<-rbind(rf.p2,rf.p1)
  #neuralnet
  n <- names(train)
  formula <- as.formula(paste("Malodour ~", paste(n[!n %in% "Malodour"], collapse = " + ")))
  neuralnet.train<-neuralnet(formula, train,hidden =c(500,500),rep=1,err.fct="ce",linear.output = FALSE)
  neuralnet.pred<-predict(neuralnet.train,test)[,2]
  ne.p1<-data.frame(cbind(subset(test,select=Malodour),matrix(neuralnet.pred)))
  ne.p2<-rbind(ne.p2,ne.p1)
}

#准确率
Metrics::accuracy(ifelse(svm.p2[,2]<0.5,1,2),svm.p2[,1])
Metrics::accuracy(ifelse(rf.p2[,2]<0.5,1,2),rf.p2[,1])
Metrics::accuracy(ifelse(ne.p2[,2]<0.5,1,2),ne.p2[,1])

#特异性、敏感性、auc
roc1<-roc(svm.p2[,1],svm.p2[,2])
mean(roc1$sensitivities)
mean(roc1$specificities)
roc1$auc
coord1=coords(roc1, "best", ret=c("threshold", "specificity", "sensitivity"), transpose = FALSE)

roc2<-roc(rf.p2[,1],rf.p2[,2],coords=T)
mean(roc2$sensitivities)
mean(roc2$specificities)
roc2$auc
coord2=coords(roc2, "best", ret=c("threshold", "specificity", "sensitivity"), transpose = FALSE)

roc3<-roc(ne.p2[,1],ne.p2[,2])
mean(roc3$sensitivities)
mean(roc3$specificities)
roc3$auc
coord3=coords(roc3, "best", ret=c("threshold", "specificity", "sensitivity"), transpose = FALSE)

g2<-ggroc(list(svm=roc1,randomforest=roc2,neuralnet=roc3),legacy.axes = TRUE)+
  geom_point(aes(x=1-coord1[1,2],y=coord1[1,3]),color="#FFC0CB",size=4,shape=17,alpha= .25)+
  geom_point(aes(x=1-coord2[1,2],y=coord2[1,3]),color="#FFC0CB",size=4,shape=18,alpha= .25)+
  geom_point(aes(x=1-coord3[1,2],y=coord3[1,3]),color="#FFC0CB",size=4,shape=19,alpha= .25)+
  annotate("text",x=.2,y=.9,label=paste0(coord1[1,1],"(",round(1-coord1[1,2],2),",",round(coord1[1,3],2),")"))+
  annotate("text",x=.2,y=.8,label=paste0(coord2[1,1],"(",round(1-coord2[1,2],2),",",round(coord2[1,3],2),")"))+
  annotate("text",x=.2,y=.7,label=paste0(coord3[1,1],"(",round(1-coord3[1,2],2),",",round(coord3[1,3],2),")"))+
  annotate("text",x=.75,y=.45,label=paste("AUC of SVM =", round(roc1$auc,2)))+
  annotate("text",x=.75,y=.35,label=paste("AUC of randomforest =", round(roc2$auc,2)))+
  annotate("text",x=.75,y=.25,label=paste("AUC of neuralnet =", round(roc3$auc,2)))
g2
