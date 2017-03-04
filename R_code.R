library(readr)
library(class)
library(AUC)
library(e1071)
# Data manipulation
X_train <- as.data.frame(read_csv("X_train.dat"))
X_train <- X_train[,-1]
X_train <- X_train[,-which(apply(X_train,2,function(x) all(x==0)))]
colnames(X_train) <- paste0(rep("X",ncol(X_train)),1:ncol(X_train))
X_train <- apply(X_train, 2, scale)
y_train <- as.data.frame(read_csv("y_train.dat", col_names = FALSE))
y_train <- as.factor(y_train[,2])

# Cross-validation: split the data into 4 parts
# KNN for 1-15 neightbours
folds <- cut(seq(1:nrow(X_train)),breaks=4,labels=F)
max_neightbours <- 15
AUC <- matrix(rep(NA,max_neightbours*length(unique(folds))),ncol=max_neightbours)
for (i in 1:length(unique(folds))){
  testDataX <- X_train[which(folds==i),]
  testDatay <- y_train[which(folds==i)]
  trainDataX <- X_train[-which(folds==i),]
  trainDatay <- y_train[-which(folds==i)]
  for (k in 1:max_neightbours){
    predictions <- knn(train=trainDataX, test=testDataX, cl=trainDatay,k=k)
    AUC[i,k] <- auc(roc(predictions,testDatay))
  }
}
AUC
apply(AUC,2,mean)

for (i in 1:ncol(trainDataX)){
  print(which(is.na(trainDataX[,i])))
}
  

# SVM with linear kernel
tuned_linear <- tune(svm, y_train ~ X_train[,1], kernel="linear", ranges=list(cost=c(0.1,1,10,100,1000)))
summary(tuned_linear)
tuned_linear$best.model
# svmfit <- svm(y_train ~ X_train, kernel="linear", cost=100, scale=F)
predictions <- predict(svmfit,X_train,type="class")
(1 - sum(predictions == y_train)/length(predictions))*100

# SVM with radial kernel
tuned_radial <- tune(svm, y_train ~ X_train[,1], kernel="radial", ranges=list(cost=c(0.1,1,10,1000), gamma=c(0.5,1,2,3,4)))
summary(tuned_radial)
tuned_radial$best.model
# svmfit <- svm(y_train ~ X_train, kernel="radial", cost=100, gamma=1, scale=F)
predictions <- predict(svmfit,X_train,type="class")
(1 - sum(predictions == y_train)/length(predictions))*100








