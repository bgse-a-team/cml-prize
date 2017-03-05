library(readr)
library(class)
library(AUC)
library(e1071)
library(MASS)
library(glmnet)
# Data manipulation
X_train <- as.data.frame(read_csv("X_train.dat"))
X_train <- X_train[,-1]
X_train <- X_train[,-which(apply(X_train,2,function(x) all(x==0)))]
colnames(X_train) <- paste0(rep("X",ncol(X_train)),1:ncol(X_train))
X_train <- apply(X_train, 2, scale)
X_test <- as.data.frame(read_csv("X_test.dat"))[,1]
X_test <- X_test[,-1]
X_test <- X_test[,-which(apply(X_test,2,function(x) all(x==0)))]
colnames(X_test) <- paste0(rep("X",ncol(X_test)),1:ncol(X_test))
X_test <- apply(X_test, 2, scale)
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
which.max(apply(AUC,2,mean)) # for 15 neightbours the AUC is 0.684

# Logistic Regression using PC
folds <- cut(seq(1:nrow(X_train)),breaks=4,labels=F)
AUC <- matrix(rep(NA,ncol(X_train)*length(unique(folds))),ncol=ncol(X_train))
principal_components <- prcomp(X_train)$x
for (i in 1:length(unique(folds))){
  testDataX <- principal_components[which(folds==i),]
  testDatay <- y_train[which(folds==i)]
  trainDataX <- principal_components[-which(folds==i),]
  trainDatay <- y_train[-which(folds==i)]
  for (k in 1:ncol(principal_components)){
    complete_data <- as.data.frame(trainDataX[,1:k, drop=F])
    logistic_model <- glm(trainDatay ~ ., data=complete_data, family=binomial)
    predictions <- predict(logistic_model, newdata=as.data.frame(testDataX[,1:k, drop=F]), type="response")
    predictions[predictions < 0.5] <- 0
    predictions[predictions >= 0.5] <- 1
    predictions <- as.factor(predictions)
    AUC[i,k] <- auc(roc(predictions,testDatay))
  }
}
AUC
which.max(apply(AUC,2,mean)) # 58 (0.709)
plot(1:73,apply(AUC,2,mean),type="l")

# Logistic Regression and lasso
folds <- cut(seq(1:nrow(X_train)),breaks=4,labels=F)
AUC <- rep(NA,length(unique(folds)))
cv_logistic_model <- cv.glmnet(X_train, as.numeric(y_train), alpha=1)
best_lambda <- cv_logistic_model$lambda.min
for (i in 1:length(unique(folds))){
  testDataX <- X_train[which(folds==i),]
  testDatay <- y_train[which(folds==i)]
  trainDataX <- X_train[-which(folds==i),]
  trainDatay <- y_train[-which(folds==i)]
  logistic_model <- glmnet(trainDataX, trainDatay, alpha=1, family="binomial", lambda=best_lambda)
  predictions <- predict(logistic_model, newx=testDataX, s=best_lambda, type="response")
  predictions[predictions < 0.5] <- 0
  predictions[predictions >= 0.5] <- 1
  predictions <- as.factor(predictions)
  AUC[i] <- AUC::auc(roc(predictions,testDatay))
}
AUC
mean(AUC) # 0.7091387
# Predicting testing data
cv_logistic_model <- cv.glmnet(X_train, as.numeric(y_train), alpha=1)
best_lambda <- cv_logistic_model$lambda.min
logistic_model <- glmnet(X_train, y_train, alpha=1, family="binomial", lambda=best_lambda)
predictions <- predict(logistic_model, newx=X_test, s=best_lambda, type="response")
predictions[predictions < 0.5] <- 0
predictions[predictions >= 0.5] <- 1
pred <- cbind(as.matrix(read_csv("X_test.dat"))[,1], predictions)
colnames(pred) <- c("Id","Prediction")
write.csv(pred, file="first_submission.csv", row.names=F)

# SVM with linear kernel
tuned_linear <- tune(svm, y_train ~ X_train[,1], kernel="linear", ranges=list(cost=c(0.1,1,10,100,1000)))
summary(tuned_linear)
tuned_linear$best.model
# svmfit <- svm(y_train ~ X_train, kernel="linear", cost=100, scale=F)
predictions <- predict(svmfit,X_train,type="class")
(1 - sum(predictions == y_train)/length(predictions))*100

# SVM with radial kernel
tuned_radial <- tune(svm, y_train ~ X_train, kernel="radial", ranges=list(cost=c(0.1,1,10,1000), gamma=c(0.5,1,2,3,4)))
summary(tuned_radial)
tuned_radial$best.model
# svmfit <- svm(y_train ~ X_train, kernel="radial", cost=100, gamma=1, scale=F)
predictions <- predict(svmfit,X_train,type="class")
(1 - sum(predictions == y_train)/length(predictions))*100








