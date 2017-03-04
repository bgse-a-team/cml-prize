library(readr)
library(class)
library(pROC)
# Data manipulation
X_train <- as.data.frame(read_csv("X_train.dat"))
X_train <- X_train[,-1]
colnames(X_train) <- paste0(rep("X",78),1:78)
y_train <- as.data.frame(read_csv("y_train.dat", col_names = FALSE))
y_train <- as.factor(y_train[,2])

# Cross-validation: split the data into 4 parts
# KNN for 1-10 neightbours
folds <- cut(seq(1:nrow(X_train)),breaks=4,labels=F)
max_neightbours <- 10
AUC <- matrix(rep(NA,max_neightbours*length(unique(folds))),ncol=max_neightbours)
for (i in 1:length(unique(folds))){
  testDataX <- X_train[which(folds==i),]
  testDatay <- y_train[which(folds==i)]
  trainDataX <- X_train[-which(folds==i),]
  trainDatay <- y_train[-which(folds==i)]
  for (k in 1:10){
    predictions <- knn(train=trainDataX, test=testDataX, cl=trainDatay,k=k)
    roc_obj <- roc(testDatay, predictions)
    AUC[i,k] <- auc(roc_obj)
  }
}
AUC
apply(AUC,2,mean)

