library(readr)
X_train <- as.data.frame(read_csv("X_train.dat"))
colnames(X_train)[2:79] <- paste0(rep("X",78),1:78)
colnames(X_train)[1] <- "index"
