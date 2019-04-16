##########################################################
###Load Packages###
library(xgboost)
##########################################################
###Read in the Data###
setwd(path)
df<-read.csv('data.csv')
###Replace the flag with NA###
for(i in 1:ncol(df)){
  df[df[,i]==9999999.000,i] = NA
}

#Remove the columns where more than 30% is missing###
checks = apply(apply(df,2,is.na),2,mean) < 0.3
df= df[,checks]

##########################################################
###K-Fold Cross Validation to find best Hyperparameters###

#Randomly Shuffle the Data#
set.seed(666)
df<-df[sample(nrow(df)),]
xdata = data.matrix(df[,c(2,3)])
ydata = as.factor(as.numeric(df[,1]))
#Create 10 equally size folds#
folds <- cut(seq(1,nrow(df)),breaks=10,labels=FALSE)
#Perform 10 fold cross validation#
nrounds = c(100,150,200,250,300)
maxdepths = c(1,2,3,4)
etas = c(0.1,0.3,0.5,0.7,0.9)
search_grid=expand.grid(nrounds,maxdepths,etas)

all_error_rate = numeric(nrow(search_grid))
sd_rate = numeric(nrow(search_grid))

for(j in 1:nrow(search_grid)){
  nround = search_grid[j,1]
  maxdepth = search_grid[j,2]
  eta = search_grid[j,3]
  error_rate = numeric(10)
  for(i in 1:10){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    xtrain = xdata[-testIndexes,]
    ytrain = ydata[-testIndexes]
    xtest = xdata[testIndexes,]
    ytest = ydata[testIndexes]
    model<-xgboost(data=xtrain,label = ytrain,nrounds = nround,max_depth =maxdepth,eta=eta,objective="multi:softmax",num_class=10,verbose=0)
    modelpred<-predict(model,xtest)
    error_rate[i]<-mean(modelpred==ytest)
    mean(modelpred==ytest)
  }
  all_error_rate[j]<-mean(error_rate)
  sd_rate[j]<-sd(error_rate)
  print(j/nrow(search_grid))
}
CV_Results<-data.frame(all_error_rate,sd_rate)
#Output and save the CV Results#
write.csv(CV_Results,'CV_Results.csv')

rm(list=ls())
#Select the parameters corresponding to the minimum in the error rate#
best=t(search_grid[which.max(CV_Results$all_error_rate),])

#Create data  matrix and scale#
set.seed(666)
df<-df[sample(nrow(df)),]
xdata = data.matrix(df[,c(3,4,5,6,7,8,9,10,11)])
means<-sapply(df,mean,na.rm=TRUE)
means= means[c(3,4,5,6,7,8,9,10,11)]
sds<-sapply(df,sd,na.rm=TRUE)
sds= sds[c(3,4,5,6,7,8,9,10,11)]
xdata = scale(xdata)
ydata = as.factor(as.numeric(df[,2]))

###Fit the data using an XGBoost, a gradient boosted decision tree algorithm.
model<-xgboost(data=xdata,label = ydata,nrounds = best[1,] ,max_depth =best[2,],eta=best[3,],objective="multi:softmax",num_class=10,verbose=1)
model_prob<-xgboost(data=xdata,label = ydata,nrounds = best[1,] ,max_depth =best[2,],eta=best[3,],objective="multi:softprob",num_class=10,verbose=1)
modelpred<-predict(model,xdata)
#Calculate the Accuracy#
mean(modelpred==ydata)

#Create a Confusion Matrix for the Model Predictions#
ydata_confusion=revalue(ydata, c('1'='AGN','2'='ATNF','3'='ATNF_BIN','4'='CV','5'='HMXB','6'='LMXB','7'='STAR','8'='WR','9'='YSO'))
modelpred_confusion=revalue(as.factor(modelpred), c('1'='AGN','2'='ATNF','3'='ATNF_BIN','4'='CV','5'='HMXB','6'='LMXB','7'='STAR','8'='WR','9'='YSO'))
table(modelpred_confusion,ydata_confusion,dnn=list('Prediction','Data'))

#Calculate the most import features in X-Ray Object Classification#
importance <- xgb.importance(feature_names = colnames(xdata), model = model)
importance
