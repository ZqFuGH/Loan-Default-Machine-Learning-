##### ECO3080 Project: Loan Default Prediction 

#################### 1. Load dataset and packages   ############################
getwd() # lists the current working directory
setwd("/Users/mac/Desktop/year 4/ECO3080/project") 
Loan <- read_dta('data.dta')

# library
library('haven')
library(dplyr)
library(corrplot)
library(ROSE)
library(ggplot2)

library(ISLR2)
library(tree)
library(rpart)
library(rpart.plot)
library(MASS)
library(randomForest)
library(gbm)
library(keras)
library(pROC)
library('caret')
library('class')
library(kknn)
library('ramify')
library(boot)
library(stargazer)
library(jtools)
library(DescTools)
library(sjstats)
library(sjPlot)
library(regclass)
library(huxtable)
library(lattice)
library(sjlabelled)
library(MLeval)
library(klaR)
library(recipes)
library(gmodels)
library(patchwork)

########################### 2. Data prepossessing   ############################
### NA values
Loan <- Loan %>% mutate_all(na_if,"") 
## 
for (i in 1:ncol(Loan)) {
  na_num = nrow(Loan[is.na(Loan[,i]),])
  col_name = names(Loan[,i])
  percent_na = na_num/nrow(Loan[,i])
  print(col_name)
  print(percent_na)
}
## omit NA values
Loan <- na.omit(Loan)

### delete redundant rows
Loan <- Loan[!(Loan$purpose=='wedding'),]
Loan <- Loan[!(Loan$home_ownership=='ANY'),]

### change variables
Loan$fico_score <- (Loan$fico_range_low + Loan$fico_range_high)/2

Loan$emp_length_char <- 
  as.factor(ifelse(Loan$emp_length == "< 1 year" | Loan$emp_length == "1 year", 1,
                   ifelse(Loan$emp_length == "2 years" | Loan$emp_length == "3 years", 2,
                          ifelse(Loan$emp_length == "4 years"| Loan$emp_length == "5 years", 3,
                                 ifelse(Loan$emp_length == "10+ years", 4, 5)))))
table(Loan$emp_length_char)

Loan$annual_inc_char <- 
  as.factor(ifelse(Loan$annual_inc < 20000 , "Poverty",
                   ifelse(Loan$annual_inc >= 20000 & Loan$annual_inc < 45000, 'Low income',
                          ifelse(Loan$annual_inc >= 45000 & Loan$annual_inc < 140000, 'Middle class',
                                 ifelse(Loan$annual_inc >= 140000 & Loan$annual_inc <150000, 'Upper middle class',
                                        ifelse(Loan$annual_inc >= 150000 & Loan$annual_inc < 200000, 'Highe income', 'Highest tax brackets'))))))
table(Loan$annual_inc_char)

## delete redundant columns
Loan <- Loan[,-c(1,5,6,8,11,14,15)]

### convert characters to factors
Loan[sapply(Loan, is.character)] <- lapply(Loan[sapply(Loan, is.character)], 
                                           as.factor)
### sketch
str(Loan)

### save the cleaned dataset into csv
write.csv(Loan, file = "/Users/mac/Desktop/year 4/ECO3080/project/cleaned_data.csv")


########################### 4. Data exploration   ##############################
### correlations of numeric variables
cor_loan = cor(Loan[,c(1, 7:17)])
corrplot(cor_loan,method="number",diag=FALSE,
         tl.cex=0.8,number.cex=0.8,cl.pos="b",cl.length=5)

### summary statistics of numeric data
par(mfrow=c(2,5))
for (i in c(2,8)){
  name=names(Loan)[i]
  col_unlist <- unlist(Loan[, i])
  col_num <- as.numeric(col_unlist)
  hist(col_num,col = "deepskyblue1",main=name)
}

### summary statistics of factor data


############################### 5. Sampling   ##################################
set.seed (911)
### Set 70% of the data as the training set, 30% as the test set
num <- round(0.7*nrow(Loan))
train <- sample(1:nrow(Loan), num, replace = FALSE)
Trainset <- Loan[train, ]
Testset <- Loan[-train, ]
status.test <- Testset$loan_status

# Since dataset is imbalanced on the dependent variable loan_status,
# we use oversampling technique
table(Loan$loan_status)
Trainset_over <- ovun.sample(loan_status~., data = Trainset, 
                             method = "over", N = 140000)$data

############################### 6. Modeling   ##################################
###################### 6.1 Logistic regression   ###############################
#backward selection
null_model <- glm(loan_status ~ 1, data = Trainset_over, family = "binomial")
full_model <- glm(loan_status ~ ., data = Trainset_over, family = "binomial")
step_model <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "backward")

tr1<-glm(data=Trainset_over, family = "binomial", loan_status~loan_amnt+term+grade+
           home_ownership+annual_inc_char+dti+delinq_2yrs+inq_last_6mths+open_acc+
           total_acc+mort_acc+fico_score+bc_util+num_bc_tl+num_il_tl)
summary(tr1)

#manual selection (by correlation table & logit results)
tr2<-glm(data=Trainset_over, family = "binomial", loan_status~loan_amnt+term+
           home_ownership+annual_inc_char+dti+delinq_2yrs+fico_score+inq_last_6mths+
           open_acc+mort_acc+num_bc_tl+num_il_tl)
summary(tr2)


# comparing AIC the backward selection model is better
te1<-predict.glm(tr1,type = "response", newdata=Testset)

te1_cat = ifelse(te1 > 0.5, "Fully Paid", "Charged Off")
te1_cat<-as.factor(te1_cat)

confusionMatrix(te1_cat, status.test, positive = "Charged Off")

# roc curve and auc 
test_prob1 = predict(tr1, newdata = Testset, type = "response")
test_roc1 = roc(status.test ~ test_prob1, plot = TRUE, print.auc = TRUE)
auc(test_roc1)

plot(test_roc1, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')

#AIC: 176349, Accuracy : 0.3214, Sensitivity : 0.43602

############################  6.2 Naive Bayes   ################################
#selection procedures
#candidate 1: drop 2 var by correlation table

#candidate 2: based on candidate 1, drop 2 numeric variables
Loan %>%
  ggplot()+
  geom_smooth(aes(loan_amnt, purpose,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(loan_amnt, inq_last_6mnths,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(open_acc, inq_last_6mths,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(loan_amnt, open_acc,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(open_acc, delinq_amnt,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(open_acc, mort_acc,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(open_acc, num_bc_tl,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(open_acc, num_il_tl,col=loan_status))

Loan %>%
  ggplot()+
  geom_smooth(aes(open_acc, dti,col=loan_status))

#candidate 3: based on candidate 2, drop another 3 factor var
p1<-ggplot(Trainset_over,aes(x=purpose,fill=loan_status))+geom_bar()
p2<-ggplot(Trainset_over,aes(x=home_ownership,fill=loan_status))+geom_bar()
p3<-ggplot(Trainset_over,aes(x=term,fill=loan_status))+geom_bar()
p4<-ggplot(Trainset_over,aes(x=grade,fill=loan_status))+geom_bar()
p5<-ggplot(Trainset_over,aes(x=emp_length_char,fill=loan_status))+geom_bar()
p6<-ggplot(Trainset_over,aes(x=annual_inc_char,fill=loan_status))+geom_bar()
p1+p2+p3+p4+p5+p6

#candidate 4: based on candidate 1, drop 3 factor var only

# use cross validation to calculate model sensitivity and accuracy for comparison
fold = 5
n_model = 4
set.seed(911)
folds <- sample(1:fold, nrow(Trainset_over), replace = TRUE)
cv.accuracy <- matrix(NA, fold, n_model, dimnames = list(NULL, paste(1:n_model)))
cv.sensitivity <- matrix(NA, fold, n_model, dimnames = list(NULL, paste(1:n_model)))

for (j in 1:fold) {
  # Candidate 1: from full model, delete total_acc, num_bc_tl based on correlations
  Bayes1 <- NaiveBayes(loan_status~loan_amnt+term+grade+home_ownership+purpose+dti+inq_last_6mths+mort_acc+
                         open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+emp_length_char+annual_inc_char,
                       data = Trainset_over[folds != j, ])
  pre_Bayes1 <- predict(Bayes1, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_Bayes1$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 1] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 1] <- confusion$byClass["Sensitivity"]
  # Candidate 2: from Candidate 1, delete delinq_amnt, mort_acc, num_bc_tl, num_il_tl, dti based on graphical display
  Bayes2 <- NaiveBayes(loan_status~loan_amnt+term+grade+purpose+home_ownership+annual_inc_char+
                         delinq_2yrs+inq_last_6mths+open_acc+fico_score+emp_length_char,
                       data = Trainset_over[folds != j, ])
  pre_Bayes2 <- predict(Bayes2, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_Bayes2$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 2] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 2] <- confusion$byClass["Sensitivity"]
  # Candidate 3: from Candidate 2, delete purpose, emp_length_char, home_ownership
  Bayes3 <- NaiveBayes(loan_status~loan_amnt+term+grade+annual_inc_char+
                         delinq_2yrs+inq_last_6mths+open_acc+fico_score,
                       data = Trainset_over[folds != j, ])
  pre_Bayes3 <- predict(Bayes3, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_Bayes3$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 3] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 3] <- confusion$byClass["Sensitivity"]
  # Candidate 4: from Candidate 1, delete purpose, emp_length_char, home_ownership
  Bayes4 <- NaiveBayes(loan_status~loan_amnt+term+grade+dti+inq_last_6mths+mort_acc+
                         open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+annual_inc_char,
                       data = Trainset_over[folds != j, ])
  pre_Bayes4 <- predict(Bayes4, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_Bayes4$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 4] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 4] <- confusion$byClass["Sensitivity"]
}

# compare accuracy and sensitivity of the candidates
mean.cv.accuracy <- apply(cv.accuracy, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.accuracy, type = "b")

mean.cv.sensitivity <- apply(cv.sensitivity, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.sensitivity, type = "b")

# best fit is Candidate 4
Bayes_best <- NaiveBayes(loan_status~loan_amnt+term+grade+dti+inq_last_6mths+mort_acc+
                           open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+annual_inc_char,
                         data = Trainset_over)
Bayes_best[1:length(Bayes_best)]
par(mfrow = c(3, 4))

pre_Bayes_best <- predict(Bayes_best, Testset)
confusionMatrix(pre_Bayes_best$class, Testset$loan_status)
bay_prob <- pre_Bayes_best$posterior[,2]

roc_bay = roc(Testset$loan_status,
              bay_prob)
auc(roc_bay)

#AUC:0.7024	accuracy: 0.6625	sensitivity: 0.6254	mean.cv.accuracy: 0.6509611

plot(roc_bay, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')


######################### 6.3 Discriminant Analysis  ###########################
##################### 6.3.1 Linear Discriminant Analys##########################
#follow the previous candidate selection procedures

#lda
fold = 5
n_model = 4
set.seed(911)
folds <- sample(1:fold, nrow(Trainset_over), replace = TRUE)
cv.accuracy <- matrix(NA, fold, n_model, dimnames = list(NULL, paste(1:n_model)))
cv.sensitivity <- matrix(NA, fold, n_model, dimnames = list(NULL, paste(1:n_model)))


for (j in 1:fold) {
  lda1 <- lda(loan_status~loan_amnt+term+grade+home_ownership+purpose+dti+inq_last_6mths+mort_acc+
                open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+emp_length_char+annual_inc_char,
              data = Trainset_over[folds != j, ])
  pre_lda1 <- predict(lda1, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_lda1$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 1] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 1] <- confusion$byClass["Sensitivity"]
  
  lda2 <- lda(loan_status~loan_amnt+term+grade+purpose+home_ownership+annual_inc_char+
                delinq_2yrs+inq_last_6mths+open_acc+fico_score+emp_length_char,
              data = Trainset_over[folds != j, ])
  pre_lda2 <- predict(lda2, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_lda2$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 2] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 2] <- confusion$byClass["Sensitivity"]
  
  lda3 <- lda(loan_status~loan_amnt+term+grade+annual_inc_char+
                delinq_2yrs+inq_last_6mths+open_acc+fico_score,
              data = Trainset_over[folds != j, ])
  pre_lda3 <- predict(lda3, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_lda3$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 3] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 3] <- confusion$byClass["Sensitivity"]
  
  lda4 <- lda(loan_status~loan_amnt+term+grade+dti+inq_last_6mths+mort_acc+
                open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+annual_inc_char,
              data = Trainset_over[folds != j, ])
  pre_lda4 <- predict(lda4, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_lda4$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 4] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 4] <- confusion$byClass["Sensitivity"]
  
}

# compare accuracy and sensitivity of the candidates
mean.cv.accuracy <- apply(cv.accuracy, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.accuracy, type = "b")

mean.cv.sensitivity <- apply(cv.sensitivity, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.sensitivity, type = "b")

# the best model is candidate 1
lda_best <- lda(loan_status~loan_amnt+term+grade+home_ownership+purpose+dti+inq_last_6mths+mort_acc+
                  open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+emp_length_char+annual_inc_char,
                data = Trainset_over)

pred_lda_best <- predict(lda_best, newdata=Testset) 
names(pred_lda_best)
lda_class <- pred_lda_best$class
lda_prob <- pred_lda_best$posterior[,2]
confusionMatrix(lda_class, Testset$loan_status)

roc_lda = roc(Testset$loan_status,
              lda_prob)
plot(roc_lda)
auc(roc_lda)

#AUC: 0.7193	accuracy: 0.677	sensitivity: 0.6262	mean.cv.accuracy: 0.6595896 

plot(roc_lda, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')


##################### 6.3.2 Quadratic Discriminant Analys#######################
#follow the previous candidate selection procedures
fold = 5
n_model = 4
set.seed(911)
folds <- sample(1:fold, nrow(Trainset_over), replace = TRUE)
cv.accuracy <- matrix(NA, fold, n_model, dimnames = list(NULL, paste(1:n_model)))
cv.sensitivity <- matrix(NA, fold, n_model, dimnames = list(NULL, paste(1:n_model)))

for (j in 1:fold) {
  qda1 <- qda(loan_status~loan_amnt+term+grade+home_ownership+purpose+dti+inq_last_6mths+mort_acc+
                open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+emp_length_char+annual_inc_char,
              data = Trainset_over[folds != j, ])
  pre_qda1 <- predict(qda1, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_qda1$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 1] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 1] <- confusion$byClass["Sensitivity"]
  
  qda2 <- qda(loan_status~loan_amnt+term+grade+purpose+home_ownership+annual_inc_char+
                delinq_2yrs+inq_last_6mths+open_acc+fico_score+emp_length_char,
              data = Trainset_over[folds != j, ])
  pre_qda2 <- predict(qda2, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_qda2$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 2] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 2] <- confusion$byClass["Sensitivity"]
  
  qda3 <- qda(loan_status~loan_amnt+term+grade+annual_inc_char+
                delinq_2yrs+inq_last_6mths+open_acc+fico_score,
              data = Trainset_over[folds != j, ])
  pre_qda3 <- predict(qda3, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_qda3$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 3] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 3] <- confusion$byClass["Sensitivity"]
  
  qda4 <- qda(loan_status~loan_amnt+term+grade+dti+inq_last_6mths+mort_acc+
                open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+annual_inc_char,
              data = Trainset_over[folds != j, ])
  pre_qda4 <- predict(qda4, Trainset_over[folds == j, ])
  confusion <- confusionMatrix(pre_qda4$class, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
  cv.accuracy[j, 4] <- confusion$overall["Accuracy"]
  cv.sensitivity[j, 4] <- confusion$byClass["Sensitivity"]
}

# compare accuracy and sensitivity of the candidates
mean.cv.accuracy <- apply(cv.accuracy, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.accuracy, type = "b")

mean.cv.sensitivity <- apply(cv.sensitivity, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.sensitivity, type = "b")

# the best model candidate 4
qda_best <- qda(loan_status~loan_amnt+term+grade+dti+inq_last_6mths+mort_acc+
                  open_acc+bc_util+delinq_2yrs+delinq_amnt+num_il_tl+fico_score+annual_inc_char,
                data = Trainset_over)

pred_qda_best <- predict(qda_best, newdata=Testset)
qda_class <- pred_qda_best$class
qda_prob <- pred_qda_best$posterior[,2]
confusionMatrix(qda_class, Testset$loan_status)

roc_qda = roc(Testset$loan_status,
              qda_prob)
plot(roc_qda)
auc(roc_qda)

#AUC: 0.7018	 accuracy: 0.7243	 sensitivity:0.7243	mean.cv.accuracy:0.6462724

plot(roc_qda, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')


################################## 6.4 KNN  ####################################
# select k using cross validation
fold = 5
k_list = list(5, 10, 50, 100, 118, 250)
k_MAX = length(k_list)
set.seed(911)
folds <- sample(1:fold, nrow(Trainset_over), replace = TRUE)
cv.accuracy <- matrix(NA, fold, k_MAX, dimnames = list(NULL, paste(1:k_MAX)))
cv.sensitivity <- matrix(NA, fold, k_MAX, dimnames = list(NULL, paste(1:k_MAX)))

for (j in 1:fold) {
  print(j)
  for (i in 1:k_MAX){
    print(i)
    cv.fit <- kknn(loan_status ~ ., Trainset_over[folds != j, ], 
                   Trainset_over[folds == j, ], k=k_list[[i]])
    pred <- fitted(cv.fit)
    confusion <- confusionMatrix(pred, Trainset_over[folds == j, ]$loan_status, positive = "Charged Off")
    cv.accuracy[j, i] <- confusion$overall["Accuracy"]
    cv.sensitivity[j, i] <- confusion$byClass["Sensitivity"]
  }
}

mean.cv.accuracy <- apply(cv.accuracy, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.accuracy, type = "b")

mean.cv.sensitivity <- apply(cv.sensitivity, 2, mean)
par(mfrow = c(1, 1))
plot(mean.cv.sensitivity, type = "b")

best.fit <- kknn(loan_status ~ ., Trainset_over, 
                 Testset, k=5)
knn_pred <- fitted(best.fit)
confusionMatrix(knn_pred, 
                status.test, 
                positive = "Charged Off")
# roc
roc_knn <- roc(status.test, as.numeric(knn_pred))
plot(roc_knn)
auc(roc_knn)

plot(roc_knn, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = FALSE, main = 'ROC')


####################### 6.5 Tree based Methods #################################
#####################   6.5.1 classification tree   ############################
tree.loan <- rpart(formula = loan_status ~.,
                   data = Trainset_over,
                   method = 'class')
rpart.plot(x = tree.loan)

Predict.tree <- predict(tree.loan, Testset, type = "class")
confusionMatrix(Predict.tree, status.test, positive = 'Charged Off')

tree_pred_prob = predict(tree.loan,  
                         newdata = Testset,   
                         type = "prob")
# roc
roc_tree = roc(Testset$loan_status,
               tree_pred_prob[,"Charged Off"])
ggroc(roc_tree)
auc(roc_tree)
plot(roc_tree, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')

###############################   6.5.2 bagging   ##############################
bag.status <- randomForest(loan_status ~ ., data = Trainset_over, mtry = 18,
                           importance = TRUE) 
varImpPlot(bag.status)
predict.bag <- predict(bag.status, newdata = Testset, type = "class")
confusionMatrix(predict.bag, status.test, positive = 'Charged Off')

bag_pred_prob = predict(object = bag.status,
                        newdata = Testset,
                        type = "prob")
# roc
roc_bag = roc(Testset$loan_status,
              bag_pred_prob[,"Charged Off"])
ggroc(roc_bag)
auc(roc_bag)
plot(roc_bag, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')


############################   6.5.3 random forest   ###########################
randforest.status <- randomForest(loan_status ~ ., data = Trainset_over, mtry = 6,
                                  importance = TRUE)
varImpPlot(randforest.status)
predict.rf <- predict(randforest.status, newdata = Testset, type = "class")
confusionMatrix(predict.rf, status.test, positive = 'Charged Off')
credit_rf_pred_prob = predict(object = randforest.status,
                              newdata = Testset,
                              type = "prob")
# roc
roc_rf = roc(Testset$loan_status,
             credit_rf_pred_prob[,"Charged Off"])
ggroc(roc_rf)
auc(roc_rf)
plot(roc_rf, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')


########################## 6.6 Neural Networks #################################
set.seed(911)
random_state = 110

x_train_over <- scale(model.matrix(loan_status ~ .-1, data = Trainset_over))
Trainset_over$loan_status<-recode(Trainset_over$loan_status, "Fully Paid"= 1, "Charged Off" = 0)
y_train_over <- to_categorical(Trainset_over$loan_status)

x_test <- scale(model.matrix(loan_status ~ .-1, data = Testset))
Loan$loan_status<-recode(Loan$loan_status,"Fully Paid"= 1, "Charged Off" = 0)
y_test <- to_categorical(Loan[-train,]$loan_status)

modnn <-keras_model_sequential() 

modnn %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = ncol(x_train_over)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 2, activation = 'sigmoid')

#  Define the loss function
modnn %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# training
history <- modnn %>% fit(
  x_train_over, y_train_over, epochs = 50, batch_size = 32,
  validation_split = 0.1
)

print(history)
plot(history)

# performance
modnn %>%
  evaluate(x_test, y_test)
predictedscore <-  predict(modnn, x_test, type='prob')
predictedclasses <- as.factor(argmax(predictedscore))
levels(predictedclasses) <- list('Charged Off' = '1', 'Fully Paid' = '2')

colnames(predictedscore) <- c('Charged Off', 'Fully Paid')

confusionMatrix(predictedclasses, status.test)

#roc
roc_nn = roc(status.test,
             predictedscore[,"Charged Off"])
ggroc(roc_nn)
auc(roc_nn)
plot(roc_nn, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col = c("green", "red"), max.auc.polygon = TRUE,
     auc.polygon.col = "skyblue", print.thres = TRUE, main = 'ROC')





