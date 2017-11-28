library(tidyverse)
library(dplyr)
library(caret)
library(e1071)

crash.data<- read.csv(file="All_data.csv", header=TRUE, sep=",")
#crash.data[["incidence"]][is.na(crash.data[["incidence"]])] <- 0
#crash.data$accidents <- rowSums(crash.data[11:120])
crash.data<- na.omit(crash.data)
crash.data <- transform(crash.data,
                        accidents = Accidents_12_B15 + Accidents_13_B15 + Accidents_14_B15 + Accidents_15_B15 + Accidents_16_B15 + Accidents_17_B15)
print(table(crash.data$accidents))

#drop unneccassry columns
crash.data <- subset(crash.data, select = -c(13:120))
crash.data <- subset(crash.data, select = -c(1))

# print the percentage of zero-accident intersections
print(nrow(crash.data[crash.data$accidents %in% 0,])/nrow(crash.data))

# classify zero- and non-zero-accidents interactions
crash.data$accidents_class <- cut(crash.data$accidents, c(-Inf, 0, Inf),
                            labels=c('0', '>0'))
print(table(crash.data$accidents_class))

# sampling
set.seed(9560)
# crash.data <- downSample(x = crash.data[, -ncol(crash.data)], y = crash.data$accidents_class)
# crash.data <- as.data.frame(crash.data)
# colnames(crash.data)  <-c("Zero","Nonzero")
crash.data <- downSample(x = crash.data[, -ncol(crash.data)],
                         y = crash.data$accidents_class)
table(crash.data$Class)

# drop accidents
crash.data <- subset(crash.data, select = -c(accidents))

# remove zero-accident intersections
#crash.data<-subset(crash.data, accidents!=0)

# GLM
glmfit<- glm(Class ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max,
             data = crash.data,
             family = gaussian())

print(summary(glmfit))


#crash.data[["accidents"]] = factor(crash.data[["accidents"]])

#crash.data$accidents<-cut(crash.data$accidents, c(0,1,2,4,5,10))

# segment the data
# crash.data$accidents <- cut(crash.data$accidents, c(-Inf, 1, Inf),
#                     labels=c('1', '>1'))
# print(table(crash.data$accidents))

#write.csv(crash.data, file= "bikecrash.csv")
# required package for svm: install.packages('caret', dependencies = TRUE)

# construct training and testing dataset
set.seed(9560)
intrain <- createDataPartition(y = crash.data$Class, p= 0.8, list = FALSE)
training <- crash.data[intrain,]
testing <- crash.data[-intrain,]
print(dim(training) + dim(testing))


# I'm not sure if we run it as a regression, how to measure the predication performance, so I coverted accidents into categorical in order to run classification.
# training[["accidents"]] = factor(training[["accidents"]])
# testing[["accidents"]] = factor(testing[["accidents"]])

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
set.seed(9560)

# SVM
svmfit <- train(Class ~., data = training, method = "svmLinear",
                trControl=trctrl,
                preProcess = c("center", "scale"),
                tuneLength = 5,
                probability = TRUE)
print(summary(svmfit))

#svmfit <- svm(accidents ~., data = training)

test_pred <- predict(svmfit, testing)

plot(test_pred)

#testing <- transform(testing,pred_acc = test_pred)
print(confusionMatrix(data = test_pred, reference = testing$Class))
print(confusionMatrix(data = test_pred, reference = testing$Class, mode = "prec_recall"))
