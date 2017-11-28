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

# print the percentage of zero-accident intersections
print(nrow(crash.data[crash.data$accidents %in% 0,])/nrow(crash.data))

#remove zero-accident intersections
#crash.data<-subset(crash.data, accidents!=0)

# GLM
glmfit<- glm(accidents ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max,
             data = crash.data,
             family = gaussian())

print(summary(glmfit))


#drop unneccassry columns
crash.data <- subset(crash.data, select = -c(11:120))
crash.data <- subset(crash.data, select = -c(1))

#crash.data[["accidents"]] = factor(crash.data[["accidents"]])

#crash.data$accidents<-cut(crash.data$accidents, c(0,1,2,4,5,10))

crash.data$accidents <- cut(crash.data$accidents, c(-Inf, 0, 1, 3, 6, 10, Inf),
                    labels=c('<1', '1', '2-3', '4-5','6-10', '>10'))

#write.csv(crash.data, file= "bikecrash.csv")
# required package for svm: install.packages('caret', dependencies = TRUE)

# construct training and testing dataset
set.seed(5)
intrain <- createDataPartition(y = crash.data$accidents, p= 0.8, list = FALSE)
training <- crash.data[intrain,]
testing <- crash.data[-intrain,]
print(dim(training) + dim(testing))


# I'm not sure if we run it as a regression, how to measure the predication performance, so I coverted accidents into categorical in order to run classification.
# training[["accidents"]] = factor(training[["accidents"]])
# testing[["accidents"]] = factor(testing[["accidents"]])

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
set.seed(5)

# SVM
svmfit <- train(accidents ~., data = training, method = "svmLinear",
                trControl=trctrl,
                preProcess = c("center", "scale"),
                tuneLength = 5,
                probability = TRUE)
print(summary(svmfit))

#svmfit <- svm(accidents ~., data = training)

test_pred <- predict(svmfit, testing)

plot(test_pred)

#testing <- transform(testing,pred_acc = test_pred)
print(confusionMatrix(data = test_pred, reference = testing$accidents))
print(confusionMatrix(data = test_pred, reference = testing$accidents, mode = "prec_recall"))
