library(tidyverse)
library(dplyr)
library(caret)
library(e1071)
library(Matrix)
library(xgboost)

# required package for svm: install.packages('caret', dependencies = TRUE)

crash.data<- read.csv(file="data/All_data.csv", header=TRUE, sep=",")
#crash.data[["incidence"]][is.na(crash.data[["incidence"]])] <- 0
#crash.data$accidents <- rowSums(crash.data[11:120])
crash.data<- na.omit(crash.data)

# Buffer 15 meters
crash.data <- transform(crash.data,
                        accidents = Accidents_12_B15 + Accidents_13_B15 + Accidents_14_B15 + Accidents_15_B15 + Accidents_16_B15 + Accidents_17_B15)
print(table(crash.data$accidents))

# Buffer 30 meters
# crash.data <- transform(crash.data,
#                         accidents = Accidents_12_B30 + Accidents_13_B30 + Accidents_14_B30 + Accidents_15_B30 + Accidents_16_B30 + Accidents_17_B30)
# print(table(crash.data$accidents))

#drop unneccassry columns
crash.data <- subset(crash.data, select = -c(13:120))
crash.data <- subset(crash.data, select = -c(1))

# print the percentage of zero-accident intersections
print(nrow(crash.data[crash.data$accidents %in% 0,])/nrow(crash.data))
backup.data <- crash.data[, c(1:12)]

# =============== Classification ========================
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

# drop original accidents
crash.data <- subset(crash.data, select = -c(accidents))

#crash.data[["accidents"]] = factor(crash.data[["accidents"]])
#crash.data$accidents<-cut(crash.data$accidents, c(0,1,2,4,5,10))
#write.csv(crash.data, file= "bikecrash.csv")

# construct training and testing dataset
set.seed(9560)
intrain <- createDataPartition(y = crash.data$Class, p= 0.8, list = FALSE)
training <- crash.data[intrain,]
testing <- crash.data[-intrain,]
print(dim(training) + dim(testing))

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
set.seed(9560)

# SVM
svmfit <- train(Class ~., data = training, method = "svmRadial",
                trControl=trctrl,
                preProcess = c("center", "scale"),
                tuneLength = 5,
                probability = TRUE)
print(summary(svmfit))

test_pred <- predict(svmfit, testing)

plot(test_pred)

#testing <- transform(testing,pred_acc = test_pred)
print(confusionMatrix(data = test_pred, reference = testing$Class))
print(confusionMatrix(data = test_pred, reference = testing$Class, mode = "prec_recall"))


# XGBOOST!
training.sparse <- sparse.model.matrix(Class ~ .-1, data = training)
testing.sparse <- sparse.model.matrix(Class ~ .-1, data = testing)

# Create the model.
xgb.crash.model <- xgboost(data = training.sparse,
                           label = as.numeric(training$Class) - 1,
                           max_depth = 4,
                           nthread = 2,
                           nrounds = 200,
                           objective = "binary:logistic",
                           verbose = 0)

# Visualize which factors are most important.
impt <- xgb.importance(feature_names = colnames(training.sparse), model = blah)
xgb.plot.importance(importance_matrix = impt)

# Predict on test dataset.
predicty <- predict(xgb.crash.model, testing.sparse)

# Get accuracy or prediction.
mean(as.numeric(predicty > 0.5) == as.numeric(testing$Class) - 1)

# =============== Regression ========================
# remove zero-accident intersections
crash.data<-subset(backup.data, accidents!=0)

# GLM
glmfit<- glm(accidents ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max,
             data = crash.data,
             family = gaussian())

print(summary(glmfit))

# Multiple Linear Regression
lmfit <- lm(accidents ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max, data=crash.data)
print(summary(lmfit))

# perform step-wise feature selection
# lmfit <- step(lmfit)
# make predictions
predictions <- predict(lmfit, crash.data)
# summarize accuracy
mse <- mean((crash.data$accidents - predictions)^2)
RMSE <- sqrt(mse)
print(RMSE)
