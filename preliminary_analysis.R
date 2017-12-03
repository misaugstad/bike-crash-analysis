library(readr)
library(tidyverse)
library(dplyr)
library(caret)
library(e1071)
library(Matrix)
library(xgboost)
library(party)
library(randomForest)

# required package for svm: install.packages('caret', dependencies = TRUE)

independent.vars <- c('total_population',
                      'housing_units',
                      'household_income',
                      'NEAR_DIST',
                      'Width_max',
                      'Rating_min',
                      'Speed_max',
                      'ACC_max',
                      'OneWay_max')
dependent.var <- c('accidents')
lat.lng.vars <- c('Latitude', 'Longitude')
accident.year.cols <- c('Accidents_12_B15', 'Accidents_13_B15', 'Accidents_14_B15', 'Accidents_15_B15',
                          'Accidents_16_B15', 'Accidents_17_B15')

# add read_csv for handling big data
crash.data <- read_csv(file = "data/All_data.csv", col_names = TRUE)

# Buffer 15 meters
crash.data <-
  crash.data %>%
  mutate(accidents = # years 2012-2017
           Accidents_12_B15 + Accidents_13_B15 + Accidents_14_B15 + Accidents_15_B15 +
           Accidents_16_B15 + Accidents_17_B15) %>%
  select(one_of(independent.vars, dependent.var)) %>%
  na.omit()

print(table(crash.data$accidents))

# Buffer 30 meters
# crash.data <- transform(crash.data,
#                         accidents = Accidents_12_B30 + Accidents_13_B30 + Accidents_14_B30 + Accidents_15_B30 + Accidents_16_B30 + Accidents_17_B30)
# print(table(crash.data$accidents))


# print the percentage of zero-accident intersections
print(nrow(crash.data[crash.data$accidents %in% 0,])/nrow(crash.data))
backup.data <- crash.data[, c(1:12)]

# =============== Classification ========================
# classify zero- and non-zero-accidents interactions
crash.data.with.class <-
  crash.data %>%
  mutate(accidents_class = cut(accidents, c(-Inf, 0, Inf), labels = c('0', '>0'))) %>%
  select(-accidents)

print(table(crash.data.with.class$accidents_class))

# sampling
set.seed(9560)
# crash.data <- downSample(x = crash.data[, -ncol(crash.data)], y = crash.data$accidents_class)
# crash.data <- as.data.frame(crash.data)
# colnames(crash.data)  <-c("Zero","Nonzero")
# crash.data.down.sampled <- downSample(x = crash.data.with.class[, -ncol(crash.data)],
#                                       y = crash.data.with.class$accidents_class)
crash.data.down.sampled <-
  downSample(x = crash.data.with.class %>% select(one_of(independent.vars)),
             y = crash.data.with.class$accidents_class)

table(crash.data.down.sampled$Class)

#crash.data[["accidents"]] = factor(crash.data[["accidents"]])
#crash.data$accidents<-cut(crash.data$accidents, c(0,1,2,4,5,10))
#write.csv(crash.data, file= "bikecrash.csv")

# construct training and testing dataset
set.seed(9560)
intrain <- createDataPartition(y = crash.data.down.sampled$Class, p = 0.8, list = FALSE)
training <- crash.data.down.sampled[intrain,]
testing <- crash.data.down.sampled[-intrain,]
print(nrow(training) + nrow(testing))

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

# run SVM
svmfit <- train(Class ~., data = training, method = "svmRadial",
                trControl = trctrl,
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
impt <- xgb.importance(feature_names = colnames(training.sparse), model = xgb.crash.model )
xgb.plot.importance(importance_matrix = impt)

# Predict on test dataset.
predicty <- predict(xgb.crash.model, testing.sparse)

# plot(predicty)

# Get accuracy of prediction.
print(confusionMatrix(data = factor(predicty > 0.5, levels = c(FALSE, TRUE), labels = c('0', '>0')),
                      reference = testing$Class, positive = '>0'))
print(confusionMatrix(data = factor(predicty > 0.95, levels = c(FALSE, TRUE), labels = c('0', '>0')),
                      reference = testing$Class, mode = "prec_recall", positive = '>0'))

# random forest

# build forest
output.forest <- randomForest(Class ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max,
                              data = training, importance = T, proximity = T, ntree = 300, mtry = 2, do.trace = 100)

# running test data
predictRF <- predict(output.forest, testing)

# predictRF <- predict(output.forest, crash.data)


# Predicition accuracy
confusionMatrix(data = predictRF,
                reference = testing$Class,
                positive = '>0')


#confusionMatrix(data=predictRF,
#                reference=crash.data$accidents_class,
#                positive='>0')

# plot how the error changes
plot(output.forest, log = "y")

# showing the importance of each factor
varImpPlot(output.forest)

# basic information of randomforest
print(output.forest)


# =============== Regression ========================
# remove zero-accident intersections
crash.data.for.regression <- subset(backup.data, accidents != 0)

# GLM
glmfit <- glm(accidents ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max,
              data = crash.data.for.regression,
              family = gaussian())

print(summary(glmfit))

# Multiple Linear Regression
lmfit <- lm(accidents ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max,
            data = crash.data.for.regression)
print(summary(lmfit))

# perform step-wise feature selection
# lmfit <- step(lmfit)
# make predictions
predictions <- predict(lmfit, crash.data.for.regression)
# summarize accuracy
mse <- mean((crash.data.for.regression$accidents - predictions)^2)
RMSE <- sqrt(mse)
print(RMSE)


# random forest

output.forest <- randomForest(accidents ~ total_population + housing_units + household_income + NEAR_DIST + Width_max + Rating_min + Speed_max + ACC_max + OneWay_max,
                              data = crash.data.for.regression, importance = T, proximity = T, ntree = 500, mtry = 2, do.trace = 100)


plot(predictRF)
plot(output.forest, log = "y")
varImpPlot(output.forest)
print(output.forest)
round(importance(output.forest))
plot(importance(output.forest))

