library(readr)
library(tidyverse)
library(dplyr)
library(caret)
library(e1071)
library(Matrix)
library(xgboost)
library(reshape2)
library(party)
library(randomForest)
library(MASS)
library(glmnet)
library(rpart)

# required package for svm: install.packages('caret', dependencies = TRUE)

independent.vars <- c('census.block.population',
                      'census.block.num.housing.units',
                      'census.block.household.income',
                      'dist.to.bike.parking',
                      'road.width.max',
                      'pavement.rating.min',
                      'speed.limit.max',
                      'ACC_max',
                      'includes.oneway')
dependent.var <- c('accidents')
lat.lng.vars <- c('lat', 'lng')

# add read_csv for handling big data
crash.data <- read_csv(file = "data/All_data.csv", col_names = TRUE)

# Buffer 15 meters
crash.data <-
  crash.data %>%
  mutate(accidents = # years 2012-2017
           Accidents_12_B15 + Accidents_13_B15 + Accidents_14_B15 + Accidents_15_B15 +
           Accidents_16_B15 + Accidents_17_B15) %>%
  rename(census.block.population = total_population,
         census.block.num.housing.units = housing_units,
         census.block.household.income = household_income,
         dist.to.bike.parking = NEAR_DIST,
         road.width.max = Width_max,
         pavement.rating.min = Rating_min,
         speed.limit.max = Speed_max,
         ACC_max = ACC_max,
         includes.oneway = OneWay_max,
         lat = Latitude,
         lng = Longitude) %>%
  dplyr::select(one_of(independent.vars, dependent.var)) %>%
  na.omit()

print(table(crash.data$accidents))

# print the percentage of zero-accident intersections
print(nrow(crash.data[crash.data$accidents %in% 0,])/nrow(crash.data))
backup.data <- crash.data %>% dplyr::select_all()


# =============== Classification ========================
# classify zero- and non-zero-accidents interactions
crash.data.with.class <-
  crash.data %>%
  mutate(accidents.class = cut(accidents, c(-Inf, 0, Inf), labels = c('0', '>0'))) %>%
  dplyr::select(-accidents)

print(table(crash.data.with.class$accidents.class))

# sampling
set.seed(9560)
# crash.data <- downSample(x = crash.data[, -ncol(crash.data)], y = crash.data$accidents.class)
# crash.data <- as.data.frame(crash.data)
# colnames(crash.data)  <-c("Zero","Nonzero")
# crash.data.down.sampled <- downSample(x = crash.data.with.class[, -ncol(crash.data)],
#                                       y = crash.data.with.class$accidents.class)
crash.data.down.sampled <-
  downSample(x = crash.data.with.class %>% dplyr::select(one_of(independent.vars)),
             y = crash.data.with.class$accidents.class)

table(crash.data.down.sampled$Class)

#crash.data[["accidents"]] = factor(crash.data[["accidents"]])
#crash.data$accidents<-cut(crash.data$accidents, c(0,1,2,4,5,10))
#write.csv(crash.data, file= "bikecrash.csv")

# construct training and testing dataset
intrain <- createDataPartition(y = crash.data.down.sampled$Class, p = 0.8, list = FALSE)
training <- crash.data.down.sampled[intrain,]
testing <- crash.data.down.sampled[-intrain,]
print(nrow(training) + nrow(testing))

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

# run SVM
svmfit <- train(Class ~., data = training,
                method = "svmRadial",
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

# Feature Weights
gbmImp <- varImp(svmfit, scale = FALSE)
print(gbmImp)

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

# Look at how choosing a cutoff changes precision, recall, and overall accuracy
calculate.accuracies <- Vectorize( function(cutoff) {
  prediction.factor <- factor(predicty > cutoff, levels = c(FALSE, TRUE), labels = c('0', '>0'))
  conf.mat <- confusionMatrix(data = prediction.factor, reference = testing$Class, positive = '>0')
  conf.mat$byClass[c('Precision','Recall', 'F1', 'Balanced Accuracy')]
})
cutoffs <- seq(0, 1, 0.01)
xgb.accuracies <-
  calculate.accuracies(cutoff = cutoffs) %>%
  t() %>% # transpose
  data.frame() %>% # convert to data frame
  mutate(cutoff = cutoffs) # add column with the cutoff value

xgb.accuracies.melt <- melt(xgb.accuracies, id.vars = 'cutoff', value.name = 'accuracy')
ggplot(data = xgb.accuracies.melt, mapping = aes(x = cutoff, y = accuracy)) +
  geom_line(aes(colour = variable))

# random forest

# build forest
output.forest <-
  randomForest(Class ~ census.block.population + census.block.num.housing.units +
                 census.block.household.income + dist.to.bike.parking + road.width.max +
                 pavement.rating.min + speed.limit.max + ACC_max + includes.oneway,
               data = training, importance = T, proximity = T, ntree = 300, mtry = 2, do.trace = 100)

# running test data
predictRF <- predict(output.forest, testing)

# predictRF <- predict(output.forest, crash.data)


# Predicition accuracy
confusionMatrix(data = predictRF,
                reference = testing$Class,
                positive = '>0')


#confusionMatrix(data=predictRF,
#                reference=crash.data$accidents.class,
#                positive='>0')

# plot how the error changes
plot(output.forest, log = "y")

# showing the importance of each factor
varImpPlot(output.forest)

# basic information of randomforest
print(output.forest)


# =============== Ordinal Regression ================

crash.data.ord <-
  crash.data %>%
  mutate(accidents.class = cut(accidents, c(-Inf, 2, 5, Inf),
                               labels = c('0-2', '3-5', '6+'),
                               ordered_result = TRUE)) %>%
  dplyr::select(-accidents)

print(table(crash.data.ord$accidents.class))

# sampling
set.seed(9560)

crash.data.down.ord.sampled <-
  downSample(x = crash.data.ord %>% dplyr::select(one_of(independent.vars)),
             y = crash.data.ord$accidents.class)

table(crash.data.down.ord.sampled$Class)

# construct training and testing dataset
intrain.ord <- createDataPartition(y = crash.data.down.ord.sampled$Class, p = 0.8, list = FALSE)
training.ord <- crash.data.down.ord.sampled[intrain.ord,]
testing.ord <- crash.data.down.ord.sampled[-intrain.ord,]
print(nrow(training.ord) + nrow(testing.ord))

ord.reg <- polr(Class ~ census.block.population + census.block.num.housing.units +
                  census.block.household.income + dist.to.bike.parking + road.width.max +
                  pavement.rating.min + speed.limit.max + ACC_max + includes.oneway,
                data = training.ord, Hess = TRUE, method = 'logistic')
summary(ord.reg)

# prediction time!
prediction.ord <- predict(ord.reg, testing.ord)
summary(prediction.ord)
print(confusionMatrix(data = prediction.ord,
                      reference = testing.ord$Class, mode = "prec_recall"))

# =============== Regression ========================
# remove zero-accident intersections
crash.data.for.regression <-
  subset(backup.data, accidents != 0) %>%
  mutate(accidents = case_when(accidents %in% c(1, 2) ~ 1,
                               accidents %in% c(3, 4) ~ 2,
                               accidents %in% c(5, 6) ~ 3,
                               TRUE ~ 4))

print(table(crash.data.for.regression$accidents))

# what if we take out the intersections with more than 5 accidents?
#crash.data.for.regression <- subset(crash.data.for.regression, accidents < 7)

# training & testing
## 80% of the sample size
smp_size <- floor(0.8 * nrow(crash.data.for.regression))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(crash.data.for.regression)), size = smp_size)

train <- crash.data.for.regression[train_ind, ]
test <- crash.data.for.regression[-train_ind, ]

# const denominator
dem <- sum((test$accidents - mean(test$accidents))^2)

# GLM
glmfit <- glm(accidents ~ census.block.population + census.block.num.housing.units +
                census.block.household.income + dist.to.bike.parking + road.width.max +
                pavement.rating.min + speed.limit.max + ACC_max + includes.oneway,
              data = crash.data.for.regression,
              family = gaussian())

print(summary(glmfit))

# Multiple Linear Regression
lmfit <- lm(accidents ~ census.block.population + census.block.num.housing.units +
              census.block.household.income + dist.to.bike.parking + road.width.max +
              pavement.rating.min + speed.limit.max + ACC_max + includes.oneway,
            data = train)
print(summary(lmfit))

# make predictions
predictions <- predict(lmfit, test)
# summarize accuracy
mse <- mean((test$accidents - predictions)^2)
RMSE <- sqrt(mse)
print(RMSE)
R2 <- 1 - (sum((test$accidents - predictions )^2)/dem)
print(R2)

# Stepwise regression (usually deal with multiple IVs)
# perform step-wise feature selection
min.lm <- lm(accidents ~ 1,
             data = train)
biggest <- formula(lmfit)
fwd.stepfit <- step(min.lm, direction = 'forward', scope = biggest)
print(summary(fwd.stepfit))
back.stepfit <- step(min.lm, direction = 'backward', scope = biggest)
print(summary(back.stepfit))

# make predictions
predictions <- predict(fwd.stepfit, test)
# summarize accuracy
mse <- mean((test$accidents - predictions)^2)
RMSE <- sqrt(mse)
print(RMSE)
R2 <- 1 - (sum((test$accidents - predictions )^2)/dem)
print(R2)

# lasso, ridge, and elastic net regression
x <- as.matrix(train[1:9]) # feature matrix
y <- as.double(as.matrix(train[, 10])) # Only class

x.test <- as.matrix(test[1:9]) # feature matrix
y.test <- as.double(as.matrix(test[, 10])) # Only class

fit.lasso <- glmnet(x, y, family = "gaussian", alpha=1)
fit.ridge <- glmnet(x, y, family = "gaussian", alpha=0)
fit.elnet <- glmnet(x, y, family = "gaussian", alpha=.5)

# lambdas <- 10^seq(3, -2, by = -.1)
# cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas)

# 10-fold Cross validation for each alpha = 0, 0.2, ... , 0.8, 1.0
# (For plots on Right)
for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x, y, type.measure = "mse",
                                            alpha = i/10,family = "gaussian"))
}

# Plot solution paths:
par(mfrow = c(3,2))
# For plotting options, type '?plot.glmnet' in R console
plot(fit.lasso, xvar = "lambda")
plot(fit10, main = "LASSO")

plot(fit.ridge, xvar = "lambda")
plot(fit0, main = "Ridge")

plot(fit.elnet, xvar = "lambda")
plot(fit5, main = "Elastic Net")

# make predictions
predictions <- predict(fit.elnet, test)
# summarize accuracy
mse <- mean((test$accidents - predictions)^2)
RMSE <- sqrt(mse)
print(RMSE)
R2 <- 1 - (sum((test$accidents - predictions )^2)/dem)
print(R2)

# plot linear relationship
treefit <- pairs(accidents ~ census.block.population + census.block.num.housing.units +
                   census.block.household.income + dist.to.bike.parking + road.width.max +
                   pavement.rating.min + speed.limit.max + ACC_max + includes.oneway,
                data = train)

# Regression Tree
treefit <- rpart(accidents ~ census.block.population + census.block.num.housing.units +
                  census.block.household.income + dist.to.bike.parking + road.width.max +
                  pavement.rating.min + speed.limit.max + ACC_max + includes.oneway,
      data = train)

summary(treefit)
printcp(treefit)
plotcp(treefit)

# make prediction
predictions <- predict(treefit, test)
mse <- mean((test$accidents - predictions)^2)
RMSE <- sqrt(mse)
R2 <- 1 - (sum((test$accidents - predictions )^2)/dem)
print(R2)


# random forest
output.forest <-
  randomForest(accidents ~ census.block.population + census.block.num.housing.units +
                 census.block.household.income + dist.to.bike.parking + road.width.max +
                 pavement.rating.min + speed.limit.max + ACC_max + includes.oneway,
               data = crash.data.for.regression, importance = T, proximity = T, ntree = 500,
               mtry = 2, do.trace = 100)


plot(predictRF)
plot(output.forest, log = "y")
varImpPlot(output.forest)
print(output.forest)
round(importance(output.forest))
plot(importance(output.forest))

