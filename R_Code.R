
library(ISLR)
data(College)
df <- College


# Create Train and Test set - maintain % of event rate (70/30 split)
set.seed(123)
trainIndex <- sample(x = nrow(df), size = nrow(df)* 0.7)
train <- df[trainIndex,]
test <- df[- trainIndex,]

train_x <- model.matrix(Grad.Rate ~ ., train)[,-1]

test_x <- model.matrix(Grad.Rate ~ ., test)[,-1]


train_y <- train$Grad.Rate
test_y <- test$Grad.Rate


# Ridge Regression


# find best values of lambda
# find the best lambda using cross-Validation

# install.packages("glmnet", dependencies=TRUE)

library(glmnet)

par(mfrow=c(1,1))

# set.seed(1234)
cv.ridge <- cv.glmnet(train_x, train_y,alpha = 0,nfolds = 10) 
cv.ridge$lambda.min
cv.ridge$lambda.1se
plot(cv.ridge)


# Fit models based on lambda

fit.ridge <- glmnet(train_x, train_y,alpha = 0) 

plot(fit.ridge, xvar = "lambda", label = TRUE)

# Fit the final model on the training data using lambda.min and lambda.1se
# alpha = 1 for lasso(L2)
# alpha = 0 for Ridge(L1)


model.ridge.min <- glmnet(train_x,train_y , alpha = 0 , lambda = cv.ridge$lambda.min)
coef(model.ridge.min)

model.ridge.1se <- glmnet(train_x,train_y , alpha = 0 , lambda = cv.ridge$lambda.1se)
coef(model.ridge.1se)

# Train set predictions
# Determine the performance of the fit model against the training set by using RMSE and lambda min
# install.packages("Metrics")
library(Metrics)
pred.ridge.min.train <- predict(model.ridge.min , newx = train_x)
train.ridge.rmse.min <- rmse(train_y , pred.ridge.min.train)
train.ridge.rmse.min

pred.ridge.1se.train <- predict(model.ridge.1se , newx = train_x)
train.ridge.rmse.1se.train <- rmse(train_y , pred.ridge.1se.train)
train.ridge.rmse.1se.train

# Determine the performance of the fit model against the test set by using RMSE and lambda min
pred.ridge.min.test <- predict(model.ridge.min , newx = test_x)
test.ridge.rmse.min <- rmse(test_y , pred.ridge.min.test)
test.ridge.rmse.min

pred.ridge.1se.test <- predict(model.ridge.1se , newx = test_x)
test.ridge.rmse.1se.test <- rmse(test_y , pred.ridge.1se.test)
test.ridge.rmse.1se.test

# compare 


# lasso Regression


# find best values of lambda
# find the best lambda using cross-Validation

# install.packages("glmnet", dependencies=TRUE)


set.seed(123)
cv.lasso <- cv.glmnet(train_x, train_y,alpha = 1,nfolds = 10) 
cv.lasso$lambda.min
cv.lasso$lambda.1se
plot(cv.lasso)

# Fit models based on lambda

fit.lasso <- glmnet(train_x, train_y,alpha = 1) 


plot(fit.lasso, xvar = "lambda", label = TRUE)

# Fit the final model on the training data using lambda.min and lambda.1se
# alpha = 1 for lasso(L2)
# alpha = 0 for Ridge(L1)

model.lasso.min <- glmnet(train_x,train_y , alpha = 1 , lambda = cv.lasso$lambda.min)
coef(model.lasso.min)

model.lasso.1se <- glmnet(train_x,train_y , alpha = 1 , lambda = cv.lasso$lambda.1se)
coef(model.lasso.1se)

# Train set predictions
# make prediction on the test data using lambda.min

pred.lasso.min.train <- predict(model.lasso.min , newx = train_x)
train.lasso.rmse.min.train <- rmse(train_y , pred.lasso.min.train)
train.lasso.rmse.min.train

pred.lasso.1se.train <- predict(model.lasso.1se , newx = train_x)
train.lasso.rmse.1se.train <- rmse(train_y , pred.lasso.1se.train)
train.lasso.rmse.1se.train

pred.lasso.min.test <- predict(model.lasso.min , newx = test_x)
test.lasso.rmse.min.test <- rmse(test_y , pred.lasso.min.test)
test.lasso.rmse.min.test

pred.lasso.1se.test <- predict(model.lasso.1se , newx = test_x)
test.lasso.rmse.1se.test <- rmse(test_y , pred.lasso.1se.test)
test.lasso.rmse.1se.test



#  compare stepwise selection method
step(lm(Grad.Rate ~ ., data = df), direction = 'both') 
model_step <- step(lm(Grad.Rate ~ ., data = df), direction = 'both') 
summary(model_step) 

#use fitted best model to make predictions

y_predicted <- predict(model.lasso.min ,s = cv.lasso$lambda.min, newx = train_x)

#find SST and SSE
sst <- sum((train_y - mean(train_y))^2)
sse <- sum((y_predicted - train_y)^2)

#find R-Squared
rsq <- 1 - sse/sst
rsq

