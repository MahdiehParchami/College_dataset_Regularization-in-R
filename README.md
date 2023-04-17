# College_dataset_Regularization-in-R
implementing Regularization on College dataset in R

I use the “college” data set from the ISLR library that has information from many US 
Colleges. The data set includes one categorical variable and 17 numeric variables. I wish to fit a model 
that will predict the graduation rate of a college based on the rest of the information available in the 
dataset, using the regularization methods of Ridge and LASSO to prevent the model from being too 
complex and overfit.

To conduct regularization methods, it is essential to choose an appropriate value to prevent the models to 
be overfitting or underfitting. If the lambda is too high, the model might be too simple and underfitting. 
While if the lambda is too low, the model might be overfitting. There are two types of lambda values: 
lambda. min, which indicates the minimum mean cross-validated error, and lambda.1se, which is the 
largest value of lambda within 1 standard error of lambda min. I used cv.glmnet() function to find 
lambda.min and lambda.1se. Whereas the Ridge penalty pushes variables to approximately but not equal 
to zero, the lasso penalty will push coefficients to zero. Switching to the lasso penalty not only improves 
the model but also conducts automated feature selection however Ridge regression does not perform 
feature selection and will retain all available features in the final model. 
In the Ridge method, once log(λ) gets larger, MSE starts to increase, and all features will remain in the 
model. In the Lasso method, once log(λ) gets larger, MSE starts to increase, and the number of variables 
retained in the lasso model decrease. The Ridge method will not get rid of irrelevant features but 
rather minimize their impact on the trained model. Lasso method overcomes the disadvantage of Ridge 
regression by not only punishing high values of the coefficients but setting them to zero if they are not 
relevant. 
by implementing Ridge and Lasso methods, I found that there was no significant difference in RMSE from 
both regressions. Therefore, the accuracy performance of the two models is similar. However, since Lasso 
regression used fewer variables and produced simpler, easier to interpret models, with only a set of 
simplified predictions, I think Lasso regression predicts better than ridge regression. By using lasso 
lambda 1se, we predicted Graduate rates of universities with 9 variables. (Private, Top10perc , Top25perc, 
Undergrad, Outstate , Room.Board, Personal, perc.alumni)
