# Commonly use Machine Learnign Algorithms (With Python and R code)
[Link to the bolg](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)

>Google’s self-driving cars and robots get a lot of press, but the company’s real future is in machine learning, the technology that enables computers to get smarter and more personal.
>– Eric Schmidt 

## Welcome to the world of **Data Science!**

### Broadly, there are 3 types of Machine learnign algorithms

   1. **Supervised  Learning:** This algorithm consit of a target/outcome variable (dependent variable) which is to be predicted from a given set of predictors(independent variable). Using these set of variables, we generate a function that map inputs to desired outputs. 
      - Regression
      - Decision Tree
      - Random Forest
      - KNN
      - Logistic Regression .etc
  2. **Unsupervised Learning:** In this algorithm, we don't have any target outcome variable to predict/ estimate.
       - Apriori algorithms
       - K-means
  3. **Reinforcement Learning:** This algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an enviroment where it trains itself continually using trial and error. This machine learn from past experience and tries to capture the best possible knowledge to make the accurate decisions.
        - Markov Decision Process

  ### List of Common Machine learning Algorithms
  1. Linear Regression
  2. Logistic Regression
  3. Decision Tree
  4. SVM( Support Vector Machine)
  5. Naive Bayes
  6. kNN
  7. K-Means
  8. Random Forest
  9. Dimensitionality Reduction Algorithms
  10. Gradient Boosting algorighms
      1.  GBM
      2.  XGBoost
      3.  LightGBM
      4.  CatBoost

## 1. Linear Regrassion

It is use to estimate real values (cost of houses, number of calls, total sales etc.) based on continuous variable(s). Here, we establish relationship between independent and dependent variable by fitting a best line. This best fit line is known as **regression line** and represented by a linear equation Y = a * X + b.
- Y: Dependent variable
- a: Slope
- X: Independent variable
- b: Intercept
  
![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Linear_Regression.png)

Linear Regression is mainly of two types: Simple Linear Regression and Multiple Linear Regression.
- Simple LR: Characterized by one independent variable.
- Multiple LR: Characterized by multiple independent variables.
- 
**Python Code**

xxxxxxxxxxxxxxxxxxxx

**R Code**
```
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays

x_train <- input_variables_values_training_datasets
y_train <- target_variables_values_training_datasets
x_test <- input_variables_values_test_datasets
x <- cbind(x_train,y_train)

# Train the model using the training sets and check score

linear <- lm(y_train ~ ., data = x)
summary(linear)

#Predict Output

predicted= predict(linear,x_test)
```

## 2. Logistic Regression

Don't get confused by its name! It is a classification not a regression algorithm. It is used to estimate discrete values (Binary values like 0/1, yes/no, true/false) based on given set of independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to logit function. Hence it is also known as **logit regression**. Since, it predicts the probability, its output values lies between 0 and 1.  

Coming to the math, the log odds of the outcome is modeled as a linear combinatin of predictor variables

```
odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
ln(odds) = ln(p/(1-p))
logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
```
p is the probability of presence of the characteristic of interest.

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Logistic_Regression.png)

**Python code**

**R Code**

```
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
logistic <- glm(y_train ~ ., data = x,family='binomial')
summary(logistic)
#Predict Output
predicted= predict(logistic,x_test)
```

## 3. Decision Tree

It is a type of supervised learning algorithm that is mostly used for classificaition problems. Superisingly, it works for both categorical and continuous dependent variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on most significant attributes/ independent variables to make as distinct groups as possible. 

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/IkBzK.png)

In this above image, you can see that population is classified into four groups based on multiple attributes to identify 'if they will play or not'. 

**Python Code**

**R Code**

```
library(rpart)
x <- cbind(x_train,y_train)

# grow tree 
fit <- rpart(y_train ~ ., data = x,method="class")
summary(fit)

#Predict Output 
predicted= predict(fit,x_test)
```
## 4. SVM (Support Vector Machine)

It is a classification method. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. 

For example, if we have two feature like Hieght and Hair lenght of an individual, we will first plot these two variables in two dimensional space where each point has two co-ordinates (these co-ordinates are known as **Support Vetors**)

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/SVM1.png)

Now, we will find some line that splits the data between the two differently classified groups of data. This will be the line such that the distance from the closest point in each of the two groups will be farthest away. 

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/SVM2.png)

The line which splits the data intotwo differently classifed groups is the black line, since the two closest points are the farthest apart from the line.This line is our classifier. 

**Python Code**

**R Code**

```
library(e1071)
x <- cbind(x_train,y_train)

# Fitting model
fit <-svm(y_train ~ ., data = x)
summary(fit)

#Predict Output 
predicted= predict(fit,x_test)
```
## 5. Naive Bayes

It is a classification technique based on Bayes' theorem with an assumption of independence between predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of an other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability that this fruit is an apple.

Naive Bayesian model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

Bayes theorem provides a way of calculating posterior probability P(c|x) from P(c), P(x) and P(x|c). Look at the equation below:

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Bayes_rule.png)

- P(c|x) is the posterior probability of class (target) given predictor (attribute).
- P(c) is the prior probability of class.
- P(x|c) is the likelihood which is the probability of predictor given class.
- P(x) is the prior probability of predictor.

**Python Code**

**R Code**
```
library(e1071)
x <- cbind(x_train,y_train)
# Fitting model
fit <-naiveBayes(y_train ~ ., data = x)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
```

## 6. kNN (k- Nearest Neighbors)

It can be use for both classification and regression problems. But mostly used in classification problem. K nearest neighbors is a simple algorithms that stores all available cases and classifies new cases by a majority vote of its k neighbors. The case being assigned to the class is most common amongst its K nearast neighbors measured by a distance function.

This funcion can ve Euclidean, Manhattan, Minkowski and Hamming distance.

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/KNN.png)

**Python Code**

**R Code**

```
library(knn)
x <- cbind(x_train,y_train)

# Fitting model
fit <-knn(y_train ~ ., data = x,k=5)
summary(fit)

#Predict Output 
predicted= predict(fit,x_test)
```

## 7. K-means

It is a type of unsupervised algorithm which solves the clustering problem. Its procedure follows a simple and easy way to classify a given data set through a certain number of clusters (assume k clusters). Data points inside a cluster are homogeneous and hetrogeneous to peer groups.

**How K-means forms cluster**

1. K-means picks k number of points for each cluster known as centroids.
2. Each data point forms a cluster with the closest centroids i.e. k clusters.  
3. Find the centroid of each cluster based on existing custer members. Here we have new centroids.
4. As we have new centroids, repeat step 2 and 3. Find the closest distance for each data point from new centroids and get associated with new k-clusters. Repeat this process until covergence occurs i.e. centroids does not change.

**How to determine value of K**

In K-means, we have clusters and each cluster has its own centroid. Sum of square of difference between centroid and the data points within a cluster constitutes within sum of square value for that cluster. Also, when the sum of square values for all the clusters are added, it becomes total within sum of square value for the cluster solution.

We know that as the number of cluster increases, this value keeps on decreasing but if you plot the result you may see that the sum of squared distance decreases sharply up to some value of k, and then much more slowly after that. Here, we can find the optimum number of cluster.

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Kmenas.png)

**Python Code**

**R Code**

```
library(cluster)
fit <- kmeans(X, 3) # 5 cluster solution
```

## 8. Random Forest

Random Forest is a treadmark term for an ensenble of decision trees. In Random Forest, we've collection of decision trees (so knon as "Forest"). To classify a new object based on attributes, each tree gives a classification and we say the tree "votes" for the class. The forest chooses the classification having the most votes (over all the trees in the forest).

**Python Code**

**R Code**

```
library(randomForest)
x <- cbind(x_train,y_train)

# Fitting model
fit <- randomForest(Species ~ ., x,ntree=500)
summary(fit)

#Predict Output 
predicted= predict(fit,x_test)
```
## 9. Dimensionality Reduction Algorithms

In the last 4-5 years, there has been an exponential increase in data capturing at every possible stages. Corporates/ Government Agencies/ Research organisations are not only coming with new sources but also they are capturing data in great detail.

For example: E-commerce companies are capturing more details about customer like their demographics, web crawling history, what they like or dislike, purchase history, feedback and many others to give them personalized attention more than your nearest grocery shopkeeper.

As a data scientist, the data we are offered also consist of many features, this sounds good for building good robust model but there is a challenge. How’d you identify highly significant variable(s) out 1000 or 2000? In such cases, dimensionality reduction algorithm helps us along with various other algorithms like Decision Tree, Random Forest, PCA, Factor Analysis, Identify based on correlation matrix, missing value ratio and others.

**Python Code**

**R Code**

```
library(stats)
pca <- princomp(train, cor = TRUE)
train_reduced  <- predict(pca,train)
test_reduced  <- predict(pca,test)
```

## 10. Gradient Boosting Algorithms

### 10.1. GBM

GBM (Gradient Boosting Machine) is a boosting algorithm used when we deal with plenty of data to make a prediction with high prediction power. Boosting is actually an ensemble of learning algorithms which combines the prediction of several base estimators in order to improve robustness over a single estimator. It combines multiple weak or average predictors to a build strong predictor.

![image](https://www.analyticsvidhya.com/wp-content/uploads/2015/11/bigd.png)

**Python Code**

**R Code**
```
library(caret)
x <- cbind(x_train,y_train)

# Fitting model
fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)
fit <- train(y ~ ., data = x, method = "gbm", trControl = fitControl,verbose = FALSE)
predicted= predict(fit,x_test,type= "prob")[,2] 
```

### 10.2. XGBoost

The XGBoost has an immensely high predictive power which makes it the best choice for accuracy in events as it possesses both linear model and the tree learning algorithm, making the algorithm almost 10x faster than existing gradient booster techniques.

The support includes various objective functions, including regression, classification and ranking.

One of the most interesting things about the XGBoost is that it is also called a regularized boosting technique. This helps to reduce overfit modelling and has a massive support for a range of languages such as Scala, Java, R, Python, Julia and C++.

**Python Code**

**R Code**

```
require(caret)

x <- cbind(x_train,y_train)

# Fitting model

TrainControl <- trainControl( method = "repeatedcv", number = 10, repeats = 4)

model<- train(y ~ ., data = x, method = "xgbLinear", trControl = TrainControl,verbose = FALSE)

OR 

model<- train(y ~ ., data = x, method = "xgbTree", trControl = TrainControl,verbose = FALSE)

predicted <- predict(model, x_test)
```

### 10.3. LightGBM

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency
- Lower memory usage
- Better accuracy
- Parallel and GPU learning supported
- Capable of handling large-scale data

The framework is a fast and high-performance gradient boosting one based on decision tree algorithms, used for ranking, classification and many other machine learning tasks. It was developed under the Distributed Machine Learning Toolkit Project of Microsoft.

Since the LightGBM is based on decision tree algorithms, it splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’.

**Python Code**
```
data = np.random.rand(500, 10) # 500 entities, each contains 10 features
label = np.random.randint(2, size=500) # binary target

train_data = lgb.Dataset(data, label=label)
test_data = train_data.create_valid('test.svm')

param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
param['metric'] = 'auc'

num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

bst.save_model('model.txt')

# 7 entities, each contains 10 features
data = np.random.rand(7, 10)
ypred = bst.predict(data)
```

**R Code**

```
library(RLightGBM)
data(example.binary)
#Parameters

num_iterations <- 100
config <- list(objective = "binary",  metric="binary_logloss,auc", learning_rate = 0.1, num_leaves = 63, tree_learner = "serial", feature_fraction = 0.8, bagging_freq = 5, bagging_fraction = 0.8, min_data_in_leaf = 50, min_sum_hessian_in_leaf = 5.0)

#Create data handle and booster
handle.data <- lgbm.data.create(x)

lgbm.data.setField(handle.data, "label", y)

handle.booster <- lgbm.booster.create(handle.data, lapply(config, as.character))

#Train for num_iterations iterations and eval every 5 steps

lgbm.booster.train(handle.booster, num_iterations, 5)

#Predict
pred <- lgbm.booster.predict(handle.booster, x.test)

#Test accuracy
sum(y.test == (y.pred > 0.5)) / length(y.test)

#Save model (can be loaded again via lgbm.booster.load(filename))
lgbm.booster.save(handle.booster, filename = "/tmp/model.txt")
```

### 10.4. Catboost

CatBoost is a recently open-sourced machine learning algorithm from Yandex. It can easily integrate with deep learning frameworks like Google’s TensorFlow and Apple’s Core ML.

The best part about CatBoost is that it does not require extensive data training like other ML models, and can work on a variety of data formats; not undermining how robust it can be.

Make sure you handle missing data well before you proceed with the implementation.

Catboost can automatically deal with categorical variables without showing the type conversion error, which helps you to focus on tuning your model better rather than sorting out trivial errors.

**Python Code**
```
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor

#Read training and testing files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Imputing missing values for both train and test
train.fillna(-999, inplace=True)
test.fillna(-999,inplace=True)

#Creating a training set for modeling and validation set to check model performance
X = train.drop(['Item_Outlet_Sales'], axis=1)
y = train.Item_Outlet_Sales

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)
categorical_features_indices = np.where(X.dtypes != np.float)[0]

#importing library and building model
from catboost import CatBoostRegressormodel=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)

submission = pd.DataFrame()

submission['Item_Identifier'] = test['Item_Identifier']
submission['Outlet_Identifier'] = test['Outlet_Identifier']
submission['Item_Outlet_Sales'] = model.predict(test)
```

**R Code**
```
set.seed(1)

require(titanic)

require(caret)

require(catboost)

tt <- titanic::titanic_train[complete.cases(titanic::titanic_train),]

data <- as.data.frame(as.matrix(tt), stringsAsFactors = TRUE)

drop_columns = c("PassengerId", "Survived", "Name", "Ticket", "Cabin")

x <- data[,!(names(data) %in% drop_columns)]y <- data[,c("Survived")]

fit_control <- trainControl(method = "cv", number = 4,classProbs = TRUE)

grid <- expand.grid(depth = c(4, 6, 8),learning_rate = 0.1,iterations = 100, l2_leaf_reg = 1e-3,            rsm = 0.95, border_count = 64)

report <- train(x, as.factor(make.names(y)),method = catboost.caret,verbose = TRUE, preProc = NULL,tuneGrid = grid, trControl = fit_control)

print(report)

importance <- varImp(report, scale = FALSE)

print(importance)
```