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

