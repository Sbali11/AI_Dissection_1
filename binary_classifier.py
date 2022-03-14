from cProfile import label
import pandas as pd
import numpy as np
import sys
import math

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.optimize import minimize

# Take in alpha beta and gamma as inputs
alpha = float(sys.argv[1])
beta = float(sys.argv[2])
gamma = float(sys.argv[3])

# Calculate human confidence
confidence =  alpha - gamma / (1 + beta)

rng = np.random.default_rng()

# Create make_moons dataset and split training and test data setts
X, y = make_moons(n_samples=10000, noise = 0.1)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
target = df.label
df.drop(['label'], axis=1, inplace=True)
y = target
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)

# Create linear classification model and fit to training data
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
Linear_weights = model.coef_[0]
probs = model.predict_proba(X_test)
print(model.score(X_test, y_test))

# Function to calculate the team expected utility
def expected_team_utility(x_probs, y):
    if (np.max(x_probs) >= confidence):
        return (1 + beta)*x_probs[y] - beta
    else: 
        return (1 + beta)*alpha - beta - gamma

#function to calculate the team empirical utility
def empirical_team_utility(x_probs, y):
    if (np.max(x_probs) >= confidence):
        if (np.argmax(x_probs) == y):
            return 1
        else:
            return -beta
    else:
        human = rng.uniform(0,1)
        if human < alpha:
            return 1- gamma
        else: 
            return -beta - gamma

# Calulate the average expected and empirical utility for the linear model
expected = 0
empirical = 0
count = 0
for i in y_test.index:
    expected += expected_team_utility(probs[count], y_test[i])
    empirical += empirical_team_utility(probs[count], y_test[i])
    count += 1
print(expected/ (X_test.size/2))
print(empirical/ (X_test.size/2))

# Scale data around zero for Mulit-layered perceptron classifier
sc_X = StandardScaler()
X_trainscaled = sc_X.fit_transform(X_train)
X_valscaled = sc_X.transform(X_test)

"""# Create Multi-layered perceptron classifier
MLP = MLPClassifier(hidden_layer_sizes=(50,10), activation="relu", random_state=1).fit(X_trainscaled, y_train)
probs = MLP.predict_proba(X_valscaled)
MLP_weights = MLP.coefs_
print(MLP.score(X_valscaled, y_test))

# Calulate the average expected and empirical utility for the Mulit-layered perceptron classifier
expected = 0
empirical = 0
count = 0
for i in y_test.index:
    expected += expected_team_utility(probs[count], y_test[i])
    empirical += empirical_team_utility(probs[count], y_test[i])
    count += 1

print(expected/ (X_test.size/2))
print(empirical/ (X_test.size/2))"""

# SGD function
# input model parameters, learning rate, features and label for ith traing example
def SGD(theta, lr, features_i, label_i):
    pred = sigmoid(theta @ features_i)
    prob = np.array([1-pred, pred])
    exp = expected_team_utility(prob, label_i)
    exp = math.exp(exp)
    return theta + lr * features_i/(X_train.size/2) *(label_i-exp/(1+exp)) 

# Sigmoid function
# input a number and return the sigmoid of that number
def sigmoid(number):
    exp = math.exp(-number)
    return 1.0/(1.0+exp)

# Set theta to initial linear weights and train model to optimize for expected team utility
theta = Linear_weights
for i in range(90):
    for index in X_train.index:
        theta = SGD(theta, .1, X_train.loc[[index]].to_numpy()[0], y_train[index])

# Calculate expected and empirical team utility for new updated model
expected = 0
empirical = 0
for i in y_test.index:
    pred = sigmoid(theta @ X_test.loc[[i]].to_numpy()[0])
    prob = np.array([1-pred,pred])
    expected += expected_team_utility(prob, y_test[i])
    empirical += empirical_team_utility(prob, y_test[i])

print(expected/ (X_test.size/2))
print(empirical/ (X_test.size/2))