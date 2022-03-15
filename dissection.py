'''
    dataset types :
    1: moon
    2: fico
    3: german
    4: mimc
    5: Recidivism
    6: scenario1
'''

from cProfile import label
import pandas as pd
import numpy as np
import sys
import math
from matplotlib import pyplot as plt

import random
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.optimize import minimize


# CONSTANTS
DATASETS = {
     1: "moon",     2: "fico",     3: "german",     4: "mimc",     5: "Recidivism",     6: "scenario1"
}

RATIONALITY = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


# Take in alpha beta and gamma as inputs
alpha = float(sys.argv[1])
beta = float(sys.argv[2])
gamma = float(sys.argv[3])
dataset = int(sys.argv[4])

# Calculate human confidence
confidence =  alpha - gamma / (1 + beta)

rng = np.random.default_rng()

def get_train_test():
    if dataset == 1:
        # Create make_moons dataset and split training and test data setts
        X, y = make_moons(n_samples=10000, noise = 0.1)
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    target = df.label
    df.drop(['label'], axis=1, inplace=True)
    y = target
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test()
# Create linear classification model and fit to training data
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
Linear_weights = model.coef_[0]
probs = model.predict_proba(X_test)
print(model.score(X_test, y_test))

# Function to calculate the team expected utility
def expected_team_utility(x_probs, y, human_rationality=1):
    rational_rng = rng.uniform(0,1)
    if (np.max(x_probs) >= confidence and rational_rng <= human_rationality):
        return (1 + beta)*x_probs[y] - beta
    else: 
        return (1 + beta)*alpha - beta - gamma

#function to calculate the team empirical utility
def empirical_team_utility(x_probs, y, human_rationality=1):
    rational_rng = rng.uniform(0,1)
    if (np.max(x_probs) >= confidence and rational_rng <= human_rationality):
        if (np.argmax(x_probs) == y):
            return 1
        else:
            return -beta
    else:
        human = rng.uniform(0,1)
        if human < alpha:
            return 1 - gamma
        else: 
            return -beta - gamma

# Calulate the average expected and empirical utility for the linear model
if __name__ == "__main__":

    res_expected = []
    res_emperical = []
    for human_rationality in RATIONALITY:
        expected = 0
        empirical = 0
        count = 0
    
        for i in y_test.index:
            expected += expected_team_utility(probs[count], y_test[i], human_rationality=human_rationality)
            empirical += empirical_team_utility(probs[count], y_test[i], human_rationality=human_rationality)
            count += 1
        res_expected.append(expected/ (X_test.size/2))
        res_emperical.append(empirical/(X_test.size/2))
    plt.xlabel("Human Rationality Probability")
    plt.ylabel("Utility")
    plt.plot(RATIONALITY, res_expected, label="expected")
    plt.plot(RATIONALITY, res_emperical, label="emperical")
    leg = plt.legend(loc='upper center')
    plt.savefig("results/linear_moons" + "_" + str(alpha) + "_" + str(beta) + "_" + str(gamma) + ".png")
    plt.close()

    #print()

# Scale data around zero for Mulit-layered perceptron classifier
sc_X = StandardScaler()
X_trainscaled = sc_X.fit_transform(X_train)
X_valscaled = sc_X.transform(X_test)
