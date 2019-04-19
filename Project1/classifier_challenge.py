# -*- coding: utf-8 -*-
"""
Machines Learning Project 1: Classifier Challenge

@author: Matthew Pigliavento
"""

import numpy as np

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# load companies data from file
companies = np.genfromtxt('training.csv', delimiter=',');

# slice out the first row of labels (evaluated to NaN if not sliced)
companies = companies[1:]

# slice companies into training data and target
training = companies[:,0:64]
target = companies[:,64]

# this might take a while to fit and cross-validate
top_model = GradientBoostingClassifier(loss = 'exponential', max_features = None)

"""
Function to print the scores for each model
"""
def validate_models():
    # fold for cross-validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=0)

    # print score from each helper function
    validateKNN(kfold)
    validateDTC(kfold)
    validateGBC(kfold)
    validateLDA(kfold)

"""
Function to print the scores for the KNeighborsClassifier models
"""
def validateKNN(kfold):
    # distance metrics for KNeighborsClassifier
    metrics = ['manhattan', 'euclidean', 'chebyshev']

    i = 1
    knn = KNeighborsClassifier().fit(training, target);
    print("KNeighborsClassifier:")
    while(i <= 7):
        knn.n_neighbors = i

        for metric in metrics:
            knn.metric = metric
            print("Neighbors: {} | Distance Metric: {} | Cross-Val Score: {}".format(knn.n_neighbors, knn.metric,np.mean(cross_val_score(knn, training, target, cv=kfold))))

        i += 2

    print("="*80)

"""
Function to print the scores for the DecisionTreeClassifier models
"""
def validateDTC(kfold):
    splitters = ['best', 'random']
    max_features = [None, 'sqrt', 'log2']

    print("DecisionTreeClassifier:")

    dtc = DecisionTreeClassifier().fit(training, target)

    for splitter in splitters:
        dtc.splitter = splitter

        for max_feature in max_features:
            dtc.max_features = max_feature
            print("Splitter: {} | Max Features: {} | Cross-Val Score: {}".format(dtc.splitter, dtc.max_features, np.mean(cross_val_score(dtc, training, target, cv=kfold))))

    print("="*80)

"""
Function to print the scores for the GradientBoostingClassifier models
"""
def validateGBC(kfold):
    losses = ['deviance', 'exponential']
    max_features = [None, 'sqrt', 'log2']

    print("GradientBoostingClassifier:")

    gbc = GradientBoostingClassifier().fit(training, target)

    for loss in losses:
        gbc.loss = loss

        for max_feature in max_features:
            gbc.max_features = max_feature
            print("Loss: {} | Max Features: {} | Cross-Val Score: {}".format(gbc.loss, gbc.max_features, np.mean(cross_val_score(gbc, training, target, cv=kfold))))


"""
Function to print the scores for the LinearDiscriminantAnalysis models

NOTE: this will generate a warning, however I looked it up and it seems that the warning
      can be ignored since we only care about the classification here
"""
def validateLDA(kfold):
    solvers = ['lsqr', 'eigen']
    shrinkages = [.25, .50, .75]

    print("LinearDiscriminantAnalysis:")

    lda = LinearDiscriminantAnalysis().fit(training, target)

    for solver in solvers:
        lda.solver = solver

        for shrinkage in shrinkages:
            lda.shrinkage = shrinkage
            print("Solver: {} | Shrinkage: {} | Cross-Val Score: {}".format(lda.solver, lda.shrinkage, np.mean(cross_val_score(lda, training, target, cv=kfold))))