# Standard imports
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, LeaveOneOut

# Dataset imports
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
loo = LeaveOneOut()

def cancer_plot():
    fig, axes = plt.subplots(15, 2, figsize=(10, 20))
    malignant = cancer.data[cancer.target == 0]
    benign = cancer.data[cancer.target == 1]
    ax = axes.ravel()
    for i in range(30):
        _, bins = np.histogram(cancer.data[:, i], bins=50)
        ax[i].hist(malignant[:, i], bins=bins, color='b', alpha=.5)
        ax[i].hist(benign[:, i], bins=bins, color='y', alpha=.5)
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["malignant", "benign"], loc="best")
    fig.tight_layout()
    
def model_test(d,t):
    knn1 = KNeighborsClassifier(n_neighbors=1,metric="euclidean")
    knn5 = KNeighborsClassifier(n_neighbors=5,metric="euclidean")
    knn1_score = cross_val_score(knn1, d, t, cv=loo)
    knn5_score = cross_val_score(knn5, d, t, cv=loo)
    print("Nearest neighbors accuracy, k=1: {}".format(np.mean(knn1_score)))
    print("Nearest neighbors accuracy, k=5: {}".format(np.mean(knn5_score)))

def coef_plot(reg_model_list, mydata):
    mk = ["*","^","v","o","s","P","X","D","H"]
    for mod in reg_model_list:
        lb = "C=" + str(mod.C)
        plt.plot(mod.coef_.T, mk.pop(0), label=lb)
    plt.xticks(range(mydata.data.shape[1]), mydata.feature_names, rotation=90)
    plt.hlines(0, 0, mydata.data.shape[1])
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.legend()

def L2_dist(point1, point2):
    if (len(point1) != len(point2)):
        return -1
    sum = 0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        sum += diff * diff
    return np.sqrt(sum)

def cluster_features(data, centers):
    nrows = len(data)
    ncols = len(centers)
    result = np.zeros((nrows,ncols)) # creates an array of all zeros
    
    row =  0
    col = 0
    for point in data:
        for center in centers:
            result[row][col] = L2_dist(point, center)
            col += 1
        
        row += 1
        col = 0
    
    return result