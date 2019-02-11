# Standard imports
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Dataset imports
from sklearn.datasets import load_breast_cancer, load_iris

# Tool imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

cdata = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cdata.data,cdata.target,stratify=cdata.target,random_state=42)

lr100 = LogisticRegression(C=100,solver="liblinear",penalty="l2").fit(X_train,y_train)

def coef_plot(reg_model_list, mydata):
    mk = ["*","^","v","o","s","P","X","D","H"]
    for mod in reg_model_list:
        lb = "C=" + str(mod.C)
        plt.plot(mod.coef_.T, mk.pop(0), label=lb)
    plt.xticks(range(mydata.data.shape[1]), mydata.feature_names, rotation=90)
    plt.hlines(0, 0, mydata.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
