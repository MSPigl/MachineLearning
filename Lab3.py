# Standard imports
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# Dataset imports
from sklearn.datasets import fetch_olivetti_faces


n_row, n_col = 1,4
image_shape = (64,64)

people = fetch_olivetti_faces(data_home='Z:\\CSIS320\\data',shuffle=True)

n_samples, n_features = people.data.shape

def randIndex(target, labels):
    score = 0
    
    for i in range(0, len(target)):
        for j in range(i, len(target)):
            if i == j:
                continue
            else:
                if target[i] == target[j]:
                    if labels[i] == labels[j]:
                        score += 1
                elif target[i] != target[j]:
                    if labels[i] != labels[j]:
                        score += 1
                        
    return score / ((len(target) * (len(target) - 1)) / 2)


def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
