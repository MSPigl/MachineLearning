import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Calculates Manhattan distance
def M_dist(p1,p2):
    sum = 0
    for i in range(0,len(p1)):
        sum += abs(p1[i] - p2[i])
    return sum

# Calculates Euclidean distance
def E_dist(p1,p2):
    sum = 0
    for i in range(0, len(p1)):
        sum += (p1[i] - p2[i])**2
    return np.sqrt(sum)

# Calculates the L0 norm
def Z_dist(p1,p2):
    sum = 0
    for i in range(0, len(p1)):
        val = abs(p1[i] - p2[i])
        if val > 0:
            sum += 1
    return sum

# Calculates the infinity norm
def I_dist(p1,p2):
    vals = []
    for i in range(0, len(p1)):
        vals.append(abs(p1[i] - p2[i]))
    return max(vals)

# Nearest neighbor classifier for k=1
def test_kNN(point, data, classes, dist="Euclidean"):
    if len(data[0]) != len(point):
        print("Dimensional mismatch!")
        return
    
    # Initial conditions
    bestval = float("inf")
    bestindex = -1
    
    for i in range(0,len(data)-1):
        testpt = data[i]
        if dist == "Manhattan":
            d = M_dist(testpt, point)
        elif dist == "Zero":
            d = Z_dist(testpt, point)
        elif dist == "Infinity":
            d = I_dist(testpt, point)
        else:
            d = E_dist(testpt, point)    
        if (d < bestval):
            bestval = d
            bestindex = i
    
    return classes[bestindex]       
    
#################################################

def cross_validate_kNN(data, targets, numfolds=3, k=1):
    indices = [i for i in range(len(data))]
    random.shuffle(indices)
    
    accuracy_sum = 0
    for fold in range(numfolds):
        train_data = []
        train_targets = []
        test_data = []
        test_targets = []
        
        # Use the .append method to add the appropriate elements to each list
        # Syntax: train_data.append(data[0])
        
        for i in range(len(data)):
            if i % numfolds == fold:
                test_data.append(data[indices[i]])
                test_targets.append(targets[indices[i]])
            else:
                train_data.append(data[indices[i]])
                train_targets.append(targets[indices[i]])
        
        # These steps are needed to convert to the correct type of array for
        # the sklearn classifiers
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)
        test_data = np.array(test_data)
        test_targets = np.array(test_targets)
        
        # Define knn to be an appropriate classifier and fit the training data
        # (Use the sklearn classifier here, not ours; use Euclidean distance)
        
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='euclidean')
        knn.fit(data, targets)
        
        test_pred = knn.predict(test_data)
        accuracy = np.mean(test_pred == test_targets)
        accuracy_sum += accuracy
        
    return accuracy_sum / numfolds
