import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score as jss

rnd = np.random.RandomState(seed=42)

docs = np.genfromtxt(".\\course_descriptions.txt", delimiter='\t', skip_header=1, dtype="str", usecols=(1))

s_labels = np.genfromtxt(".\\school_labels.txt", delimiter='\t', skip_header=1, dtype="int", usecols=(1))
f_labels = np.genfromtxt(".\\field_labels.txt", delimiter='\t', skip_header=1, dtype="int", usecols=(1))

vec = CountVectorizer(max_df=0.15,binary=True).fit(docs)
feature_array = vec.transform(docs)
N = feature_array.shape[0]   # 880
F = feature_array.shape[1]   # 6133
num_pairs = N * (N-1) // 2   # 386760

###################################
# This code gives you 100 random permutations of the values from 0 to 6132
permutations = []
for i in range(100):
    permutations.append(rnd.permutation(F))
###################################

#################################################
# This method should take a vector (1-D array) and a permutation and return
# the index of the first non-zero value when scrambled according to the
# perm parameter. The default is no scrambling.
#
# Example: If vector is [0 3 0 -2 0] and perm is [2 4 3 0 1], then the
# permuted vector would be [0 0 -2 0 3] and the return value should
# be 2 (the index of the -2 in the permuted array)
def find_first_nonzero(vector, perm=range(F)):
    result = F

    # flatnonzero returns the indices of all non-zero elements
    for index in np.nonzero(vector)[1]:
        currentIndex, = np.where(perm == index)

        if currentIndex < result:
            result = currentIndex
        
    return result[0]
################################################

################################################
# This method generates a list of {numbands} LSH signatures. Each signature
# is a combination of {numrows} individual minhashes. It is vital that,
# across all of the signatures for a vector, each minhash uses a different
# permutation, and also that each call to this method use the same set of
# permutations. (i.e. LSH doesn't work if you use different permutations
# for each vector you are hashing)
def create_signatures(vector, numrows, numbands):
    result = []
    signature = ""
    index = 0
    
    for b in range(numbands):
        signature = ""
        
        # create signature with inner loop
        #signature += str(b) + "-"
        for r in range(numrows):
            signature += str(find_first_nonzero(vector, permutations[index])) + "-"
            index += 1
        
        index += 1
        #signature += "-" + str(b)
        result.append(signature)
        
    return result
################################################
 
################################################
# This method performs LSH on the input data_matrix and returns a hashmap
# (i.e. a Python dictionary) with the resulting bins. Each row of the data
# will be hashed {numbands} times using signatures of length {numrows}.
def hash_docs(data_matrix, numrows, numbands):
    result = {}
    for v in range(N):
        vector = data_matrix[v]
        
        signatures = create_signatures(vector, numrows, numbands)
        
        for sig in signatures:
            group = []
            if sig in result:
                group = result[sig]
            group.append(v)
            result[sig] = group
    return result
###############################################
    
###############################################
    
def count_matches(hashmap, labels, features):
    matches = 0
    mismatches = 0
    jaccard_sum = 0.0
    for sig in hashmap:
        # We only have collisions if a bin has more than one element
        if len(hashmap[sig]) > 1:
            # For each pair of things here, determine if they have the same or
            # different labels
            # Also, calculate a Jaccard similarity and add it to the sum
            for element in range(1, len(hashmap[sig])):
                label1 = hashmap[sig][element]
                label2 = hashmap[sig][element - 1]
                
                if labels[label1] == labels[label2]:
                    matches += 1
                else:
                    mismatches += 1
                    
                jaccard_sum += jss(features[label1], features[label2])
            
    javg = jaccard_sum / (matches + mismatches)
    return (matches,mismatches,javg)