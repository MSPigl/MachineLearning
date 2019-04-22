# utility
import numpy as np

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction import text

# models
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering

# metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score

# load data from current directory
docs = np.genfromtxt(".\\course_descriptions.txt", delimiter='\t', skip_header=1, dtype="str", usecols=(1))
s_labels = np.genfromtxt(".\\school_labels.txt", delimiter='\t', skip_header=1, dtype="int", usecols=(1))
f_labels = np.genfromtxt(".\\field_labels.txt", delimiter='\t', skip_header=1, dtype="int", usecols=(1))

# add dataset specific terms to the english stop words frozenset
stopWords = text.ENGLISH_STOP_WORDS.union(["this", "course", "will", "examine", "explore"])

# message to user
print("Use the testClustering method for individual trials, and findBestClustering to score the silhouettes, " 
      + "though this will take a while to run based on your specified range")

def testClustering(numClusters = 3, ground_truth = s_labels, countVec = False, tfidVec = False, hashVec = False):
    """Function to test clustering vectorizer-model combinations
    Parameters:
    numClusters  -- the number of clusters to provide to the clustering models           (default 3)
    ground_truth -- ground truth labels                                                  (default s_labels)
    countVec     -- boolean to decide whether or not to execute a CountVectorizer test   (default False)
    tfidVec      -- boolean to decide whether or not to execute a TfidfVectorizer test   (default False)
    hashVec      -- boolean to decide whether or not to execute a HashingVectorizer test (default False)
    """
    
    print("With " + str(numClusters) + " clusters:")
    print("")
    
    if countVec:
        print("CountVectorizer:")
        print("")
        
        vec = CountVectorizer(stop_words=stopWords)
        feature_array = vec.fit_transform(docs)
        
        print("KMeans:")
        km = KMeans(n_clusters=numClusters, init='k-means++', n_init=1, random_state = 42).fit(feature_array)
        print(adjusted_rand_score(ground_truth, km.labels_))
        
        print("Minibatch KMeans:")
        mbk = MiniBatchKMeans(n_clusters = numClusters, random_state = 42).fit(feature_array)
        print(adjusted_rand_score(ground_truth, mbk.labels_))
        
        """ SpectralClustering omitted due to convergence issues when using this vectorizer """
        
        print("")

    if tfidVec:
        print("TfidfVectorizer:")
        print("")
        
        vec = TfidfVectorizer(stop_words=stopWords)
        feature_array = vec.fit_transform(docs)
                
        print("KMeans:")
        km = KMeans(n_clusters=numClusters, init='k-means++', n_init=1, random_state = 50).fit(feature_array)
        print(adjusted_rand_score(ground_truth, km.labels_))
        
        print("Minibatch KMeans:")
        mbk = MiniBatchKMeans(n_clusters = numClusters, random_state = 37).fit(feature_array)
        print(adjusted_rand_score(ground_truth, mbk.labels_))
        
        print("SpectralClustering:")
        spec = SpectralClustering(n_clusters=numClusters, random_state=42).fit(feature_array)
        print(adjusted_rand_score(ground_truth, spec.labels_))
        
        print("")
    
    if hashVec:
        print("HashingVectorizer:")
        print("")
        
        vec = HashingVectorizer(stop_words=stopWords)
        feature_array = vec.fit_transform(docs)
                
        print("KMeans:")
        km = KMeans(n_clusters=numClusters, init='k-means++', n_init=1, random_state = 42).fit(feature_array)
        print(adjusted_rand_score(ground_truth, km.labels_))
        
        print("Minibatch KMeans:")
        mbk = MiniBatchKMeans(n_clusters = numClusters, random_state = 50).fit(feature_array)
        print(adjusted_rand_score(ground_truth, mbk.labels_))
        
        print("SpectralClustering:")
        spec = SpectralClustering(n_clusters=numClusters, random_state=42).fit(feature_array)
        print(adjusted_rand_score(ground_truth, spec.labels_))
        
        print("")

def findBestClustering(low = 3, high = 100):
    """ Function to find the best number of clusters based on the average silhouette
    Parameters:
    low  -- the lower bound for the cluster range             (default 3)
    high -- the (inclusive) upper bound for the cluster range (default 100)
    """
    
    topScore = 0.0
    topClusters = 0
    topVec = ""
    topModel = ""
    
    for numClusters in range(low, high + 1):
        # TfidfVectorizer
        vec = TfidfVectorizer(stop_words=stopWords)
        feature_array = vec.fit_transform(docs)
        
        km = KMeans(n_clusters=numClusters, init='k-means++', n_init=1, random_state = 42).fit(feature_array)
        silhouette = silhouette_score(feature_array, km.labels_)
        
        if silhouette > topScore:
            topScore = silhouette
            topClusters = numClusters
            topModel = "KMeans"
            topVec = "TfidfVectorizer"
        
        mbk = MiniBatchKMeans(n_clusters = numClusters, random_state = 42).fit(feature_array)
        silhouette = silhouette_score(feature_array, mbk.labels_)
        
        if silhouette > topScore:
            topScore = silhouette
            topClusters = numClusters
            topModel = "MiniBatch KMeans"
            topVec = "TfidfVectorizer"
            
        spec = SpectralClustering(n_clusters=numClusters, random_state=42).fit(feature_array)
        silhouette = silhouette_score(feature_array, spec.labels_)

        if silhouette > topScore:
            topScore = silhouette
            topClusters = numClusters
            topModel = "SpectralClustering"
            topVec = "TfidfVectorizer"
            
        print("Using {} clusters: Top score = {} ({} -> {})".format(numClusters, topScore, topVec, topModel))
    
    print("")
    print("With {} clusters: Top score = {} ({} -> {})".format(topClusters, topScore, topVec, topModel))

