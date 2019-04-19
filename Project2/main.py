# utility
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# models
from sklearn.cluster import KMeans, DBSCAN

# metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score

# load data from current directory
docs = np.genfromtxt(".\\course_descriptions.txt", delimiter='\t', skip_header=1, dtype="str", usecols=(1))
s_labels = np.genfromtxt(".\\school_labels.txt", delimiter='\t', skip_header=1, dtype="int", usecols=(1))
f_labels = np.genfromtxt(".\\field_labels.txt", delimiter='\t', skip_header=1, dtype="int", usecols=(1))

vec = CountVectorizer(stop_words=["the", "this", "course", "examines"], ngram_range=(2,2)).fit(docs)
feature_array = vec.transform(docs)

km = KMeans(n_clusters=3, random_state=42).fit(feature_array)