import numpy as np
import matplotlib.pyplot as plt
import nltk

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import silhouette_score, adjusted_rand_score

ng_train = fetch_20newsgroups(subset="train",data_home=".\\")
y_train = ng_train.target
ng_test = fetch_20newsgroups(subset="test",data_home=".\\")
y_test = ng_test.target

test_string = "Our meeting today was worse than yesterday, I'm really scared of meeting the clients tomorrow."

# IMPORTANT - update this to the nltk download directory
my_path = 'C:\\Users\\Matthew\\Desktop\\MachineLearning\\CSIS320\\Lab8_nltk_data'
if my_path not in nltk.data.path:
    nltk.data.path.insert(0,my_path)

lem = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

def pos_help(p):
    if p[0] == 'V':
        return nltk.corpus.wordnet.VERB
    if p[0] == 'J':
        return nltk.corpus.wordnet.ADJ
    if p[0] == 'R':
        return nltk.corpus.wordnet.ADV
    return nltk.corpus.wordnet.NOUN

def compare_tokens(doc):
    tokens = nltk.tokenize.word_tokenize(doc.lower())
    ptokens = [(t,pos_help(p)) for (t,p) in nltk.pos_tag(tokens)]
    print("Lemmatization:")
    print([lem.lemmatize(token,pos=part) for (token,part) in ptokens])
    print("Stemming:")
    print([stemmer.stem(token) for token in tokens])

def visualize_coefficients(coefficients, feature_names, n_top_features=25):
    """Visualize coefficients of a linear model.
    Parameters
    ----------
    coefficients : nd-array, shape (n_features,)
        Model coefficients.
    feature_names : list or nd-array of strings, shape (n_features,)
        Feature names for labeling the coefficients.
    n_top_features : int, default=25
        How many features to show. The function will show the largest (most
        positive) and smallest (most negative)  n_top_features coefficients,
        for a total of 2 * n_top_features coefficients.
    """
    coefficients = coefficients.squeeze()
    if coefficients.ndim > 1:
        # this is not a row or column vector
        raise ValueError("coeffients must be 1d array or column vector, got"
                         " shape {}".format(coefficients.shape))
    coefficients = coefficients.ravel()

    if len(coefficients) != len(feature_names):
        raise ValueError("Number of coefficients {} doesn't match number of"
                         "feature names {}.".format(len(coefficients),
                                                    len(feature_names)))
    # get coefficients with large absolute values
    coef = coefficients.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients,
                                          positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["yellow" if c < 0 else "blue"
              for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients],
            color=colors)
    feature_names = np.array(feature_names)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features),
               feature_names[interesting_coefficients], rotation=60,
               ha="right")
    plt.ylabel("Coefficient magnitude")
    plt.xlabel("Feature")
    
def print_topics(topics, feature_names, sorting, topics_per_chunk=5,
                 n_words=20):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):
            try:
                print(("{:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))
            except:
                pass
        print("\n")

def topic_plot(topics, feature_names, sorting):
    sums = np.sum(topics,axis=0)
    topic_names = ["{:>2} ".format(i) + " ".join(words) for i, words in enumerate(feature_names[sorting[:, :2]])]
    plt.barh(y=range(len(sums)),width=sums,tick_label=topic_names)
