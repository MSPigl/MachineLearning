from sklearn.datasets import make_moons, make_circles, load_digits, load_breast_cancer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

# Use this for moons dataset
X, y = make_moons(n_samples=100, noise=0.25, random_state=3) 

# Use this for circles dataset   
#X, y = make_circles(n_samples=200, noise=0.12, random_state=3)

# Use these three lines for digits dataset
#digits = load_digits()
#X = digits.data
#y = digits.target
        
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
log = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(10,10),solver="sgd",learning_rate_init=1,
                    learning_rate='adaptive',max_iter=1000,random_state=120000).fit(X_train, y_train)