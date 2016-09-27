# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)

# print the iris data
print(iris.data)

# print the names of the four features
print(iris.feature_names)

# print integers representing the species of each observation
print(iris.target)

# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)

# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)

# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

print type(iris.data)
print type(iris.target)

# training several models
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
print knn
knn.fit(X, y)
test_data = [[3, 5, 4, 2], [5, 4, 3, 2]]
print knn.predict(test_data)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X, y)
print logreg.predict(test_data)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
print tree
tree.fit(X, y)
print tree.predict(test_data)

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
ada.fit(X,y)
print ada.predict(test_data)