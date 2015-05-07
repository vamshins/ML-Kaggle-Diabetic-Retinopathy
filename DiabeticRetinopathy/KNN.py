__author__ = 'Vamshi'

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
# KNeighborsClassifier(...)
print(neigh.predict([[15]]))
# [0]
print(neigh.predict_proba([[0.9]]))
# [[ 0.66666667  0.33333333]]