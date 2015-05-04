__author__ = 'Vamshi'

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import os

X = [[0, 1], [1, 1], [100, 2000]]
Y = [0, 1, 200]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print clf.predict([[100, 2000]])
print clf.predict([[1., 2.]])
print clf.predict([[100., 200.]])
print clf.predict_proba([[2, 2]])
print clf.predict_proba([[1000, 2]])

iris = load_iris()
clf = tree.DecisionTreeClassifier()
print clf
clf = clf.fit(iris.data, iris.target)
print clf
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
    print f
os.unlink('iris.dot')