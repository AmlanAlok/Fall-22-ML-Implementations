import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


def read_data_ass1(filename):
    X = []
    y = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            clean_line = line.replace('((','').replace('(','').replace('\n','').replace(')','').strip().split(', ')
            X.append([float(x) for x in clean_line[:-1]])
            y.append(clean_line[-1])
        return np.array(X), np.array(y)


filename = './datasets/Q1_train.txt'
X, y = read_data_ass1(filename)
model = DecisionTreeClassifier(criterion="entropy", max_depth=8)
model.fit(X, y)
plt.figure(figsize=(10,8))
tree.plot_tree(model, fontsize=9)