import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz


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


def main():
    filename = './datasets/Q1_train.txt'
    X, y = read_data_ass1(filename)
    model = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    model.fit(X, y)
    # print(model.score(X, y))
    # tree.plot_tree(model)
    A = tree.export_graphviz(model)
    graph = graphviz.Source(A)
    # from IPython.display import display
    # display(graph)
    graph.format = 'png'
    graph.render()
    # print(graph)
    print('END')


if __name__ == '__main__':
    main()