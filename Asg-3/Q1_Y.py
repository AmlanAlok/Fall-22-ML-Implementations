import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4)


class Node:
    def __init__(self, X, y, depth):
        self.depth = depth
        self.X = X
        self.y = y
        self.left = None
        self.right = None
        self.cutoff = None
        self.feature_id = None
        self.label = None
        self.classnumber = {}
        self.entropy = self.get_node_entropy()

    def get_node_entropy(self):
        probs = []
        p = np.unique(self.y)
        for c in np.unique(self.y):
            q = (self.y == c)
            probs.append(q.sum() / len(self.y))
            self.classnumber[c] = (self.y == c).sum()
        self.label = np.unique(self.y)[np.argmax(np.array(probs))]
        return get_entropy(probs)


class DecisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def train(self, X, y):
        self.X = X
        self.y = y
        self.root = self.add_node(Node(self.X, self.y, 0))

    def add_node(self, node):
        if node.depth >= self.max_depth:
            return node
        elif node and node.entropy == 0:
            return node
        else:
            curr_gain = 0
            for feature_id in range(3):
                feature = node.X[:, feature_id]     # selecting feature col
                sorted_index = node.X[:, feature_id].argsort()  # selecting feature col and sorting it
                feature = node.X[:, feature_id][sorted_index]

                for i in range(feature.shape[0] - 1):
                    if feature[i] != feature[i + 1]:
                        avg = (feature[i] + feature[i + 1]) / 2
                        left_probs = []
                        right_probs = []
                        for c in np.unique(node.y):
                            left_probs.append((node.y[sorted_index[:i + 1]] == c).sum() / (i + 1))
                            right_probs.append((node.y[sorted_index[i + 1:]] == c).sum() / (len(node.y) - i - 1))

                        gain = node.entropy - (((i + 1) / len(node.y)) * get_entropy(left_probs)) - (
                                    ((len(node.y) - i - 1) / len(node.y)) * get_entropy(right_probs))

                        if gain > curr_gain:
                            cutoff_index = i
                            node.feature_id = feature_id
                            left_indexes = sorted_index[:i + 1]
                            right_indexes = sorted_index[i + 1:]
                            curr_gain = gain
                            node.cutoff = avg

            if curr_gain == 0:
                return node
            else:
                node.left = self.add_node(Node(node.X[left_indexes], node.y[left_indexes], node.depth + 1))
                node.right = self.add_node(Node(node.X[right_indexes], node.y[right_indexes], node.depth + 1))
                return node

    def predict(self, queries):
        preds = []
        for query in queries:
            preds.append(self.inference(query))
        return np.array(preds)

    def inference(self, query):
        curr_node = self.root
        while curr_node.left != None and curr_node.right != None:
            if query[curr_node.feature_id] <= curr_node.cutoff:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        return curr_node.label


def read_data_ass1(filename):
    X = []
    y = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            clean_line = line.replace('((','').replace('(','').replace('\n','').replace(')','').strip().split(', ')
            X.append([float(x) for x in clean_line[:-1]])
            y.append(clean_line[-1])
        return np.array(X),np.array(y)


def get_entropy(probs):
    entropy = -1*np.sum(probs*np.log2(probs))
    return 0 if np.isnan(entropy) else entropy


def accuray(true,pred):
    return (true == pred).sum()/len(true)


def main():
    X, y = read_data_ass1('./datasets/Q1.txt')

    train_accuracy = []
    test_accuracy = []
    for depth in range(1, 9):
        decsiontree_clf = DecisionTreeClassifier(depth)
        decsiontree_clf.train(X[:90], y[:90])
        train_preds = decsiontree_clf.predict(X[0:90])
        test_preds = decsiontree_clf.predict(X[90:])
        train_accuracy.append(accuray(y[0:90], train_preds))
        test_accuracy.append(accuray(y[90:], test_preds))
    print(test_accuracy)
    plt.plot(list(range(1, 9)), train_accuracy, label='training accuracy')
    plt.plot(list(range(1, 9)), test_accuracy, label='test accuracy')
    plt.legend()


if __name__ == '__main__':
    main()