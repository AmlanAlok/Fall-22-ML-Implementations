import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):

    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def read_data(filename, col_headers):

    dataset_path = '../dataset/'
    file_path = dataset_path + filename
    input_data = fetch_data(file_path)
    df = pd.DataFrame(input_data, columns=col_headers)
    return df


def knn_implementation(train_dataset, test_dataset, k, metric):

    X_train = train_dataset.iloc[:, :-1].values
    y_train = train_dataset.iloc[:, -1].values
    X_test = test_dataset.iloc[:, :].values

    '''Normalize features so all of them can be uniformly evaluated'''
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)

    p = 0
    if metric == 'manhattan':
        p = 1
    elif metric == 'euclidean':
        p = 2
    elif metric == 'minkowski':
        p = 3

    '''KNN Classifier'''
    classifier = KNeighborsClassifier(n_neighbors=k, p=p)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    return y_pred


def library_output(k, metric):
    train_filename = '1a-training.txt'
    test_filename = '1a-test.txt'

    training_col_header = ['height', 'weight', 'age', 'label']
    test_col_header = ['height', 'weight', 'age']

    train_df = read_data(train_filename, training_col_header)
    test_df = read_data(test_filename, test_col_header)

    predictions = knn_implementation(train_df, test_df, k, metric)
    return predictions


if __name__ == '__main__':
    library_output(1, 'euclidean')