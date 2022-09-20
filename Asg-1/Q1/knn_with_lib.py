import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):
    # print('inside func '+ inspect.stack()[0][3])

    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()

    return clean_input


def read_data(filename, col_headers):

    dataset_path = '../dataset/'
    file_path = dataset_path + filename
    input_data = fetch_data(file_path)
    # csv_filename = filename.replace('.txt', '.csv')
    # np.savetxt(csv_filename, np.array(input_data), delimiter=',', fmt='%s', comments='', header=names)
    # print(input_data)
    df = pd.DataFrame(input_data, columns = col_headers)
    return df

def input_test_data():
    names = 'height,weight,age'
    file_path = '../dataset/1a-test.txt'
    test_data = fetch_data(file_path)
    np.savetxt('1a-test.csv', np.array(test_data), delimiter=',', fmt='%s', comments='', header=names)
    # print(test_data)


# def knn_implementation(training_data, test_data, k):
#
#     dataset = pd.read_csv(training_data)
#     X_train = dataset.iloc[:, :-1].values
#     y_train = dataset.iloc[:, -1].values
#
#     # le = LabelEncoder()
#     # label = le.fit_transform(y_train)
#
#     '''Normalize features so all of them can be uniformly evaluated'''
#     # scaler = StandardScaler()
#     # scaler.fit(X_train)
#     # X_train = scaler.transform(X_train)
#     # print(X_train)
#
#     '''Normalize features so all of them can be uniformly evaluated'''
#     min_max_scaler = MinMaxScaler()
#     X_train = min_max_scaler.fit_transform(X_train)
#
#     '''KNN Classifier'''
#     classifier = KNeighborsClassifier(n_neighbors=k)
#     classifier.fit(X_train, y_train)
#
#     '''Reading test data'''
#     test_dataset = pd.read_csv(test_data)
#     X_test = test_dataset.iloc[:, :].values
#     X_test =  min_max_scaler.fit_transform(X_test)
#     # print(X_test)
#
#     y_pred = classifier.predict(X_test)
#     pass
#     return y_pred

def knn_implementation(dataset, test_dataset, k):

    # dataset = pd.read_csv(training_data)
    X_train = dataset.iloc[:, :-1].values
    y_train = dataset.iloc[:, -1].values

    # le = LabelEncoder()
    # label = le.fit_transform(y_train)

    '''Normalize features so all of them can be uniformly evaluated'''
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # print(X_train)

    '''Normalize features so all of them can be uniformly evaluated'''
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)

    '''KNN Classifier'''
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)

    '''Reading test data'''
    # test_dataset = pd.read_csv(test_data)
    X_test = test_dataset.iloc[:, :].values
    X_test =  min_max_scaler.fit_transform(X_test)
    # print(X_test)

    y_pred = classifier.predict(X_test)
    pass
    return y_pred


def lib_output(k):
    train_filename = '1a-training.txt'
    test_filename = '1a-test.txt'
    # training_data_header = 'height,weight,age,label'
    # test_data_header = 'height,weight,age'
    training_col_header = ['height', 'weight', 'age', 'col']
    test_col_header = ['height', 'weight', 'age']
    # train_csv = read_data(train_filename, training_data_header)
    # test_csv = read_data(test_filename, test_data_header)
    train_df = read_data(train_filename, training_col_header)
    test_df = read_data(test_filename, test_col_header)

    # predictions = knn_implementation(train_csv, test_csv, k)
    predictions = knn_implementation(train_df, test_df, k)
    return predictions