import inspect
import math
import numpy as np

CONST_PREDICTION = 'prediction'


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):

    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def change_data_structure(input_data):

    data_list = [None] * len(input_data)
    i = 0

    for record in input_data:
        data_list[i] = {
            'index': i,
            'input': {
                'height': float(record[0]),
                'weight': float(record[1]),
                'age': float(record[2])
            },
            'output': record[3]
        }

        i += 1

    print('Converted each data point to dictionary format')

    return data_list


def data_structure_test(test_data):

    test_list = [None]*len(test_data)
    i=0

    for dp in test_data:
        test_list[i] = {
            'index': i,
            'input': {
                'height': float(dp[0]),
                'weight': float(dp[1]),
                'age': float(dp[2])
            },
            # 'knn': k_list,
            'output': []
        }
        i+=1

    return test_list


def get_minkowski_distance(input_data, test_input):

    a = [None]*len(input_data)
    i = 0

    for dp in input_data:
        minkowski_distance = (
            pow(abs(dp['input']['height']-test_input['input']['height']), 3) +
            pow(abs(dp['input']['weight']-test_input['input']['weight']), 3) +
            pow(abs(dp['input']['age']-test_input['input']['age']), 3)) ** (1./3)

        a[i] = {
            'index': dp['index'],
            'minkowski_distance': minkowski_distance,
            'output': dp['output']
        }

        i += 1

    sorted_a = sorted(a, key=lambda d: d['minkowski_distance'])
    return sorted_a


def get_manhattan_distance(input_data, test_input):
    a = [None] * len(input_data)
    i = 0

    for dp in input_data:
        manhattan_distance = \
            abs(dp['input']['height'] - test_input['input']['height']) + \
            abs(dp['input']['weight'] - test_input['input']['weight']) + \
            abs(dp['input']['age'] - test_input['input']['age'])

        a[i] = {
            'index': dp['index'],
            'manhattan_distance': manhattan_distance,
            'output': dp['output']
        }

        i += 1

    sorted_a = sorted(a, key=lambda d: d['manhattan_distance'])
    return sorted_a


def get_euclidean_distance(input_data, test_input):

    a = [None]*len(input_data)
    i = 0

    for dp in input_data:
        cartesian_distance = math.sqrt(
            pow(dp['input']['height']-test_input['input']['height'], 2) +
            pow(dp['input']['weight']-test_input['input']['weight'],2) +
            pow(dp['input']['age']-test_input['input']['age'], 2))

        a[i] = {
            'index': dp['index'],
            'cartesian_distance': cartesian_distance,
            'output': dp['output']
        }

        i += 1

    sorted_a = sorted(a, key=lambda d: d['cartesian_distance'])
    return sorted_a


def normalize_train(input_data):

    X_train = input_data[:, :-1].astype('float')
    y_train = input_data[:, -1]

    h_min = np.amin(X_train[:, 0])
    h_max = np.amax(X_train[:, 0])
    w_min = np.amin(X_train[:, 1])
    w_max = np.amax(X_train[:, 1])
    a_min = np.amin(X_train[:, 2])
    a_max = np.amax(X_train[:, 2])

    for dp in X_train:
        dp[0] = (dp[0] - h_min) / (h_max-h_min)
        dp[1] = (dp[1] - w_min) / (w_max - w_min)
        dp[2] = (dp[2] - a_min) / (a_max - a_min)

    y_train = y_train.reshape(y_train.shape[0], 1)
    data = np.concatenate((X_train, y_train), axis=1)

    return data


def normalize_test(input_data):
    X_test = input_data.astype('float')

    h_min = np.amin(X_test[:, 0])
    h_max = np.amax(X_test[:, 0])
    w_min = np.amin(X_test[:, 1])
    w_max = np.amax(X_test[:, 1])
    a_min = np.amin(X_test[:, 2])
    a_max = np.amax(X_test[:, 2])

    for dp in X_test:
        dp[0] = (dp[0] - h_min) / (h_max - h_min)
        dp[1] = (dp[1] - w_min) / (w_max - w_min)
        dp[2] = (dp[2] - a_min) / (a_max - a_min)

    return X_test


def make_prediction(k, distance_array):

    w_count = 0
    m_count = 0

    knn_array = distance_array[:k]

    for dic in knn_array:
        if dic['output'] == 'W':
            w_count += 1
        if dic['output'] == 'M':
            m_count += 1

    W_prob = w_count / k
    M_prob = m_count / k

    # print('W prob =', W_prob)
    # print('M prob =', M_prob)

    if w_count > m_count:
        # k_dict[CONST_PREDICTION] = 'W'
        return 'W'
    elif m_count > w_count:
        # k_dict[CONST_PREDICTION] = 'M'
        return 'M'
    else:
        # k_dict[CONST_PREDICTION] = '50-50 M/W'
        return '50-50 M/W'


def scratch_code_output(k, metric):
    train_filename = '1a-training.txt'
    test_filename = '1a-test.txt'

    dataset_path = '../dataset/'
    input_data = fetch_data(dataset_path + train_filename)
    input_np = np.array(input_data)
    input_np = normalize_train(input_np)
    input_data = change_data_structure(input_np)

    test_data = fetch_data(dataset_path + test_filename)
    test_np = np.array(test_data)
    test_np = normalize_test(test_np)
    test_data = data_structure_test(test_np)

    y_pred_np = []

    for tp in test_data:

        if metric == 'manhattan':
            distance_array = get_manhattan_distance(input_data, tp)
        elif metric == 'euclidean':
            distance_array = get_euclidean_distance(input_data, tp)
        elif metric == 'minkowski':
            distance_array = get_minkowski_distance(input_data, tp)

        y_pred_np.append(make_prediction(k, distance_array))

    y_pred = np.array(y_pred_np)
    return y_pred


if __name__ == '__main__':
    scratch_code_output(1, 'euclidean')
