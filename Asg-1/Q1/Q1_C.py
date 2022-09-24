import math
import numpy as np


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):

    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


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

    # print('Converted each data point to dictionary format')

    return data_list


def leave_one_out(input_data, k, exclude_age=False):

    y_pred = []

    for left_out_dp in input_data:

        a = [None] * (len(input_data)-1)
        i = 0

        for dp in input_data:

            if left_out_dp['index'] != dp['index']:
                if exclude_age:
                    cartesian_distance = math.sqrt(
                        pow(dp['input']['height'] - left_out_dp['input']['height'], 2) +
                        pow(dp['input']['weight'] - left_out_dp['input']['weight'], 2)
                    )
                else:
                    cartesian_distance = math.sqrt(
                        pow(dp['input']['height'] - left_out_dp['input']['height'], 2) +
                        pow(dp['input']['weight'] - left_out_dp['input']['weight'], 2) +
                        pow(dp['input']['age'] - left_out_dp['input']['age'], 2)
                    )

                a[i] = {
                    'index': dp['index'],
                    'cartesian_distance': cartesian_distance,
                    'output': dp['output']
                }

                i += 1

        sorted_a = sorted(a, key=lambda d: d['cartesian_distance'])

        knn_array = sorted_a[:k]
        w_count = 0
        m_count = 0

        for dic in knn_array:
            if dic['output'] == 'W':
                w_count += 1
            if dic['output'] == 'M':
                m_count += 1

        w_prob = w_count / k
        m_prob = m_count / k

        # print('W prob =', w_prob)
        # print('M prob =', m_prob)

        prediction = ''

        if w_count > m_count:
            prediction = 'W'
        elif m_count > w_count:
            prediction = 'M'
        else:
            prediction = '50-50 M/W'

        if prediction == left_out_dp['output']:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred


def result_accuracy(y_pred, k):
    accuracy = (sum(y_pred)/len(y_pred))*100
    print('For k = ' + str(k) + ', Accuracy = ' + str(accuracy))


def Q1_C():
    filename = '1c-data.txt'

    # dataset_path = '../dataset/'
    dataset_path = 'Asg-1/dataset/'

    input_data = fetch_data(dataset_path + filename)
    input_np = np.array(input_data)
    input_np = normalize_train(input_np)
    input_data = change_data_structure(input_np)

    for k in [1, 3, 5, 7, 9, 11]:
        y_pred = leave_one_out(input_data, k)
        result_accuracy(y_pred, k)

    print('END')


if __name__ == '__main__':
    Q1_C()
