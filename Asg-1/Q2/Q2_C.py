import math
import numpy as np

OUTPUT = 'output'
INPUT = 'input'
COUNT = 'count'
HEIGHT = 'height'
WEIGHT = 'weight'
AGE = 'age'
HEIGHT_TOTAL = 'height_total'
HEIGHT_MEAN = 'height_mean'
HEIGHT_VAR = 'height_var'
HEIGHT_MEAN_SQR_TOTAL = 'height_mean_sqr_total'
WEIGHT_TOTAL = 'weight_total'
WEIGHT_MEAN = 'weight_mean'
WEIGHT_VAR = 'weight_var'
WEIGHT_MEAN_SQR_TOTAL = 'weight_mean_sqr_total'
AGE_TOTAL = 'age_total'
AGE_MEAN = 'age_mean'
AGE_VAR = 'age_var'
AGE_MEAN_SQR_TOTAL = 'age_mean_sqr_total'
CLASS_PROBABILITY = 'probability'


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


def get_prediction(input_data, input_dict, test_input, exclude_age=False):
    output_labels = input_dict.keys()

    ''' totaling all features '''
    for dp in input_data:
        input_dict[dp[OUTPUT]][HEIGHT_TOTAL] += dp[INPUT][HEIGHT]
        input_dict[dp[OUTPUT]][WEIGHT_TOTAL] += dp[INPUT][WEIGHT]
        input_dict[dp[OUTPUT]][AGE_TOTAL] += dp[INPUT][AGE]

    ''' Calculating Mean and Class Probability'''
    for label in output_labels:
        input_dict[label][HEIGHT_MEAN] = input_dict[label][HEIGHT_TOTAL] / input_dict[label][COUNT]
        input_dict[label][WEIGHT_MEAN] = input_dict[label][WEIGHT_TOTAL] / input_dict[label][COUNT]
        input_dict[label][AGE_MEAN] = input_dict[label][AGE_TOTAL] / input_dict[label][COUNT]
        input_dict[label][CLASS_PROBABILITY] = input_dict[label][COUNT] / len(input_data)

    ''' Calculating square of difference from Mean for each data point'''
    for dp in input_data:
        input_dict[dp[OUTPUT]][HEIGHT_MEAN_SQR_TOTAL] += pow(dp[INPUT][HEIGHT] - input_dict[dp[OUTPUT]][HEIGHT_MEAN], 2)
        input_dict[dp[OUTPUT]][WEIGHT_MEAN_SQR_TOTAL] += pow(dp[INPUT][WEIGHT] - input_dict[dp[OUTPUT]][WEIGHT_MEAN], 2)
        input_dict[dp[OUTPUT]][AGE_MEAN_SQR_TOTAL] += pow(dp[INPUT][AGE] - input_dict[dp[OUTPUT]][AGE_MEAN], 2)

    ''' Calculating Variance '''
    for label in output_labels:
        input_dict[label][HEIGHT_VAR] = input_dict[label][HEIGHT_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)
        input_dict[label][WEIGHT_VAR] = input_dict[label][WEIGHT_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)
        input_dict[label][AGE_VAR] = input_dict[label][AGE_MEAN_SQR_TOTAL] / (input_dict[label][COUNT] - 1)

    test_input['probability'] = {}
    max_probability = 0.0

    for label in output_labels:
        test_input['probability'][label] = {
            'P(height|C)': gaussian_formula(mean=input_dict[label][HEIGHT_MEAN], var=input_dict[label][HEIGHT_VAR],
                                            test_parameter=test_input[INPUT][HEIGHT]),
            'P(weight|C)': gaussian_formula(mean=input_dict[label][WEIGHT_MEAN], var=input_dict[label][WEIGHT_VAR],
                                            test_parameter=test_input[INPUT][WEIGHT]),
            'P(age|C)': gaussian_formula(mean=input_dict[label][AGE_MEAN], var=input_dict[label][AGE_VAR],
                                         test_parameter=test_input[INPUT][AGE])
        }

        '''
        Gaussian Naive Bayes output of M = P(M)*P(height|M)*P(weight|M)*P(age|M)
        Gaussian Naive Bayes output of W = P(W)*P(height|W)*P(weight|W)*P(age|W)

        The greater probability is chosen
        '''
        if exclude_age:
            test_input['probability'][label]['final_estimate'] = input_dict[label][CLASS_PROBABILITY] * \
                                                                 test_input['probability'][label]['P(height|C)'] * \
                                                                 test_input['probability'][label]['P(weight|C)']
        else:
            test_input['probability'][label]['final_estimate'] = input_dict[label][CLASS_PROBABILITY] * \
                                                                 test_input['probability'][label]['P(height|C)'] * \
                                                                 test_input['probability'][label]['P(weight|C)'] * \
                                                                 test_input['probability'][label]['P(age|C)']

        # print('For', label, 'Final Estimate =', test_input['probability'][label]['final_estimate'])

        if test_input['probability'][label]['final_estimate'] > max_probability:
            max_probability = test_input['probability'][label]['final_estimate']
            test_input['prediction'] = label

    return test_input['prediction']


def leave_one_out(input_data, exclude_age=False):

    y_pred = []

    for left_out_dp in input_data:

        input_dict = {}

        for dp in input_data:

            '''excluding test data check'''
            if dp['index'] != left_out_dp['index']:
                if dp[OUTPUT] in input_dict:

                    '''calculating the number of W and M in dataset'''
                    input_dict[dp[OUTPUT]][COUNT] += 1
                else:
                    input_dict[dp[OUTPUT]] = {
                        COUNT: 1, CLASS_PROBABILITY: 0,
                        HEIGHT_TOTAL: 0, HEIGHT_MEAN: 0, HEIGHT_VAR: 0, HEIGHT_MEAN_SQR_TOTAL: 0,
                        WEIGHT_TOTAL: 0, WEIGHT_MEAN: 0, WEIGHT_VAR: 0, WEIGHT_MEAN_SQR_TOTAL: 0,
                        AGE_TOTAL: 0, AGE_MEAN: 0, AGE_VAR: 0, AGE_MEAN_SQR_TOTAL: 0,
                    }

        test_input_record = {
            'input': {
                'height': float(left_out_dp[INPUT][HEIGHT]),
                'weight': float(left_out_dp[INPUT][WEIGHT]),
                'age': int(left_out_dp[INPUT][AGE])
            },
            'output': left_out_dp[OUTPUT]
        }

        test_input_pred = get_prediction(input_data, input_dict, test_input_record, exclude_age)

        if test_input_record[OUTPUT] == test_input_pred:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred


def gaussian_formula(mean, var, test_parameter):
    pi = math.pi
    e = math.e
    r = (1/math.sqrt(2*pi*var)) * pow(e, (-1*pow(test_parameter-mean, 2))/(2*var))
    return r


def result_accuracy(y_pred):
    accuracy = (sum(y_pred)/len(y_pred))*100
    print('Accuracy = ' + str(accuracy))


def Q2_C():
    filename = '1c-data.txt'

    dataset_path = '../dataset/'
    # dataset_path = 'Asg-1/dataset/'
    input_data = fetch_data(dataset_path + filename)
    input_np = np.array(input_data)
    input_np = normalize_train(input_np)
    input_data = change_data_structure(input_np)

    y_pred = leave_one_out(input_data)
    result_accuracy(y_pred)

    print('END')


if __name__ == '__main__':
    Q2_C()
