import numpy as np
import matplotlib.pyplot as plt


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(file_name):
    with open(file_name, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()

    return clean_input


def get_feature_vector(x_data, k, d):
    vector = np.ones((x_data.shape[0], 1))
    # p = np.concatenate((ones, x_data), axis=1)

    i = 1

    while i <= d:
        ikx = x_data * i * k

        sin_col = np.sin(ikx) * np.sin(ikx)
        vector = np.concatenate((vector, sin_col), axis=1)

        # cos_col = np.cos(ikx)
        # p = np.concatenate((p, cos_col), axis=1)

        i += 1

    return vector


'''Uses all Data'''


def separate_input_output(input_data):
    td = np.array(input_data, dtype='float64')

    x_data_points = td[:, 0]
    x_data = x_data_points.reshape(x_data_points.shape[0], 1)

    y_data_points = td[:, 1]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)

    return x_data, y_data


'''Uses all Data'''


def separate_input_output_limit(input_data, limit=129):
    td = np.array(input_data, dtype='float64')

    x_data_points = td[:limit, 0]
    x_data = x_data_points.reshape(x_data_points.shape[0], 1)

    y_data_points = td[:limit, 1]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)

    return x_data, y_data


def train_linear_regression_model(k, d, size):
    # filename = '../datasets/Q1_B_train.txt'  # debug
    filename = 'datasets/Q1_B_train.txt'  # python command
    input_data = fetch_data(filename)

    # x_data, y_data = separate_input_output(input_data)
    x_data, y_data = separate_input_output_limit(input_data, size)

    feature_matrix = get_feature_vector(x_data, k, d)

    ''' moore-penrose pseudoinverse numpy '''
    pseudo_inv = np.linalg.pinv(feature_matrix)

    parameter_matrix = np.matmul(pseudo_inv, y_data)

    return parameter_matrix


def get_parameter_matrix_with_depth_and_size(k, depth, size=128):
    return train_linear_regression_model(k, depth, size)


def main():
    print('program started')
    print('Q1_A --------------------')
    max_k = 10
    max_d = 6

    training_size = [128]  # max = 128

    for size in training_size:
        for k in range(1, max_k + 1):
            print('\nk = ', k, '--------------------\n')
            for d in range(max_d + 1):
                parameter_matrix = train_linear_regression_model(k, d, size)
                print('parameter matrix for d = ', d)
                print(parameter_matrix)

    print('program ended')
    print('--------------------------')
    pass


if __name__ == "__main__":
    main()
