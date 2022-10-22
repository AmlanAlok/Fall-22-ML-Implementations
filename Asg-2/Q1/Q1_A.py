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
    filename = 'datasets/Q1_B_train.txt'        # python command
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
    k = 4
    max_d = 6
    # x = np.linspace(-3, 3, 1000)

    training_size = [128, 20]  # max = 128

    for size in training_size:

        line_names = []

        # for d in [1]:
        for d in range(max_d + 1):
            parameter_matrix = train_linear_regression_model(k, d, size)

            print('parameter matrix for d = ', d)
            print(parameter_matrix)
            '''You can save np array using this function'''
            # np.savetxt('./Q1/parameter-d-'+str(d)+'.csv', parameter_matrix, delimiter=',')

            ''' plotting graph '''
            # plt.plot(x, prediction(x, parameter_matrix, k, d))

            ''' Error Calculation '''
            # mse = error_calculation_test_data(parameter_matrix, k, d)

            # line_names.append('d=' + str(d) + ', MSE=' + str(mse))
            print('A')

        # # Reading the csv into an array
        # # firstarray = np.genfromtxt("firstarray.csv", delimiter=",")
        # plt.title('Training Data Size =' + str(size))
        # plt.legend(line_names)
        # # plt.savefig('python/Q1/Q1-size-'+str(size))     # with python command
        # plt.savefig('Q1/Q1-size-' + str(size))  # debug
        # # plt.savefig('./Q1/Q1-size-' + str(size) + '-overfitting')
        # # plt.show()
        # # plt.close()

    print('program ended')
    print('--------------------------')
    pass


if __name__ == "__main__":
    main()
