import numpy as np
import matplotlib.pyplot as plt


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(file_name):
    # print('inside func '+ inspect.stack()[0][3])

    with open(file_name, 'r') as f:
        input_data = f.readlines()
        # print(type(input_data ))
        # print('Number of data points =', len(input_data ))

        clean_input = list(map(clean_data, input_data))

        f.close()

    return clean_input


'''Uses all Data'''


def separate_input_output(input_data):
    td = np.array(input_data)

    height_data_points = td[:, 0]
    height_data = height_data_points.reshape(height_data_points.shape[0], 1)
    height_data = height_data.astype('float64')

    weight_data_points = td[:, 1]
    weight_data = weight_data_points.reshape(weight_data_points.shape[0], 1)
    weight_data = weight_data.astype('float64')

    age_data_points = td[:, 2]
    age_data = age_data_points.reshape(age_data_points.shape[0], 1)
    age_data = age_data.astype('int64')

    y_data_points = td[:, 3]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)

    return height_data, weight_data, age_data, y_data


def get_feature_matrix(height_data, weight_data, age_data):
    return np.array([1, height_data, weight_data, age_data], dtype='float64').reshape(4, 1)


def get_random_parameter_matrix():
    return np.random.randn(4, 1)


def change_y_data(y_data):
    value_01 = np.unique(y_data, return_inverse=True)[1]
    return value_01


def final_prediction(height_data, weight_data, age_data, parameter_matrix):

    y_pred_arr = []

    for i in range(height_data.shape[0]):
        feature_matrix = get_feature_matrix(height_data[i], weight_data[i], age_data[i])

        y_pred_arr.append(prediction(parameter_matrix, feature_matrix))

    y_pred = np.array(y_pred_arr)
    y_pred = y_pred.reshape(y_pred.shape[0], 1)
    return y_pred


def combine_result(height_data, weight_data, age_data, y_pred):

    r1 = np.concatenate((height_data, weight_data), axis=1)
    r2 = np.concatenate((r1, age_data), axis=1)
    r3 = np.concatenate((r2, y_pred), axis=1)
    return r3


def split_predictions_by_label(combined_data):

    filter_1 = combined_data[:, 3] == 1
    filter_0 = combined_data[:, 3] == 0
    pred_1 = combined_data[filter_1]
    pred_0 = combined_data[filter_0]
    return pred_1, pred_0


def get_surface_func(parameter_matrix, x, y):

    bias = parameter_matrix[0]
    h_coeff = parameter_matrix[1]
    w_coeff = parameter_matrix[2]
    a_coeff = parameter_matrix[3]
    z = -1*(bias + h_coeff*x + w_coeff*y)/a_coeff
    return z


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def prediction(parameter_matrix, feature_matrix):
    linear_regression_output = np.matmul(np.transpose(parameter_matrix), feature_matrix)

    sigmoid_output = sigmoid(linear_regression_output)

    if sigmoid_output >= 0.5:
        return 1
    return 0


def train(alpha, iterations):
    filename = '../datasets/Q3_data.txt'  # debug
    # filename = 'datasets/Q3_data.txt'   # command line
    input_data = fetch_data(filename)

    height_data, weight_data, age_data, y_data = separate_input_output(input_data)
    parameter_matrix = get_random_parameter_matrix()
    y_data_01 = change_y_data(y_data)

    for k in range(iterations):

        error_array = []

        for i in range(height_data.shape[0]):
            feature_matrix = get_feature_matrix(height_data[i], weight_data[i], age_data[i])

            y_prediction = prediction(parameter_matrix, feature_matrix)

            err = y_prediction - y_data_01[i]
            error_array.append(err)

            partial_derivative = alpha * err * feature_matrix

            parameter_matrix = parameter_matrix - partial_derivative
            # print('p')

        error_np = np.array(error_array)
        accuracy = 100 - (np.sum(np.square(error_np)) / error_np.size) * 100
        print('Itr =', k, ' accuracy =', accuracy)

    # print(parameter_matrix)

    y_pred = final_prediction(height_data, weight_data, age_data, parameter_matrix)
    combined_data = combine_result(height_data, weight_data, age_data, y_pred)
    pred_1, pred_0 = split_predictions_by_label(combined_data)

    ax = plt.axes(projection='3d')
    # ax.scatter3D(height_data, weight_data, age_data)
    ax.scatter3D(pred_1[:, 0], pred_1[:, 1], pred_1[:, 2], cmap='green')
    ax.scatter3D(pred_0[:, 0], pred_0[:, 1], pred_0[:, 2], cmap='red')

    h_vals = np.linspace(1.3, 2.0, 100)
    w_vals = np.linspace(60, 100, 100)

    X, Y = np.meshgrid(h_vals, w_vals)
    Z = get_surface_func(parameter_matrix, X, Y)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Age')
    plt.savefig('Q3_B')  # debug
    # plt.savefig('Q3/Q3_B')  # debug
    # plt.savefig('python/Q3/Q3_plot')    # command line
    plt.show()

    return parameter_matrix


def main():
    print('Q3_AB --------------------')
    print('program started')
    alpha = 0.01
    iterations = 20
    parameter_matrix = train(alpha, iterations)

    print('program ended')
    print('--------------------------')


if __name__ == "__main__":
    main()
