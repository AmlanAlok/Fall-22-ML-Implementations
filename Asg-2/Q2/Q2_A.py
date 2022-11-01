import numpy as np


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


def get_feature_vector(x):
    ones = np.ones((x.shape[0], 1))
    p = np.concatenate((ones, x), axis=1)

    return p


'''Uses all Data'''


def separate_input_output_limit(input_data, limit=129):
    td = np.array(input_data, dtype='float64')

    x_data_points = td[:limit, 0]
    x_data = x_data_points.reshape(x_data_points.shape[0], 1)

    y_data_points = td[:limit, 1]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)

    return x_data, y_data


def weight_function(x, gamma, query_point):
    numerator = np.square(x - query_point)
    denominator = (2 * (gamma ** 2))
    ans = np.exp(-1 * (numerator/ denominator))
    return ans


def get_feature_matrix(x):
    fm = np.array([1, x], dtype='float64')
    fm = fm.reshape(1, fm.shape[0])
    return fm


def prediction(parameter_matrix, query_point):
    feature_matrix = get_feature_matrix(query_point)
    ans = np.matmul(feature_matrix, parameter_matrix)
    return ans


def train_and_prediction(x, gamma, size, x_data, y_data):
    query_point = x
    # filename = '../datasets/Q1_b_training_data.txt'  # debug
    # # filename = 'datasets/Q1_b_training_data.txt'        # command
    # input_data = fetch_data(filename)
    # x_data, y_data = separate_input_output_limit(input_data, size)

    weight_of_x = weight_function(x_data, gamma, query_point)
    weighted_x = np.sqrt(weight_of_x) * x_data
    weighted_y = np.sqrt(weight_of_x) * y_data

    feature_matrix = get_feature_vector(weighted_x)

    ''' moore-penrose pseudoinverse numpy '''
    pseudo_inv = np.linalg.pinv(feature_matrix)

    parameter_matrix = np.matmul(pseudo_inv, weighted_y)

    y_prediction = prediction(parameter_matrix, query_point)

    return y_prediction[0]


# if __name__ == "__main__":
#     main()
