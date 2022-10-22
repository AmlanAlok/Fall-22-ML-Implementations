import numpy as np
import matplotlib.pyplot as plt
from Q1_A import get_parameter_matrix_with_depth_and_size
from Q1_A import fetch_data, separate_input_output
from Q1_B import prediction


def error_calculation_test_data(parameter_matrix, k, d):
    filename = '../datasets/Q1_C_test.txt'  # debug
    # filename = 'datasets/Q1_c_test_data.txt'  # python command
    test_data = fetch_data(filename)

    x_data, y_true = separate_input_output(test_data)

    y_prediction = prediction(x_data, parameter_matrix, k, d)

    ''' calculating mean square error '''
    mse = np.square(np.subtract(y_true, y_prediction)).mean()

    return mse, x_data, y_prediction


def Q1_C_sol(k, size):
    x = np.linspace(0, 1, 1000)
    line_names = []

    for d in range(7):
        parameter_matrix = get_parameter_matrix_with_depth_and_size(d, size)
        mse, x_data, y_prediction = error_calculation_test_data(parameter_matrix, k, d)
        print('For d =', d, 'MSE = ', mse)

        # parameter_matrix = get_parameter_matrix_with_depth_and_size(d)
        # plt.plot(x_data, y_prediction)
        plt.plot(x, prediction(x, parameter_matrix, k, d))
        line_names.append('d=' + str(d) + ' MSE =' + str(mse))

    plt.title('Training Data Size =' + str(size))
    plt.legend(line_names)
    plt.savefig('Q1_C_pic_size_'+str(size))


def main():
    print('Q1_C --------------------')
    Q1_C_sol(4, 128)


if __name__ == "__main__":
    main()

'''
Question

Compare the error results and try to determine for what “function depths” overfitting might be a problem. 
Which ”function depth” would you consider the best prediction function and why.

Answer
'''