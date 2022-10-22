from Q2_A import *
import numpy as np


def error_calculation_test_data(y_true, y_prediction):
    return np.square(np.subtract(y_true, y_prediction)).mean()


def Q2_C_sol(size):
    gamma = 0.204
    filename = '../datasets/Q1_C_test.txt'  # debug
    # filename = 'datasets/Q1_B_train.txt'        # command
    input_data = fetch_data(filename)
    x_data, y_true = separate_input_output_limit(input_data, size)

    y_prediction, y_array = [], []

    for x in x_data:
        prediction = train_and_prediction(x, gamma, 128, x_data, y_true)
        y_prediction.append(prediction)

    y_prediction = np.array(y_prediction)

    ''' Error Calculation '''
    mse = error_calculation_test_data(y_true, y_prediction)
    print('data size = ' + str(size) + ', MSE = ' + str(mse))


def main():
    print('Q2_C --------------------')
    Q2_C_sol(128)


if __name__ == "__main__":
    main()
