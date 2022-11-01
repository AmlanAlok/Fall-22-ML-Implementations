from Q2_A import train_and_prediction, separate_input_output_limit, fetch_data
import numpy as np
import matplotlib.pyplot as plt


def Q2_B_sol(size):
    gamma = 0.204
    # filename = '../datasets/Q1_B_train.txt'  # debug
    filename = 'datasets/Q1_B_train.txt'        # command
    input_data = fetch_data(filename)
    x_data, y_data = separate_input_output_limit(input_data, size)
    y_prediction, y_array = [], []

    ''' For a smooth line'''
    x_array = np.linspace(-3, 3, 1000)

    # for x in x_array:
    #     prediction = train_and_prediction(x, gamma, 128, x_data, y_data)
    #     y_prediction.append(prediction)

    for x_point in x_data:
        x = x_point[0]
        prediction = train_and_prediction(x, gamma, 128, x_data, y_data)
        y_prediction.append(prediction)

    y_prediction = np.array(y_prediction)

    # plt.plot(x_array, y_prediction)
    # plt.plot(x_data, y_data)

    # plt.scatter(x_data, y_data)           # to see plot of given data
    plt.scatter(x_data, y_prediction)       # gives good output

    plt.title('Locally Weighted Linear Regression')
    # plt.legend(line_names)
    # plt.savefig('Q2_B_Pic_training_points')
    plt.savefig('Q2_B_Pic')


def main():
    print('Q2_B --------------------')
    Q2_B_sol(128)


if __name__ == "__main__":
    main()
