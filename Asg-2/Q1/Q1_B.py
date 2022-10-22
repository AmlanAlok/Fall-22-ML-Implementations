import numpy as np
import matplotlib.pyplot as plt
from Q1_A import get_parameter_matrix_with_depth_and_size, get_feature_vector


def prediction(x, parameter_matrix, k, d):
    x_data = x.reshape(x.shape[0], 1)
    feature_vector = get_feature_vector(x_data, k, d)
    prediction = np.matmul(feature_vector, parameter_matrix)

    return prediction


def Q1_B_sol(k, size):
    x = np.linspace(0, 1, 1000)
    line_names = []

    for d in range(7):
        parameter_matrix = get_parameter_matrix_with_depth_and_size(k, d)
        plt.plot(x, prediction(x, parameter_matrix, k, d))
        line_names.append('d=' + str(d))

    plt.title('Training Data Size =' + str(size))
    plt.legend(line_names)
    plt.savefig('Q1/B_Pic/Q1_B_pic_size_' + str(size) + '_k_' + str(k))
    plt.clf()


def main():
    print('Q1_B --------------------')
    max_k = 10
    for k in range(1, max_k+1):
        print('Generating plots for k =', k)
        Q1_B_sol(k, 128)


if __name__ == "__main__":
    main()
