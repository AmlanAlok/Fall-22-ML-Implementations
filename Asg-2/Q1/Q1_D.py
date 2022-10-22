from Q1_B import Q1_B_sol
from Q1_C import Q1_C_sol


def main():
    print('Q1_D --------------------')
    max_k = 10
    for k in range(1, max_k + 1):
        print('Generating plots for k =', k)
        Q1_B_sol(k, 20)

    for k in range(1, max_k + 1):
        print('Generating plots for k =', k)
        Q1_C_sol(k, 20)


if __name__ == "__main__":
    main()

'''
Question

What differences do you see and why might they occur?

'''
