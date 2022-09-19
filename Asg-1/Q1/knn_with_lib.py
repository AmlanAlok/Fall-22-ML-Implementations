import unittest
import numpy as np

def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

def fetch_data(filename):
    # print('inside func '+ inspect.stack()[0][3])

    with open(filename, 'r') as f:
        input_data = f.readlines()
        # print(type(input_data ))
        # print('Number of data points =', len(input_data ))

        clean_input = list(map(clean_data, input_data))
        f.close()

    # clean_input = change_data_structure(clean_input)
    return clean_input

def input_data():
    file_path = '../dataset/1a-training.txt'
    input_data = fetch_data(file_path)
    np.savetxt('1a-training.csv', np.array(input_data), delimiter=',', fmt='%s', comments='')
    print(input_data)

class MyTestCase(unittest.TestCase):

    def test_something(self):
        input_data()
        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
