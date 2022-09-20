import unittest
from knn_with_library import lib_output


class MyTestCase(unittest.TestCase):

    def test_knn_predictions_via_sklearn(self):
        print('')
        for k in [1,3,7]:
            print(lib_output(k, 'euclidean'))
            print(lib_output(k, 'manhattan'))
            print(lib_output(k, 'minkowski'))
            print('-----------------')


if __name__ == '__main__':
    unittest.main()
