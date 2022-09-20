import unittest
from knn_with_library import library_output
from x2 import scratch_code_output


class MyTestCase(unittest.TestCase):

    def test_knn_predictions_via_sklearn(self):
        print('')
        for k in [1, 3, 7]:
            print(library_output(k, 'euclidean'))
            print(library_output(k, 'manhattan'))
            print(library_output(k, 'minkowski'))
            print('-----------------')
        print('-----')
        for k in [1, 3, 7]:
            print(scratch_code_output(k, 'euclidean'))
            print(scratch_code_output(k, 'manhattan'))
            print(scratch_code_output(k, 'minkowski'))
            print('-----------------')

    def test_x1(self):
        for k in [1, 5, 7]:
            for metric in ['manhattan', 'euclidean', 'minkowski']:
                self.assertEqual((library_output(k, metric) == scratch_code_output(k, metric)).all(), True)


if __name__ == '__main__':
    unittest.main()
