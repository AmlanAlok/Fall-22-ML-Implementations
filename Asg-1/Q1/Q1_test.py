import unittest
from knn_with_sklearn import library_output
from Q1_b_knn_without_ml_library import scratch_code_output
import numpy as np


class MyTestCase(unittest.TestCase):

    '''This test case matches my output to the output of the library'''
    def test_correct_logic(self):
        for k in [1, 5, 7]:
            for metric in ['manhattan', 'euclidean', 'minkowski']:
                self.assertEqual((library_output(k, metric) == scratch_code_output(k, metric)).all(), True)


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

    def test_expected_output(self):
        print('')
        print(library_output(1, 'euclidean'))
        print(library_output(3, 'euclidean'))
        print(library_output(7, 'euclidean'))

        print(library_output(1, 'manhattan'))
        print(library_output(3, 'manhattan'))
        print(library_output(7, 'manhattan'))

        print(library_output(1, 'minkowski'))
        print(library_output(3, 'minkowski'))
        print(library_output(7, 'minkowski'))



    def test_for_students(self):
        self.assertEqual(True, (scratch_code_output(1, 'euclidean') == np.array(['W', 'M', 'M', 'W'])).all())


if __name__ == '__main__':
    unittest.main()
