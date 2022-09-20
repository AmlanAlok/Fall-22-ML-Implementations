import unittest
from knn_with_lib import lib_output


class MyTestCase(unittest.TestCase):

    def test_something(self):
        print('')
        print(lib_output(1))
        # print(lib_output(3))
        # print(lib_output(5))
        # print(lib_output(7))


if __name__ == '__main__':
    unittest.main()
