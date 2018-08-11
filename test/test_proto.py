import unittest
import numpy as np

from prototyping.model.prep_and_train import zero_pad


class TestNLPMethods(unittest.TestCase):

    def test_zero_pad(self):
        expected = [[[1, 1], [0, 0]], [[1, 3], [2, 4]]]
        actual = zero_pad([[[1, 1]], [[1, 3], [2, 4]]])

        for i in range(len(expected)):
            for j in range(len(expected[0])):
                for k in range(len(expected[0][0])):
                    self.assertEqual(expected[i][j][k], actual[i][j][k])

    def test_nonexistent_words(self):
        pass
