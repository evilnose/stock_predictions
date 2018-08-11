import unittest

import numpy as np
from model.prep_and_train import get_packaged_ind, preprocessing, from_2d_ind


class TestDataPrepMethods(unittest.TestCase):
    def test_package_ind(self):
        m = 50
        window = 6
        n_days = 1
        x_ind, y_ind = get_packaged_ind(m, window, n_days)
        # Basic tests
        self.assertEqual(len(x_ind), len(y_ind))
        self.assertEqual(len(x_ind), m - n_days - window + 1)  # len should be m - 1 - n_days - (window - 1) + 1
        # Test that size of each example is correct
        for ind in x_ind:
            self.assertEqual(len(ind), window)
        for ind in y_ind:
            self.assertEqual(type(ind), int)
        # Test n_days
        for i in range(len(x_ind)):
            self.assertEqual(x_ind[i][-1] + n_days, y_ind[i])
        # Test boundaries
        self.assertEqual(x_ind[0], list(range(window)))
        self.assertEqual(x_ind[-1], list(range(m - n_days - window, m - n_days)))
        self.assertEqual(y_ind[0], window - 1 + n_days)
        self.assertEqual(y_ind[-1], m - 1)

    def test_packaged_input_shapes(self):
        data_shape = (600, 10)
        win = 5
        n_days = 7
        m = data_shape[0]
        sample_data = np.zeros(shape=data_shape)
        x_ind, y_ind = get_packaged_ind(m, window=win, n_days_forward=n_days)
        x = from_2d_ind(sample_data, x_ind)
        y = sample_data[y_ind]
        m_res = m - n_days - win + 1  # The resulting num of training examples
        self.assertEqual(x.shape, (m_res,) + data_shape[1:])
        self.assertEqual(y.shape, (m_res,) + data_shape[1:])
