from unittest import TestCase

import numpy as np
from scipy.special import iv

from dists import bgvm


class TestBGvM(TestCase):
    def test_moment_indep_zeroth(self):
        """
        Test the moments method for normalising constant with independent components
        :return:
        """
        k1 = np.array([10., 1.])
        m1 = np.array([0.5, -0.9]) * np.pi
        W = np.zeros([2, 2])

        model = bgvm.BGvM(k1=k1, m1=m1, W=W)
        z1 = model.moment(0, 0)
        z2 = model.moment(1, 0)
        z_gt = 4. * np.pi ** 2 * iv(0, k1[0]) * iv(0, k1[1])

        np.testing.assert_almost_equal(z1, z2)
        np.testing.assert_almost_equal(z1, z_gt)

    def test_moment_indep_first(self):
        """
        Test the moments method for 1st moment with independent components
        :return:
        """
        k1 = np.array([10., 1.])
        m1 = np.array([0.5, -0.9]) * np.pi
        W = np.zeros([2, 2])

        model = bgvm.BGvM(k1=k1, m1=m1, W=W)
        z1 = model.moment(idx=0, n=1)
        z2 = model.moment(idx=1, n=1)

        z1_gt = iv(1, k1[0]) / iv(0, k1[0]) * np.exp(1.j * m1[0])
        z2_gt = iv(1, k1[1]) / iv(0, k1[1]) * np.exp(1.j * m1[1])

        np.testing.assert_almost_equal(z1, z1_gt)
        np.testing.assert_almost_equal(z2, z2_gt)
