from unittest import TestCase

import numpy as np
from scipy import pi

from vm import vM as vM
from vm import vM2 as vM2


class TestVM(TestCase):
    """
    unit tests for the vM class
    """

    def test_trig_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        m1 = np.array([[-0.33, +0.25]]) * pi

        theta = vM(k1, m1)
        mc, ms, mc2, msc, ms2, dmc, dms, dmc2, dmsc, dms2 = theta.moments(get_gradients=True)

        delta = 1.e-8
        for ii in xrange(0, 2):
            if ii == 0:
                theta_m = vM(k1 - delta, m1)
                theta_p = vM(k1 + delta, m1)
            elif ii == 1:
                theta_m = vM(k1, m1 - delta)
                theta_p = vM(k1, m1 + delta)
            else:
                raise NotImplementedError

            mc_m, ms_m, mc2_m, msc_m, ms2_m = theta_m.moments(get_gradients=False)
            mc_p, ms_p, mc2_p, msc_p, ms2_p = theta_p.moments(get_gradients=False)

            mc_fd = (mc_p - mc_m) / (2. * delta)
            ms_fd = (ms_p - ms_m) / (2. * delta)
            mc2_fd = (mc2_p - mc2_m) / (2. * delta)
            msc_fd = (msc_p - msc_m) / (2. * delta)
            ms2_fd = (ms2_p - ms2_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dmc[ii], mc_fd)
            np.testing.assert_array_almost_equal(dms[ii], ms_fd)
            np.testing.assert_array_almost_equal(dmc2[ii], mc2_fd)
            np.testing.assert_array_almost_equal(dmsc[ii], msc_fd)
            np.testing.assert_array_almost_equal(dms2[ii], ms2_fd)

    def test_entropy_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        m1 = np.array([[-0.33, +0.25]]) * pi

        theta = vM(k1, m1)
        dh = theta.entropy(get_gradients=True)[1]

        delta = 1.e-8
        for ii in xrange(0, 2):
            if ii == 0:
                theta_m = vM(k1 - delta, m1)
                theta_p = vM(k1 + delta, m1)
            elif ii == 1:
                theta_m = vM(k1, m1 - delta)
                theta_p = vM(k1, m1 + delta)
            else:
                raise NotImplementedError

            h_m = theta_m.entropy(get_gradients=False)
            h_p = theta_p.entropy(get_gradients=False)

            h_fd = (h_p - h_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dh[ii], h_fd)

    def test_alternate_vm_trig_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        m1 = np.array([[-0.33, +0.25]]) * pi

        theta = vM2(k1 * np.cos(m1), k1 * np.sin(m1))
        mc, ms, mc2, msc, ms2, dmc, dms, dmc2, dmsc, dms2 = theta.moments(get_gradients=True)

        delta = 1.e-8
        for ii in xrange(0, 2):
            if ii == 0:
                theta_m = vM2(k1 * np.cos(m1) - delta, k1 * np.sin(m1))
                theta_p = vM2(k1 * np.cos(m1) + delta, k1 * np.sin(m1))
            elif ii == 1:
                theta_m = vM2(k1 * np.cos(m1), k1 * np.sin(m1) - delta)
                theta_p = vM2(k1 * np.cos(m1), k1 * np.sin(m1) + delta)
            else:
                raise NotImplementedError

            mc_m, ms_m, mc2_m, msc_m, ms2_m = theta_m.moments(get_gradients=False)
            mc_p, ms_p, mc2_p, msc_p, ms2_p = theta_p.moments(get_gradients=False)

            mc_fd = (mc_p - mc_m) / (2. * delta)
            ms_fd = (ms_p - ms_m) / (2. * delta)
            mc2_fd = (mc2_p - mc2_m) / (2. * delta)
            msc_fd = (msc_p - msc_m) / (2. * delta)
            ms2_fd = (ms2_p - ms2_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dmc[ii], mc_fd)
            np.testing.assert_array_almost_equal(dms[ii], ms_fd)
            np.testing.assert_array_almost_equal(dmc2[ii], mc2_fd)
            np.testing.assert_array_almost_equal(dmsc[ii], msc_fd)
            np.testing.assert_array_almost_equal(dms2[ii], ms2_fd)

    def test_alternate_vm_entropy_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        m1 = np.array([[-0.33, +0.25]]) * pi

        theta = vM2(k1 * np.cos(m1), k1 * np.sin(m1))
        dh = theta.entropy(get_gradients=True)[1]

        delta = 1.e-8
        for ii in xrange(0, 2):
            if ii == 0:
                theta_m = vM2(k1 * np.cos(m1) - delta, k1 * np.sin(m1))
                theta_p = vM2(k1 * np.cos(m1) + delta, k1 * np.sin(m1))
            elif ii == 1:
                theta_m = vM2(k1 * np.cos(m1), k1 * np.sin(m1) - delta)
                theta_p = vM2(k1 * np.cos(m1), k1 * np.sin(m1) + delta)
            else:
                raise NotImplementedError

            h_m = theta_m.entropy(get_gradients=False)
            h_p = theta_p.entropy(get_gradients=False)

            h_fd = (h_p - h_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dh[ii], h_fd)
