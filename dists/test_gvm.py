import time
from unittest import TestCase

from numba.decorators import jit
from scipy import pi

import dists.fast_gvm as fgvm
from dists.gvm import *
from dists.gvm import GvM as GvM
from utils.plotting import *
from vm import vM as vM


def z_func(params):
    th = np.linspace(0., 2. * np.pi, 5000)
    return np.sum(np.exp(params[0] * np.cos(th - params[2]) + params[1] * np.cos(2. * (th - params[3]))))


def log_z_euler(k1, k2, m1, m2):
    n_pts = 10000
    R, C = k1.shape
    N = R * C

    k1 = np.repeat(k1.reshape([N, 1]), n_pts, 1)
    k2 = np.repeat(k2.reshape([N, 1]), n_pts, 1)
    m1 = np.repeat(m1.reshape([N, 1]), n_pts, 1)
    m2 = np.repeat(m2.reshape([N, 1]), n_pts, 1)

    th = np.repeat(np.reshape(np.linspace(-np.pi, np.pi, n_pts), (1, n_pts)), N, 0)
    delta = th[0, 1] - th[0, 0]

    T = np.log(np.sum(np.exp(k1 * np.cos(th - m1) + k2 * np.cos(2 * (th - m2)) - k1 - k2), 1) * delta) + k1[:, 0] + k2[
                                                                                                                    :,
                                                                                                                    0]

    return T.reshape([R, C])


def loopy_log_z_euler(th, k1, k2, m1, m2):
    delta = th[1] - th[0]
    R, C = k1.shape
    T = np.empty((R, C), dtype=np.float)
    for rr in range(0, R):
        for cc in range(0, C):
            T[rr, cc] = np.log(np.sum(np.exp(k1[rr, cc] * np.cos(th - m1[rr, cc]) +
                                             k2[rr, cc] * np.cos(2 * (th - m2[rr, cc])) -
                                             k1[rr, cc] - k2[rr, cc])) * delta) + \
                        k1[rr, cc] + k2[rr, cc]
    return T


def loopy_log_z_trapz(th, k1, k2, m1, m2):
    N = th.shape[0]
    R, C = k1.shape
    T = np.empty((R, C), dtype=np.float)
    for rr in range(0, R):
        for cc in range(0, C):
            T[rr, cc] = np.log(np.pi / N * (np.exp(k1[rr, cc] * np.cos(th[0] - m1[rr, cc]) +
                                                   k2[rr, cc] * np.cos(2 * (th[0] - m2[rr, cc])) -
                                                   k1[rr, cc] - k2[rr, cc]) +
                                            np.exp(k1[rr, cc] * np.cos(th[N - 1] - m1[rr, cc]) +
                                                   k2[rr, cc] * np.cos(2 * (th[N - 1] - m2[rr, cc])) -
                                                   k1[rr, cc] - k2[rr, cc]) +
                                            2. * np.sum(np.exp(k1[rr, cc] * np.cos(th[1:-1] - m1[rr, cc]) +
                                                               k2[rr, cc] * np.cos(2 * (th[1:-1] - m2[rr, cc])) -
                                                               k1[rr, cc] - k2[rr, cc])))
                               ) + k1[rr, cc] + k2[rr, cc]
    return T


def loopy_naive_log_z_series(k1, k2, m1, m2, n):
    R, C = k1.shape
    G = np.empty((R, C), dtype=np.float)
    for rr in range(0, R):
        for cc in range(0, C):
            G[rr, cc] = ive(0, k2[rr, cc]) * ive(0, k1[rr, cc])
            delta = m1[rr, cc] - m2[rr, cc]
            for ii in range(n):
                G[rr, cc] += 2. * np.cos(2. * ii * delta) * ive(ii, k2[rr, cc]) * ive(2 * ii, k1[rr, cc])
    return G


class TestGvM(TestCase):
    """
    unit tests for the GvM Class
    """

    # Grid integration
    # OK
    def test_norm_const_vals_against_vm(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[0., 0.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[0., 0.]]) * pi

        phi = GvM(k1, k2, m1, m2)
        theta = vM(k1, m1)

        # test without gradients
        T0 = phi._grid_fun(0)

        T0_theta = 2. * pi * ive(0, theta.k1)

        np.testing.assert_array_almost_equal(T0, T0_theta)

    # OK
    def test_norm_const_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[5., 30.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2)
        delta = 1.e-8

        # test without gradients
        dT = phi._grad_grid_fun(0)
        dT_fd = [None, None, None, None]

        for ii in xrange(0, 4):
            if ii == 0:
                phi_m = GvM(k1 - delta, k2, m1, m2)
                phi_p = GvM(k1 + delta, k2, m1, m2)
            elif ii == 1:
                phi_m = GvM(k1, k2 - delta, m1, m2)
                phi_p = GvM(k1, k2 + delta, m1, m2)
            elif ii == 2:
                phi_m = GvM(k1, k2, m1 - delta, m2)
                phi_p = GvM(k1, k2, m1 + delta, m2)
            elif ii == 3:
                phi_m = GvM(k1, k2, m1, m2 - delta)
                phi_p = GvM(k1, k2, m1, m2 + delta)
            else:
                raise NotImplementedError

            phi_md = phi_m._grid_fun(0).real
            phi_pd = phi_p._grid_fun(0).real

            dT_fd[ii] = (phi_pd - phi_md) / (2. * delta)

        np.testing.assert_array_almost_equal(dT[0], dT_fd[0])
        np.testing.assert_array_almost_equal(dT[1], dT_fd[1])
        np.testing.assert_array_almost_equal(dT[2], dT_fd[2])
        np.testing.assert_array_almost_equal(dT[3], dT_fd[3])

    # OK
    def test_1st_moment_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[5., 30.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2)
        delta = 1.e-8

        # test without gradients
        dT = phi._grad_grid_fun(1)
        dT_fd = [None, None, None, None]

        for ii in xrange(0, 4):
            if ii == 0:
                phi_m = GvM(k1 - delta, k2, m1, m2)
                phi_p = GvM(k1 + delta, k2, m1, m2)
            elif ii == 1:
                phi_m = GvM(k1, k2 - delta, m1, m2)
                phi_p = GvM(k1, k2 + delta, m1, m2)
            elif ii == 2:
                phi_m = GvM(k1, k2, m1 - delta, m2)
                phi_p = GvM(k1, k2, m1 + delta, m2)
            elif ii == 3:
                phi_m = GvM(k1, k2, m1, m2 - delta)
                phi_p = GvM(k1, k2, m1, m2 + delta)
            else:
                raise NotImplementedError

            phi_md = phi_m._grid_fun(1)
            phi_pd = phi_p._grid_fun(1)

            dT_fd[ii] = (phi_pd - phi_md) / (2. * delta)

        np.testing.assert_array_almost_equal(dT[0], dT_fd[0])
        np.testing.assert_array_almost_equal(dT[1], dT_fd[1])
        np.testing.assert_array_almost_equal(dT[2], dT_fd[2])
        np.testing.assert_array_almost_equal(dT[3], dT_fd[3])

    # OK
    def test_2nd_moment_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[5., 30.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2)
        delta = 1.e-8

        # test without gradients
        dT = phi._grad_grid_fun(2)
        dT_fd = [None, None, None, None]

        for ii in xrange(0, 4):
            if ii == 0:
                phi_m = GvM(k1 - delta, k2, m1, m2)
                phi_p = GvM(k1 + delta, k2, m1, m2)
            elif ii == 1:
                phi_m = GvM(k1, k2 - delta, m1, m2)
                phi_p = GvM(k1, k2 + delta, m1, m2)
            elif ii == 2:
                phi_m = GvM(k1, k2, m1 - delta, m2)
                phi_p = GvM(k1, k2, m1 + delta, m2)
            elif ii == 3:
                phi_m = GvM(k1, k2, m1, m2 - delta)
                phi_p = GvM(k1, k2, m1, m2 + delta)
            else:
                raise NotImplementedError

            phi_md = phi_m._grid_fun(2)
            phi_pd = phi_p._grid_fun(2)

            dT_fd[ii] = (phi_pd - phi_md) / (2. * delta)

        np.testing.assert_array_almost_equal(dT[0], dT_fd[0])
        np.testing.assert_array_almost_equal(dT[1], dT_fd[1])
        np.testing.assert_array_almost_equal(dT[2], dT_fd[2])
        np.testing.assert_array_almost_equal(dT[3], dT_fd[3])

    # OK
    def test_trig_vals_against_vm(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[0., 0.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[0., 0.]]) * pi

        phi = GvM(k1, k2, m1, m2)
        theta = vM(k1, m1)

        # test with gradients
        mc, ms, m2c, m2s = phi.moments(get_gradients=False)
        mc_theta, ms_theta, m2c_theta, m2s_theta = theta.moments(get_gradients=False)

        np.testing.assert_array_almost_equal(mc, mc_theta)
        np.testing.assert_array_almost_equal(ms, ms_theta)
        np.testing.assert_array_almost_equal(m2c, m2c_theta)
        np.testing.assert_array_almost_equal(m2s, m2s_theta)

    # OK
    def test_trig_grad_against_vm(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[0., 0.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[0., 0.]]) * pi

        phi = GvM(k1, k2, m1, m2)
        theta = vM(k1, m1)
        dmc, dms, dm2c, dm2s = phi.moments(get_gradients=True)[4:]
        dmc_th, dms_th, dm2c_th, dm2s_th = theta.moments(get_gradients=True)[4:]

        np.testing.assert_array_almost_equal(dmc[0], dmc_th[0], err_msg='mc, k1')
        np.testing.assert_array_almost_equal(dms[0], dms_th[0], err_msg='ms, k1')
        np.testing.assert_array_almost_equal(dm2c[0], dm2c_th[0], err_msg='mcc, k1')
        np.testing.assert_array_almost_equal(dm2c[0], dm2c_th[0], err_msg='msc, k1')

        np.testing.assert_array_almost_equal(dmc[2], dmc_th[1], err_msg='mc, m1')
        np.testing.assert_array_almost_equal(dms[2], dms_th[1], err_msg='ms, m1')
        np.testing.assert_array_almost_equal(dm2c[2], dm2c_th[1], err_msg='mcc, m1')
        np.testing.assert_array_almost_equal(dm2c[2], dm2c_th[1], err_msg='msc, m1')

    # OK
    def test_trig_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[5., 30.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2)
        dmc, dms, dm2c, dm2s = phi.moments(get_gradients=True)[4:]

        delta = 1.e-8
        for ii in xrange(0, 4):
            if ii == 0:
                phi_m = GvM(k1 - delta, k2, m1, m2)
                phi_p = GvM(k1 + delta, k2, m1, m2)
            elif ii == 1:
                phi_m = GvM(k1, k2 - delta, m1, m2)
                phi_p = GvM(k1, k2 + delta, m1, m2)
            elif ii == 2:
                phi_m = GvM(k1, k2, m1 - delta, m2)
                phi_p = GvM(k1, k2, m1 + delta, m2)
            elif ii == 3:
                phi_m = GvM(k1, k2, m1, m2 - delta)
                phi_p = GvM(k1, k2, m1, m2 + delta)
            else:
                raise NotImplementedError

            mc_m, ms_m, m2c_m, m2s_m = phi_m.moments(get_gradients=False)
            mc_p, ms_p, m2c_p, m2s_p = phi_p.moments(get_gradients=False)

            mc_fd = (mc_p - mc_m) / (2. * delta)
            ms_fd = (ms_p - ms_m) / (2. * delta)
            m2c_fd = (m2c_p - m2c_m) / (2. * delta)
            m2s_fd = (m2s_p - m2s_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dmc[ii], mc_fd)
            np.testing.assert_array_almost_equal(dms[ii], ms_fd)
            np.testing.assert_array_almost_equal(dm2c[ii], m2c_fd)
            np.testing.assert_array_almost_equal(dm2s[ii], m2s_fd)

    # OK
    def test_entropy_vals_against_vm(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[0., 0.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[0., 0.]]) * pi

        phi = GvM(k1, k2, m1, m2)
        theta = vM(k1, m1)

        # test without gradients
        h_theta = theta.entropy(get_gradients=False)
        h = phi.entropy(get_gradients=False)

        np.testing.assert_array_almost_equal(h, h_theta)

    # OK
    def test_entropy_grad_against_vm(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[0., 0.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[0., 0.]]) * pi

        phi = GvM(k1, k2, m1, m2)
        theta = vM(k1, m1)

        dh = phi.entropy(get_gradients=True)[1]
        dh_theta = theta.entropy(get_gradients=True)[1]

        # Check the gradients for k1 and m1 (k2 has a non-zero gradient in some cases)
        np.testing.assert_array_almost_equal(dh[0], dh_theta[0])
        np.testing.assert_array_almost_equal(dh[2], dh_theta[1])

    # OK
    def test_entropy_grad_against_fd(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[5., 30.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2)
        dh = phi.entropy(get_gradients=True)[1]

        delta = 1.e-8
        for ii in xrange(0, 4):
            if ii == 0:
                phi_m = GvM(k1 - delta, k2, m1, m2)
                phi_p = GvM(k1 + delta, k2, m1, m2)
            elif ii == 1:
                phi_m = GvM(k1, k2 - delta, m1, m2)
                phi_p = GvM(k1, k2 + delta, m1, m2)
            elif ii == 2:
                phi_m = GvM(k1, k2, m1 - delta, m2)
                phi_p = GvM(k1, k2, m1 + delta, m2)
            elif ii == 3:
                phi_m = GvM(k1, k2, m1, m2 - delta)
                phi_p = GvM(k1, k2, m1, m2 + delta)
            else:
                raise NotImplementedError

            h_m = phi_m.entropy(get_gradients=False)
            h_p = phi_p.entropy(get_gradients=False)

            h_fd = (h_p - h_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dh[ii], h_fd, err_msg='ii = ' + str(ii))

    # Series approximation
    # OK
    def test_series_vals_against_vm(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[0., 0.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[0., 0.]]) * pi

        phi = GvM(k1, k2, m1, m2)
        theta = vM(k1, m1)

        # test without gradients
        mc_phi, ms_phi, m2c_phi, m2s_phi = phi.moments(get_gradients=False)
        mc_theta, ms_theta, m2c_theta, m2s_theta = theta.moments(get_gradients=False)

        np.testing.assert_array_almost_equal(mc_phi, mc_theta, err_msg='mc')
        np.testing.assert_array_almost_equal(ms_phi, ms_theta, err_msg='ms')
        np.testing.assert_array_almost_equal(m2s_phi, m2c_theta, err_msg='m2c')
        np.testing.assert_array_almost_equal(m2s_phi, m2s_theta, err_msg='m2s')

    # OK
    def test_series_vals_against_grid(self):
        k1 = np.array([[10., 5.]])
        k2 = np.array([[5., 10.]])
        m1 = np.array([[+0.33, +0.84]]) * 2. * pi
        m2 = np.array([[+0.55, +1.00]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')
        theta = GvM(k1, k2, m1, m2, method='grid')

        # test without gradients
        mc_phi, ms_phi, m2c_phi, m2s_phi = phi.moments(get_gradients=False)
        mc_theta, ms_theta, m2c_theta, m2s_theta = theta.moments(get_gradients=False)

        np.testing.assert_array_almost_equal(mc_phi, mc_theta, err_msg='mc')
        np.testing.assert_array_almost_equal(ms_phi, ms_theta, err_msg='ms')
        np.testing.assert_array_almost_equal(m2c_phi, m2c_theta, err_msg='m2c')
        np.testing.assert_array_almost_equal(m2s_phi, m2s_theta, err_msg='m2s')

    # OK
    def test_series_grad_against_fd(self):
        k1 = np.array([[10., 1.]])
        k2 = np.array([[5., 20.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2)

        # test without gradients
        g0, dg0 = phi._series_fun(0, get_gradients=True)
        g1, dg1 = phi._series_fun(1, get_gradients=True)
        g2, dg2 = phi._series_fun(2, get_gradients=True)

        delta = 1.e-9
        for ii in xrange(0, 4):

            dmc = (dg1[ii].real * g0.real - g1.real * dg0[ii].real) / (g0.real ** 2)
            dms = (dg1[ii].imag * g0.real - g1.imag * dg0[ii].real) / (g0.real ** 2)
            dmcc = +0.5 * (dg2[ii].real * g0.real - g2.real * dg0[ii].real) / (g0.real ** 2)
            dmss = -0.5 * (dg2[ii].real * g0.real - g2.real * dg0[ii].real) / (g0.real ** 2)
            dmsc = +0.5 * (dg2[ii].imag * g0.real - g2.imag * dg0[ii].real) / (g0.real ** 2)

            if ii == 0:
                phi_m = GvM(k1 - delta, k2, m1, m2)
                phi_p = GvM(k1 + delta, k2, m1, m2)
            elif ii == 1:
                phi_m = GvM(k1, k2 - delta, m1, m2)
                phi_p = GvM(k1, k2 + delta, m1, m2)
            elif ii == 2:
                phi_m = GvM(k1, k2, m1 - delta, m2)
                phi_p = GvM(k1, k2, m1 + delta, m2)
            elif ii == 3:
                phi_m = GvM(k1, k2, m1, m2 - delta)
                phi_p = GvM(k1, k2, m1, m2 + delta)
            else:
                raise NotImplementedError

            g0_m = phi_m._series_fun(0)
            g1_m = phi_m._series_fun(1)
            g2_m = phi_m._series_fun(2)

            g0_p = phi_p._series_fun(0)
            g1_p = phi_p._series_fun(1)
            g2_p = phi_p._series_fun(2)

            mc_m = g1_m.real / g0_m.real
            ms_m = g1_m.imag / g0_m.real
            mcc_m = 0.5 + 0.5 * g2_m.real / g0_m.real
            mss_m = 0.5 - 0.5 * g2_m.real / g0_m.real
            msc_m = 0.5 * g2_m.imag / g0_m.real

            mc_p = g1_p.real / g0_p.real
            ms_p = g1_p.imag / g0_p.real
            mcc_p = 0.5 + 0.5 * g2_p.real / g0_p.real
            mss_p = 0.5 - 0.5 * g2_p.real / g0_p.real
            msc_p = 0.5 * g2_p.imag / g0_p.real

            mc_fd = (mc_p - mc_m) / (2. * delta)
            ms_fd = (ms_p - ms_m) / (2. * delta)
            mcc_fd = (mcc_p - mcc_m) / (2. * delta)
            msc_fd = (msc_p - msc_m) / (2. * delta)
            mss_fd = (mss_p - mss_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dmc, mc_fd, err_msg='mc, ii = ' + str(ii))
            np.testing.assert_array_almost_equal(dms, ms_fd, err_msg='ms, ii = ' + str(ii))
            np.testing.assert_array_almost_equal(dmcc, mcc_fd, err_msg='mcc, ii = ' + str(ii))
            np.testing.assert_array_almost_equal(dmss, mss_fd, err_msg='mss, ii = ' + str(ii))
            np.testing.assert_array_almost_equal(dmsc, msc_fd, err_msg='msc, ii = ' + str(ii))

    # OK
    def test_series_grad_against_grid(self):
        k1 = np.array([[10., 1.]])
        k2 = np.array([[5., 20.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')
        theta = GvM(k1, k2, m1, m2, method='grid')

        dmc_phi, dms_phi, dm2c_phi, dm2s_phi = phi.moments(get_gradients=True)[4:]
        dmc_theta, dms_theta, dm2c_theta, dm2s_theta = theta.moments(get_gradients=True)[4:]

        delta = 1.e-9
        for ii in xrange(0, 4):
            np.testing.assert_array_almost_equal(dmc_phi[ii], dmc_theta[ii], err_msg='mc, ii = ' + str(ii))
            np.testing.assert_array_almost_equal(dms_phi[ii], dms_theta[ii], err_msg='ms, ii = ' + str(ii))
            np.testing.assert_array_almost_equal(dm2c_phi[ii], dm2c_theta[ii], err_msg='m2c, ii = ' + str(ii))
            np.testing.assert_array_almost_equal(dm2s_phi[ii], dm2s_theta[ii], err_msg='m2s, ii = ' + str(ii))

    # OK
    def test_entropy_series_vals_against_vm(self):
        k1 = np.array([[10., 100.]])
        k2 = np.array([[0., 0.]])
        m1 = np.array([[+0.33, +0.25]]) * 2. * pi
        m2 = np.array([[+0.55, +0.95]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')
        theta = vM(k1, m1)

        h_phi = phi.entropy(get_gradients=False)
        h_theta = theta.entropy(get_gradients=False)

        np.testing.assert_array_almost_equal(h_phi, h_theta)

    # OK
    def test_entropy_series_vals_against_grid(self):
        k1 = np.array([[10., 1.]])
        k2 = np.array([[5., 20.]])
        m1 = np.array([[+0.33, +0.25]]) * 2. * pi
        m2 = np.array([[+0.55, +0.95]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')
        theta = GvM(k1, k2, m1, m2)

        # test without gradients
        h_phi = phi.entropy(get_gradients=False)
        h_theta = theta.entropy(get_gradients=False)

        np.testing.assert_array_almost_equal(h_phi, h_theta)

    # OK
    def test_entropy_series_grad_against_grid(self):
        k1 = np.array([[10., 1.]])
        k2 = np.array([[5., 20.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')
        theta = GvM(k1, k2, m1, m2, method='grid', n_pts=int(1E6))

        dh_phi = phi.entropy(get_gradients=True)[1]
        dh_theta = theta.entropy(get_gradients=True)[1]

        for ii in xrange(1, 4):
            np.testing.assert_array_almost_equal(dh_phi[ii], dh_theta[ii], err_msg='ii = ' + str(ii))

    # OK
    def test_entropy_series_grad_against_fd(self):
        k1 = np.array([[10., 1.]])
        k2 = np.array([[5., 20.]])
        m1 = np.array([[-0.33, +0.25]]) * pi
        m2 = np.array([[+0.55, -0.15]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')

        # test without gradients
        dh = phi.entropy(get_gradients=True)[1]

        delta = 1.e-8
        for ii in xrange(0, 4):
            if ii == 0:
                phi_m = GvM(k1 - delta, k2, m1, m2, method='series')
                phi_p = GvM(k1 + delta, k2, m1, m2, method='series')
            elif ii == 1:
                phi_m = GvM(k1, k2 - delta, m1, m2, method='series')
                phi_p = GvM(k1, k2 + delta, m1, m2, method='series')
            elif ii == 2:
                phi_m = GvM(k1, k2, m1 - delta, m2, method='series')
                phi_p = GvM(k1, k2, m1 + delta, m2, method='series')
            elif ii == 3:
                phi_m = GvM(k1, k2, m1, m2 - delta, method='series')
                phi_p = GvM(k1, k2, m1, m2 + delta, method='series')
            else:
                raise NotImplementedError

            h_m = phi_m.entropy(get_gradients=False)
            h_p = phi_p.entropy(get_gradients=False)

            h_fd = (h_p - h_m) / (2. * delta)

            np.testing.assert_array_almost_equal(dh[ii], h_fd, err_msg='ii = ' + str(ii))

    def test__find_modes(self):

        k1 = np.array([[1., 1.]])
        k2 = np.array([[0.5, 2.]])
        m1 = np.array([[0.12, 0.6]]) * 2. * pi
        m2 = np.array([[0.55, 0.15]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')
        modes, modas = phi._find_modes()

        th = np.linspace(0, 2 * np.pi, int(1E7))
        f = lambda theta, kappa1, kappa2, mu1, mu2: np.exp(
            kappa1 * np.cos(theta - mu1) + kappa2 * np.cos(2. * (theta - mu2)))

        u = f(th, k1[0, 0], k2[0, 0], m1[0, 0], m2[0, 0])
        v = f(th, k1[0, 1], k2[0, 1], m1[0, 1], m2[0, 1])

        plt.plot(th, u)
        plt.plot(th, v)
        plt.show()

        print 'OK'

    def test__qmc_z(self):
        k1 = np.array([[100.]])
        k2 = np.array([[500]])
        m1 = np.array([[0.12]]) * 2. * pi
        m2 = np.array([[0.55]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')

        t0 = time.time()
        z_qmc = phi._qmc_z()
        t1 = time.time()
        t2 = time.time()
        z_int = phi._grid_fun(0) * np.exp(k1 + k2)
        t3 = time.time()

        delta_t1 = t1 - t0
        delta_t2 = t3 - t2

        print 'ok'

    def test_benchmark_integration_methods(self):

        delta_t = list()

        k1 = np.array([[500., 10., 20.]])
        k2 = np.array([[500., 20., 2.]])
        m1 = np.array([[0.12, 0., 1.]]) * 2. * pi
        m2 = np.array([[0.55, 0.3, 0.2]]) * pi

        phi = GvM(k1, k2, m1, m2, method='series')

        # Method 0: Serial QMC
        t_ini = time.time()
        z_qmc = np.log(phi._qmc_z())
        t_end = time.time()
        delta_t.append(t_end - t_ini)

        # Method 1: Serial Euler
        t_ini = time.time()
        z_serial_euler = np.log(phi._grid_fun(0).real) + k1 + k2
        t_end = time.time()
        delta_t.append(t_end - t_ini)

        # Method 2: von Mises analytic
        t_ini = time.time()
        z_von_mises = np.log(2 * np.pi * ive(0, k1))
        t_end = time.time()
        delta_t.append(t_end - t_ini)

        # Method 3: Jitted Euler
        n_pts = 10000
        th = np.linspace(-np.pi, np.pi, n_pts)
        z = loopy_log_z_euler(th, k1, k2, m1, m2)
        jitted_log_z = jit(loopy_log_z_euler)
        z = jitted_log_z(th, k1, k2, m1, m2)

        t_ini = time.time()
        z_jit_euler = jitted_log_z(th, k1, k2, m1, m2)
        t_end = time.time()
        delta_t.append(t_end - t_ini)

        # Method 4: Jitted Naive series
        n = 100
        z = loopy_naive_log_z_series(k1, k2, m1, m2, n)
        jitted_log_z_naive_series = jit(loopy_naive_log_z_series)
        z = jitted_log_z_naive_series(k1, k2, m1, m2, n)

        t_ini = time.time()
        z_jit_series = jitted_log_z_naive_series(k1, k2, m1, m2, n)
        t_end = time.time()
        delta_t.append(t_end - t_ini)

        # Method 3: Jitted Trapezoid
        n_pts = 100
        th = np.linspace(-np.pi, np.pi, n_pts)
        z = loopy_log_z_trapz(th, k1, k2, m1, m2)
        jitted_log_z_trapz = jit(loopy_log_z_trapz)
        z = jitted_log_z_trapz(th, k1, k2, m1, m2)

        t_ini = time.time()
        z_jit_trapz = jitted_log_z_trapz(th, k1, k2, m1, m2)
        t_end = time.time()
        delta_t.append(t_end - t_ini)

        # Method 4: AOT Compiled Trapezoid
        n_pts = 100
        th = np.linspace(-np.pi, np.pi, n_pts)

        t_ini = time.time()
        z_aot_trapz = fgvm.get_log_z(th, k1, k2, m1, m2)
        t_end = time.time()
        delta_t.append(t_end - t_ini)

        print '--------- Results -----------------'
        print 'Serial QMC           = ' + str(z_qmc)
        print 'Serial Euler         = ' + str(z_serial_euler)
        print 'vM (only for speed)  = ' + str(z_von_mises)
        print 'Jitted Euler         = ' + str(z_jit_euler)
        print 'Jitted Naive Series  = ' + str(z_jit_series)
        print 'Jitted Trapezoidal   = ' + str(z_jit_trapz)
        print 'AOT-C  Trapezoidal   = ' + str(z_aot_trapz)
        print '--------- Concluded ---------------'

        print '--------- Results -----------------'
        print 'Serial QMC           = ' + str(delta_t[0])
        print 'Serial Euler         = ' + str(delta_t[1])
        print 'vM analytic          = ' + str(delta_t[2])
        print 'Jitted Euler         = ' + str(delta_t[3])
        print 'Jitted Naive Series  = ' + str(delta_t[4])
        print 'Jitted Trapezoidal   = ' + str(delta_t[5])
        print 'AOT-C Trapezoidal    = ' + str(delta_t[6])
        print '--------- Concluded ---------------'
