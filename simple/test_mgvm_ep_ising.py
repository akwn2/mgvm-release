from unittest import TestCase
import numpy as np
import scipy.special as ss
import scipy.linalg as la
import mgvm_ep_ising as mei
import utils.kernels as kernels
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


class TestMgvmEpIsing(TestCase):
    def test_grad_inv_ive(self):
        n_var = 10
        kappa = np.random.rand(n_var, 1) * 10
        nu = 0

        x = np.log(ss.ive(nu, kappa)) + kappa
        z = np.random.rand(n_var, 1) * 10

        grad = mei.grad_inv_ive(z, nu, x)

        fd = np.zeros([n_var, 1])
        delta = 1.E-8
        for ii in xrange(0, 10):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta

            fp = mei.obj_inv_ive(z + pert, nu, x)
            fm = mei.obj_inv_ive(z - pert, nu, x)

            fd[ii] = (fp[ii] - fm[ii]) / (2. * delta)

        np.testing.assert_almost_equal(fd, grad, decimal=6)

    def test_inv_log_ive(self):
        kappa = np.random.rand(10, 1) * 10
        nu = 0

        x = np.log(ss.ive(nu, kappa)) + kappa

        kcalc = mei.inv_log_iv(nu=nu, x=x, z0=np.log(x))
        np.testing.assert_almost_equal(kappa, kcalc, decimal=6)

    def test_moments_to_vm(self):
        n_var = 10
        k1_gt = np.random.rand(n_var, 1) * 100
        m1_gt = np.random.rand(n_var, 1) * 2 * np.pi - np.pi

        log_mom_0 = np.log(2. * np.pi * ss.iv(0, k1_gt))
        mom_1 = ss.ive(0, k1_gt) * np.exp(1.j * m1_gt)

        m1_calc, k1_calc = mei.moments_to_vm(log_mom_0=log_mom_0, mom_1=mom_1)

        np.testing.assert_almost_equal(k1_gt, k1_calc, decimal=6)
        print 'k1 ok'

        np.testing.assert_almost_equal(m1_gt, m1_calc, decimal=6)
        print 'm1 ok'

    def test_get_moments_gvm(self):
        n_var = 10
        k1_gt = np.random.rand(n_var, 1) * 100
        m1_gt = np.random.rand(n_var, 1) * 2 * np.pi - np.pi
        k2_gt = np.zeros_like(k1_gt)
        m2_gt = np.ones_like(m1_gt) * 0.5 * np.pi

        log_mom_0_gt = np.log(2. * np.pi * ss.iv(0, k1_gt))
        mom_1_gt = ss.ive(1, k1_gt) * np.exp(1.j * m1_gt) / ss.ive(0, k1_gt)

        log_mom_0, mom_1 = mei.gvm_moments(m1=m1_gt, m2=m2_gt, k1=k1_gt, k2=k2_gt)

        np.testing.assert_almost_equal(log_mom_0_gt, log_mom_0, decimal=6)
        print 'log_mom_0 ok'

        np.testing.assert_almost_equal(mom_1_gt, mom_1, decimal=6)
        print 'mom1 ok'

    def test_run_ep(self):

        x_t = np.array([+0.50, +1.00])
        y_t = np.array([+2.50, -1.25])
        n_data = np.alen(x_t)

        print('--> Calcualte kernel')
        # Set kernel parameters
        params = {
            's2': 200.00,
            'ell2': 5.0E-2 ** 2,
        }

        y_t = y_t.reshape(n_data, 1)
        x_t = x_t.reshape(n_data, 1)

        # Calculate kernels
        mat_k_cc = kernels.se_iso(x_t, x_t, params)
        mat_k_ss = kernels.se_iso(x_t, x_t, params)

        mat_k = np.bmat([[mat_k_cc, np.zeros_like(mat_k_cc)], [np.zeros_like(mat_k_ss), mat_k_ss]])
        mat_k = np.asarray(mat_k)
        mat_k += 1E-3 * np.eye(mat_k.shape[0])

        # Find inverse
        mat_ell = la.cholesky(mat_k, lower=True)
        mat_kin = la.solve(mat_ell.T, la.solve(mat_ell, np.eye(mat_ell.shape[0])))

        print('--> Initialising model variables')

        config = {
            'N_data': n_data,
            'c_data': np.cos(y_t),
            's_data': np.sin(y_t),
            'Kinv': mat_kin,
            'damping': 1E-2,
            'max_sweeps': 500,
            'tol': 1E-4,
            'k1': 10.,
        }

        results = mei.run_ep(config)
        k1_f = results[0]
        m1_f = results[1]
        res = results[2]
        swept = results[3]
        converged = results[4]

        np.testing.assert_(converged)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(-np.pi, np.pi, 0.1)
        Y = np.arange(-np.pi, np.pi, 0.1)
        X, Y = np.meshgrid(X, Y)
        F = np.exp(- 0.5 * (mat_kin[0, 0] * np.cos(X) ** 2 + mat_kin[1, 1] * np.cos(Y) ** 2 +
                            mat_kin[2, 2] * np.sin(X) ** 2 + mat_kin[3, 3] * np.sin(Y) ** 2 +
                            2 * mat_kin[0, 1] * np.cos(X) + np.cos(Y) +
                            2 * mat_kin[2, 3] * np.sin(X) + np.sin(Y) +
                            2 * (mat_kin[0, 2] * np.cos(X) * np.sin(X) + mat_kin[1, 3] * np.cos(Y) * np.sin(Y) +
                                 mat_kin[0, 3] * np.cos(X) + np.sin(Y) + mat_kin[1, 2] * np.cos(Y) + np.sin(X))
                            )
                   - np.sum(mat_kin))

        Z = np.sum(F)
        surf = ax.plot_surface(X, Y, F / Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        th = np.arange(-np.pi, np.pi, 0.001)
        p0 = np.exp(k1_f[0, 0] * (np.cos(th - m1_f[0, 0]) - 1)) / (2 * np.pi * ss.ive(0, k1_f[0, 0]))
        p1 = np.exp(k1_f[0, 1] * (np.cos(th - m1_f[0, 1]) - 1)) / (2 * np.pi * ss.ive(0, k1_f[0, 1]))

        # Let's just rescale all to match the bgvm
        max_bgvm = np.max((F / Z).flatten())
        p0 *= max_bgvm / np.max(p0)
        p1 *= max_bgvm / np.max(p1)

        ax.plot(th, -np.pi * np.ones_like(th), p0, '-b')
        ax.plot(- np.pi * np.ones_like(th), th, p1, '-r')

        plt.show()
        return results