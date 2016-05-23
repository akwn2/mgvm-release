from unittest import TestCase

import numpy as np

from dists import fast_gvm as fgvm


class TestFGvM(TestCase):
    def test_get_log_z(self):
        pass

    def test_get_log_z_lik(self):

        N, D = 5, 3
        k1 = np.random.rand(N, D) * 100.
        k2 = np.random.rand(N, D) * 50.
        cos_m1 = np.random.rand(N, D)
        sin_m1 = np.random.rand(N, D)
        cos_2m2 = np.random.rand(N, D)
        sin_2m2 = np.random.rand(N, D)

        # Calculate gradients
        method = 'euler'
        z_calc = fgvm.get_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, method)
        z_gt = fgvm._log_z_lik_gt(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2)

        # Compare
        np.testing.assert_array_almost_equal(z_calc, z_gt, decimal=6)

    def test_get_grad_log_z_lik(self):
        method = 'euler'
        N, D = 5, 3
        k1 = np.random.rand(N, D) * 100.
        k2 = np.random.rand(N, D) * 50.
        cos_m1 = np.random.rand(N, D)
        sin_m1 = np.random.rand(N, D)
        cos_2m2 = np.random.rand(N, D)
        sin_2m2 = np.random.rand(N, D)

        # Calculate gradients
        grad_k1_calc, grad_k2_calc, grad_cos_m1_calc, grad_sin_m1_calc, grad_cos_2m2_calc, grad_sin_2m2_calc = \
            fgvm.get_grad_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, method)

        # Finite differences
        grad_k1_fd = np.zeros([N, D])
        grad_k2_fd = np.zeros([N, D])
        grad_cos_m1_fd = np.zeros([N, D])
        grad_sin_m1_fd = np.zeros([N, D])
        grad_cos_2m2_fd = np.zeros([N, D])
        grad_sin_2m2_fd = np.zeros([N, D])

        delta = 1.e-7
        for ii in xrange(0, N):
            for jj in xrange(0, D):
                pert = np.zeros([N, D])
                pert[ii, jj] = delta

                f_k1_p = fgvm.get_log_z_lik(k1 + pert, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, method)
                f_k1_m = fgvm.get_log_z_lik(k1 - pert, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, method)

                grad_k1_fd[ii, jj] = (f_k1_p[ii, jj] - f_k1_m[ii, jj]) / (2. * delta)

                f_k2_p = fgvm.get_log_z_lik(k1, k2 + pert, cos_m1, sin_m1, cos_2m2, sin_2m2, method)
                f_k2_m = fgvm.get_log_z_lik(k1, k2 - pert, cos_m1, sin_m1, cos_2m2, sin_2m2, method)

                grad_k2_fd[ii, jj] = (f_k2_p[ii, jj] - f_k2_m[ii, jj]) / (2. * delta)

                f_cos_m1_p = fgvm.get_log_z_lik(k1, k2, cos_m1 + pert, sin_m1, cos_2m2, sin_2m2, method)
                f_cos_m1_m = fgvm.get_log_z_lik(k1, k2, cos_m1 - pert, sin_m1, cos_2m2, sin_2m2, method)

                grad_cos_m1_fd[ii, jj] = (f_cos_m1_p[ii, jj] - f_cos_m1_m[ii, jj]) / (2. * delta)

                f_sin_m1_p = fgvm.get_log_z_lik(k1, k2, cos_m1, sin_m1 + pert, cos_2m2, sin_2m2, method)
                f_sin_m1_m = fgvm.get_log_z_lik(k1, k2, cos_m1, sin_m1 - pert, cos_2m2, sin_2m2, method)

                grad_sin_m1_fd[ii, jj] = (f_sin_m1_p[ii, jj] - f_sin_m1_m[ii, jj]) / (2. * delta)

                f_cos_2m2_p = fgvm.get_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2 + pert, sin_2m2, method)
                f_cos_2m2_m = fgvm.get_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2 - pert, sin_2m2, method)

                grad_cos_2m2_fd[ii, jj] = (f_cos_2m2_p[ii, jj] - f_cos_2m2_m[ii, jj]) / (2. * delta)

                f_sin_2m2_p = fgvm.get_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2 + pert, method)
                f_sin_2m2_m = fgvm.get_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2 - pert, method)

                grad_sin_2m2_fd[ii, jj] = (f_sin_2m2_p[ii, jj] - f_sin_2m2_m[ii, jj]) / (2. * delta)

        # Compare
        np.testing.assert_array_almost_equal(grad_k1_fd, grad_k1_calc, decimal=6)
        np.testing.assert_array_almost_equal(grad_k2_fd, grad_k2_calc, decimal=6)
        np.testing.assert_array_almost_equal(grad_cos_m1_fd, grad_cos_m1_calc, decimal=6)
        np.testing.assert_array_almost_equal(grad_sin_m1_fd, grad_sin_m1_calc, decimal=6)
        np.testing.assert_array_almost_equal(grad_cos_2m2_fd, grad_cos_2m2_calc, decimal=6)
        np.testing.assert_array_almost_equal(grad_sin_2m2_fd, grad_sin_2m2_calc, decimal=6)

    def test_get_trig_mom(self):
        pass

    def test_get_grad_u_trig_m(self):
        pass
