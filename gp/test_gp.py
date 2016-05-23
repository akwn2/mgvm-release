from unittest import TestCase
import numpy as np
import gp


class TestGP(TestCase):

    def test_se_iso_model_grad(self):
        n_pts = 10
        xi = np.random.rand(n_pts, 2)
        y = np.random.rand(n_pts, 1)

        config = {
            'xi': xi,
            'y': y,
        }

        n_vars = 3
        ell2 = 0.5 ** 2
        s2 = 400.
        noise = 10.0

        x = np.zeros([n_vars, 1])
        x[0] = ell2
        x[1] = s2
        x[2] = noise
        x = np.log(x)

        grad = gp.se_iso_model_grad(x, config)
        fd = np.zeros([n_vars, 1])
        delta = 1.E-7
        for ii in xrange(0, n_vars):
            pert = np.zeros(x.shape)
            pert[ii] = delta

            f_p = gp.se_iso_model_obj(x + pert, config)
            f_m = gp.se_iso_model_obj(x - pert, config)

            fd[ii] = (f_p - f_m) / (2. * delta)

        np.testing.assert_almost_equal(fd, grad, decimal=6)
