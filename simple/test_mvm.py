from unittest import TestCase
import numpy as np
import mvm


class TestFuncs(TestCase):

    def test_inference_model_1_gradient(self):
        n_data = 10
        n_pred = 5
        n_totp = n_data + n_pred

        data = np.random.rand(10, 1) * 2 * np.pi - np.pi
        mat_ell = np.tril(np.random.rand(n_totp, n_totp))
        mat_kin = np.dot(mat_ell, mat_ell.T)

        psi_p = np.random.rand(n_pred, 1) * 2. * np.pi - np.pi
        mf_k1 = np.log(np.random.rand(n_totp, 1) * 10)
        mf_m1 = np.random.rand(n_totp, 1) * 2 * np.pi - np.pi
        k1 = np.log(np.random.rand(1, 1))

        n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + k1.shape[0]
        idx = np.arange(0, n_var)

        config = {
            'N_data':  n_data,
            'N_pred':  n_pred,
            'c_data': np.cos(data),
            's_data': np.sin(data),
            'Kinv':    mat_kin,
            'idx_psi_p': idx[0:psi_p.shape[0]],
            'idx_mf_k1': idx[psi_p.shape[0]:psi_p.shape[0] + mf_k1.shape[0]],
            'idx_mf_m1': idx[psi_p.shape[0] + mf_k1.shape[0]:psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]],
            'idx_k1': idx[psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]:n_var],
        }

        xin = np.vstack((psi_p, mf_k1, mf_m1, k1))
        grad = mvm.inference_model_grad(xin, config)

        delta = 1E-7
        fd = np.zeros([n_var, 1])
        for ii in xrange(0, n_var):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta
            xp = xin + pert
            xm = xin - pert

            f_p = mvm.inference_model_obj(xp, config)
            f_m = mvm.inference_model_obj(xm, config)

            fd[ii] = (f_p - f_m) / (2. * delta)

        fd = fd.flatten()

        np.testing.assert_array_almost_equal(fd[config['idx_psi_p']], grad[config['idx_psi_p']], decimal=6)
        print 'psi_p ok'

        np.testing.assert_array_almost_equal(fd[config['idx_k1']], grad[config['idx_k1']], decimal=6)
        print 'k1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_m1']], grad[config['idx_mf_m1']], decimal=6)
        print 'mf_m1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_k1']], grad[config['idx_mf_k1']], decimal=6)
        print 'mf_k1 ok'

        np.testing.assert_array_almost_equal(fd, grad, decimal=6)
        print 'all items ok'

        return
