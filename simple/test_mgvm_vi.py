from unittest import TestCase
import numpy as np
import mgvm_vi
import scipy.special as ss


class TestMgvmVi(TestCase):

    def test_inference_model_1_gradient(self):
        n_data = 10
        n_pred = 5
        n_totp = n_data + n_pred

        data = np.random.rand(10, 1) * 2 * np.pi - np.pi
        mat_ell = np.tril(np.random.rand(2 * n_totp, 2 * n_totp))
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
        grad = mgvm_vi.inference_model_1_grad(xin, config)

        delta = 1E-7
        fd = np.zeros([n_var, 1])
        for ii in xrange(0, n_var):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta
            xp = xin + pert
            xm = xin - pert

            f_p = mgvm_vi.inference_model_1_obj(xp, config)
            f_m = mgvm_vi.inference_model_1_obj(xm, config)

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

    def test_inference_model_2_gradient(self):
        n_data = 10
        n_pred = 5
        n_totp = n_data + n_pred

        data = np.random.rand(10, 1) * 2 * np.pi - np.pi
        mat_ell = np.tril(np.random.rand(2 * n_totp, 2 * n_totp))
        mat_kin = np.dot(mat_ell, mat_ell.T)

        two_psi_p = np.random.rand(n_pred, 1) * 2. * np.pi - np.pi
        mf_k1 = np.log(np.random.rand(n_totp, 1) * 10)
        mf_m1 = np.random.rand(n_totp, 1) * 2. * np.pi - np.pi
        k2 = np.log(np.random.rand(1, 1))

        n_var = two_psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + k2.shape[0]
        idx = np.arange(0, n_var)

        config = {
            'N_data': n_data,
            'N_pred': n_pred,
            'c_2data': np.cos(2 * data),
            's_2data': np.sin(2 * data),
            'Kinv': mat_kin,
            'idx_2psi_p': idx[0:two_psi_p.shape[0]],
            'idx_mf_k1': idx[two_psi_p.shape[0]:two_psi_p.shape[0] + mf_k1.shape[0]],
            'idx_mf_m1': idx[two_psi_p.shape[0] + mf_k1.shape[0]:two_psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]],
            'idx_k2': idx[two_psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]:n_var],
        }

        xin = np.vstack((two_psi_p, mf_k1, mf_m1, k2))
        grad = mgvm_vi.inference_model_2_grad(xin, config)

        delta = 1E-7
        fd = np.zeros([n_var, 1])
        for ii in xrange(0, n_var):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta
            xp = xin + pert
            xm = xin - pert

            f_p = mgvm_vi.inference_model_2_obj(xp, config)
            f_m = mgvm_vi.inference_model_2_obj(xm, config)

            fd[ii] = (f_p - f_m) / (2. * delta)

        fd = fd.flatten()

        np.testing.assert_array_almost_equal(fd[config['idx_2psi_p']], grad[config['idx_2psi_p']], decimal=6)
        print 'two_psi_p ok'

        np.testing.assert_array_almost_equal(fd[config['idx_k2']], grad[config['idx_k2']], decimal=6)
        print 'k2 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_k1']], grad[config['idx_mf_k1']], decimal=6)
        print 'mf_k1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_m1']], grad[config['idx_mf_m1']], decimal=6)
        print 'mf_m1 ok'

        np.testing.assert_array_almost_equal(fd, grad, decimal=6)
        print 'all items ok'

        return

    def test_inference_model_3_gradient(self):
        n_data = 10
        n_pred = 5
        n_totp = n_data + n_pred

        data = np.random.rand(10, 1) * 2 * np.pi - np.pi
        mat_ell = np.tril(np.random.rand(2 * n_totp, 2 * n_totp))
        mat_kin = np.dot(mat_ell, mat_ell.T)

        psi_p = np.random.rand(n_pred, 1) * 2. * np.pi - np.pi
        mf_k1 = np.log(np.random.rand(n_totp, 1) * 10)
        mf_m1 = np.random.rand(n_totp, 1) * 2. * np.pi - np.pi
        k1 = np.log(np.random.rand(1, 1))
        k2 = np.log(np.random.rand(1, 1))

        n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + k1.shape[0] + k2.shape[0]
        idx = np.arange(0, n_var)

        config = {
            'N_data': n_data,
            'N_pred': n_pred,
            'c_data': np.cos(data),
            's_data': np.sin(data),
            'c_2data': np.cos(2 * data),
            's_2data': np.sin(2 * data),
            'Kinv': mat_kin,
            'idx_psi_p': idx[0:psi_p.shape[0]],
            'idx_mf_k1': idx[psi_p.shape[0]:psi_p.shape[0] + mf_k1.shape[0]],
            'idx_mf_m1': idx[psi_p.shape[0] + mf_k1.shape[0]:psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]],
            'idx_k1': idx[-2],
            'idx_k2': idx[-1],
        }

        xin = np.vstack((psi_p, mf_k1, mf_m1, k1, k2))
        grad = mgvm_vi.inference_model_3_grad(xin, config)

        delta = 1E-7
        fd = np.zeros([n_var, 1])
        for ii in xrange(0, n_var):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta
            xp = xin + pert
            xm = xin - pert

            f_p = mgvm_vi.inference_model_3_obj(xp, config)
            f_m = mgvm_vi.inference_model_3_obj(xm, config)

            fd[ii] = (f_p - f_m) / (2. * delta)

        fd = fd.flatten()

        np.testing.assert_array_almost_equal(fd[config['idx_psi_p']], grad[config['idx_psi_p']], decimal=6)
        print 'psi_p ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_k1']], grad[config['idx_mf_k1']], decimal=6)
        print 'mf_k1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_m1']], grad[config['idx_mf_m1']], decimal=6)
        print 'mf_m1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_k1']], grad[config['idx_k1']], decimal=6)
        print 'k1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_k2']], grad[config['idx_k2']], decimal=6)
        print 'k2 ok'

        np.testing.assert_array_almost_equal(fd, grad, decimal=6)
        print 'all items ok'

        return

    def test_inference_model_4_gradient(self):
        n_data = 10
        n_pred = 5
        n_totp = n_data + n_pred

        data = np.random.rand(10, 1) * 2 * np.pi - np.pi
        mat_ell = np.tril(np.random.rand(2 * n_totp, 2 * n_totp))
        mat_kin = np.dot(mat_ell, mat_ell.T)

        psi_p = np.random.rand(n_pred, 1) * 2. * np.pi - np.pi
        mf_k1 = np.log(np.random.rand(n_totp, 1) * 10)
        mf_m1 = np.random.rand(n_totp, 1) * 2. * np.pi - np.pi
        k1 = np.random.rand(1, 1)
        k2 = np.random.rand(1, 1)

        n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]
        idx = np.arange(0, n_var)

        config = {
            'N_data': n_data,
            'N_pred': n_pred,
            'c_data': np.cos(data),
            's_data': np.sin(data),
            'c_2data': np.cos(2 * data),
            's_2data': np.sin(2 * data),
            'Kinv': mat_kin,
            'idx_psi_p': idx[0:psi_p.shape[0]],
            'idx_mf_k1': idx[psi_p.shape[0]:psi_p.shape[0] + mf_k1.shape[0]],
            'idx_mf_m1': idx[psi_p.shape[0] + mf_k1.shape[0]:psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]],
            'k1': k1,
            'k2': k2,
        }

        xin = np.vstack((psi_p, mf_k1, mf_m1))
        grad = mgvm_vi.inference_model_4_grad(xin, config)

        delta = 1E-7
        fd = np.zeros([n_var, 1])
        for ii in xrange(0, n_var):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta
            xp = xin + pert
            xm = xin - pert

            f_p = mgvm_vi.inference_model_4_obj(xp, config)
            f_m = mgvm_vi.inference_model_4_obj(xm, config)

            fd[ii] = (f_p - f_m) / (2. * delta)

        fd = fd.flatten()

        np.testing.assert_array_almost_equal(fd[config['idx_psi_p']], grad[config['idx_psi_p']], decimal=6)
        print 'psi_p ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_k1']], grad[config['idx_mf_k1']], decimal=6)
        print 'mf_k1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_m1']], grad[config['idx_mf_m1']], decimal=6)
        print 'mf_m1 ok'

        np.testing.assert_array_almost_equal(fd, grad, decimal=6)
        print 'all items ok'

        return

    def test_inference_model_5_gradient(self):
        n_data = 10
        n_pred = 5
        n_totp = n_data + n_pred

        data = np.random.rand(10, 1) * 2 * np.pi - np.pi
        mat_ell = np.tril(np.random.rand(2 * n_totp, 2 * n_totp))
        mat_kin = np.dot(mat_ell, mat_ell.T)

        psi_p = np.random.rand(n_pred, 1) * 2. * np.pi - np.pi
        mf_k1 = np.log(np.random.rand(n_totp, 1) * 10)
        mf_m1 = np.random.rand(n_totp, 1) * 10
        mf_k2 = np.log(np.random.rand(n_totp, 1) * 10)
        mf_m2 = np.random.rand(n_totp, 1) * 10
        k1 = np.random.rand(1, 1)
        k2 = np.random.rand(1, 1)

        n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + mf_k2.shape[0] + mf_m2.shape[0] + k1.shape[0] + k2.shape[0]
        idx = np.arange(0, n_var)

        config = {
            'N_data': n_data,
            'N_pred': n_pred,
            'c_data': np.cos(data),
            's_data': np.sin(data),
            'c_2data': np.cos(2 * data),
            's_2data': np.sin(2 * data),
            'Kinv': mat_kin,
            'idx_psi_p': idx[0:psi_p.shape[0]],
            'idx_mf_k1': idx[psi_p.shape[0]:
                             psi_p.shape[0] + mf_k1.shape[0]],
            'idx_mf_m1': idx[psi_p.shape[0] + mf_k1.shape[0]:
                             psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]],
            'idx_mf_k2': idx[psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]:
                             psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + mf_k2.shape[0]],
            'idx_mf_m2': idx[psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + mf_k2.shape[0]:
                             psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + mf_k2.shape[0] + mf_m2.shape[0]],
            'idx_k1': idx[-2],
            'idx_k2': idx[-1],
        }

        xin = np.vstack((psi_p, mf_k1, mf_m1, mf_k2, mf_m2, k1, k2))
        grad = mgvm_vi.inference_model_5_grad(xin, config)

        delta = 1E-7
        fd = np.zeros([n_var, 1])
        for ii in xrange(0, n_var):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta
            xp = xin + pert
            xm = xin - pert

            f_p = mgvm_vi.inference_model_5_obj(xp, config)
            f_m = mgvm_vi.inference_model_5_obj(xm, config)

            fd[ii] = (f_p - f_m) / (2. * delta)

        fd = fd.flatten()

        np.testing.assert_array_almost_equal(fd[config['idx_psi_p']], grad[config['idx_psi_p']], decimal=6)
        print 'psi_p ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_k1']], grad[config['idx_mf_k1']], decimal=6)
        print 'mf_k1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_m1']], grad[config['idx_mf_m1']], decimal=6)
        print 'mf_m1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_k2']], grad[config['idx_mf_k2']], decimal=6)
        print 'mf_k2 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_mf_m2']], grad[config['idx_mf_m2']], decimal=6)
        print 'mf_m2 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_k1']], grad[config['idx_k1']], decimal=6)
        print 'k1 ok'

        np.testing.assert_array_almost_equal(fd[config['idx_k2']], grad[config['idx_k2']], decimal=6)
        print 'k2 ok'

        np.testing.assert_array_almost_equal(fd, grad, decimal=6)
        print 'all items ok'

        return

    def test_gve(self):
        n_var = 1000
        k1 = np.linspace(0, n_var, n_var, False).reshape(n_var, 1)
        k2 = np.zeros_like(k1)
        m1 = np.random.rand(n_var, 1) * 2 * np.pi - np.pi
        m2 = np.random.rand(n_var, 1) * 2 * np.pi - np.pi

        gt = ss.ive(0, k1) * 2 * np.pi # ground truth value when distribution is a von Mises
        calc = mgvm_vi.gve(0, m1, m2, k1, k2)

        np.testing.assert_almost_equal(gt, calc, decimal=6)
        print 'Passed: Matched von Mises normalising constant'

    def test_grad_gve(self):
        n_var = 1000
        k1 = np.linspace(0, n_var, n_var, False).reshape(n_var, 1)
        k2 = np.zeros_like(k1)
        m1 = np.random.rand(n_var, 1) * 2 * np.pi - np.pi
        m2 = np.random.rand(n_var, 1) * 2 * np.pi - np.pi

        dk1_gt = ss.ive(1, k1) * 2 * np.pi # ground truth value when distribution is a von Mises
        dm1_gt = np.zeros_like(m1)
        dm2_gt = np.zeros_like(m2)
        dm1, dm2, dk1, dk2 = mgvm_vi.grad_gve(0, m1, m2, k1, k2)

        np.testing.assert_almost_equal(dm1_gt, dm1, decimal=6)
        print 'dm1 analytic ok'

        np.testing.assert_almost_equal(dm2_gt, dm2, decimal=6)
        print 'dm2 analytic ok'

        np.testing.assert_almost_equal(dk1_gt, dk1, decimal=6)
        print 'dk1 analytic ok'

        # dk2 Cannot be analytically tested
        #
        # We now run to finite differences

        delta = 1E-7
        dm1_fd = np.zeros([n_var, 1])
        dm2_fd = np.zeros([n_var, 1])
        dk1_fd = np.zeros([n_var, 1])
        dk2_fd = np.zeros([n_var, 1])
        for ii in xrange(0, n_var):
            pert = np.zeros([n_var, 1])
            pert[ii] = delta

            f_p = mgvm_vi.gve(0, m1 + pert, m2, k1, k2)
            f_m = mgvm_vi.gve(0, m1 - pert, m2, k1, k2)
            dm1_fd[ii] = (f_p[ii] - f_m[ii]) / (2. * delta)

            f_p = mgvm_vi.gve(0, m1, m2 + pert, k1, k2)
            f_m = mgvm_vi.gve(0, m1, m2 - pert, k1, k2)
            dm2_fd[ii] = (f_p[ii] - f_m[ii]) / (2. * delta)

            f_p = mgvm_vi.gve(0, m1, m2, k1 + pert, k2)
            f_m = mgvm_vi.gve(0, m1, m2, k1 - pert, k2)
            dk1_fd[ii] = (f_p[ii] - f_m[ii]) / (2. * delta)

            f_p = mgvm_vi.gve(0, m1, m2, k1, k2 + pert)
            f_m = mgvm_vi.gve(0, m1, m2, k1, k2 - pert)
            dk2_fd[ii] = (f_p[ii] - f_m[ii]) / (2. * delta)

        dm1_fd = dm1_fd.flatten()
        dm2_fd = dm2_fd.flatten()
        dk1_fd = dk1_fd.flatten()
        dk2_fd = dk2_fd.flatten()

        np.testing.assert_almost_equal(dm1_fd, dm1, decimal=6)
        print 'dm1 analytic ok'

        np.testing.assert_almost_equal(dm2_fd, dm2, decimal=6)
        print 'dm2 analytic ok'

        np.testing.assert_almost_equal(dk1_fd, dk1, decimal=6)
        print 'dk1 analytic ok'

        np.testing.assert_almost_equal(dk2_fd, dk2, decimal=6)
        print 'dk2 analytic ok'
