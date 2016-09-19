"""
gp.py
implementation of GP inference and learning for exact inference models
"""
import scipy.optimize as so
import numpy as np
import utils.kernels as kernels
import utils.linalg as la


def predict_se_iso(y, xi, xp, params, noise):

    # Covariance bits
    mat_k_xi_xi = kernels.se_iso(xi, xi, params)
    mat_k_xi_xp = kernels.se_iso(xi, xp, params)
    mat_k_xp_xp = kernels.se_iso(xp, xp, params)

    # Noise addition
    mat_k_xi_xi += noise * np.eye(mat_k_xi_xi.shape[0])

    # Expected values
    mat_ell, jitter = la.rchol(mat_k_xi_xi)
    a = la.inv_mult_chol(mat_ell, y)
    yp = np.dot(mat_k_xi_xp.T, a)

    # Variances
    v = la.solve_chol(mat_ell, mat_k_xi_xp)
    var_yp = mat_k_xp_xp - np.dot(v.T, v)

    return yp, var_yp


def predict_pe_iso(y, xi, xp, params, noise):

    # Covariance bits
    mat_k_xi_xi = kernels.pe_iso(xi, xi, params)
    mat_k_xi_xp = kernels.pe_iso(xi, xp, params)
    mat_k_xp_xp = kernels.pe_iso(xp, xp, params)

    # Noise addition
    mat_k_xi_xi += noise * np.eye(mat_k_xi_xi.shape[0])

    # Expected values
    mat_ell, jitter = la.rchol(mat_k_xi_xi)
    a = la.inv_mult_chol(mat_ell, y)
    yp = np.dot(mat_k_xi_xp.T, a)

    # Variances
    v = la.solve_chol(mat_ell, mat_k_xi_xp)
    var_yp = mat_k_xp_xp - np.dot(v.T, v)

    return yp, var_yp


def se_iso_model_obj(hyp, config):
    xi = config['xi']
    y = config['y']

    ell2 = np.exp(hyp[0])
    s2 = np.exp(hyp[1])
    noise = np.exp(hyp[2])

    params = {
        's2': s2,
        'ell2': ell2
    }
    mat_k = kernels.se_iso(xi, xi, params)
    mat_k += noise * np.eye(mat_k.shape[0])

    mat_ell, jitter = la.rchol(mat_k)
    a = la.inv_mult_chol(mat_ell, y)

    # Compute objective
    obj = -0.5 * np.sum(np.dot(y.T, a)) - np.sum(np.log(np.diag(mat_ell))) - 0.5 * y.shape[0] * np.log(2. * np.pi)

    return - obj.flatten()


def se_iso_model_grad(hyp, config):

    xi = config['xi']
    y = config['y']

    ell2 = np.exp(hyp[0])
    s2 = np.exp(hyp[1])
    noise = np.exp(hyp[2])

    params = {
        's2': s2,
        'ell2': ell2
    }
    mat_k, grad = kernels.se_iso(xi, xi, params, get_gradients=True)
    mat_k += noise * np.eye(mat_k.shape[0])
    mat_ell, jitter = la.rchol(mat_k)

    a = la.inv_mult_chol(mat_ell, y)
    aa_t = np.dot(a, a.T)

    # Compute gradients
    dell2 = 0.5 * np.trace(np.dot(aa_t, grad['ell2']) -
                           la.inv_mult_chol(mat_ell, grad['ell2']))
    ds2 = 0.5 * np.trace(np.dot(aa_t, grad['s2']) -
                         la.inv_mult_chol(mat_ell, grad['s2']))
    dnoise = 0.5 * np.trace(np.dot(aa_t, np.eye(mat_ell.shape[0])) -
                            la.inv_mult_chol(mat_ell, np.eye(mat_ell.shape[0])))
    # Pack gradients
    grad = np.zeros([3, 1])
    grad[0] = dell2 * ell2
    grad[1] = ds2 * s2
    grad[2] = dnoise * noise

    return - grad.flatten()


def learning_se_iso(hyp0, config):
    return so.minimize(fun=se_iso_model_obj, x0=hyp0, args=config, method='L-BFGS-B', jac=se_iso_model_grad)


def pe_iso_model_obj(hyp, config):
    xi = config['xi']
    y = config['y']

    ell2 = np.exp(hyp[0])
    s2 = np.exp(hyp[1])
    per = np.exp(hyp[2])
    noise = np.exp(hyp[3])

    params = {
        's2': s2,
        'ell2': ell2,
        'per': per
    }
    mat_k = kernels.pe_iso(xi, xi, params)
    mat_k += noise * np.eye(mat_k.shape[0])

    mat_ell, jitter = la.rchol(mat_k)
    a = la.inv_mult_chol(mat_ell, y)

    # Compute objective
    obj = -0.5 * np.sum(np.dot(y.T, a)) - np.sum(np.log(np.diag(mat_ell))) - 0.5 * y.shape[0] * np.log(2. * np.pi)

    return - obj.flatten()


def pe_iso_model_grad(hyp, config):

    xi = config['xi']
    y = config['y']

    ell2 = np.exp(hyp[0])
    s2 = np.exp(hyp[1])
    per = np.exp(hyp[2])
    noise = np.exp(hyp[3])

    params = {
        's2': s2,
        'ell2': ell2,
        'per': per
    }
    mat_k, grad = kernels.pe_iso(xi, xi, params, get_gradients=True)
    mat_k += noise * np.eye(mat_k.shape[0])
    mat_ell, jitter = la.rchol(mat_k)

    a = la.inv_mult_chol(mat_ell, y)
    aa_t = np.dot(a, a.T)

    # Compute gradients
    dell2 = 0.5 * np.trace(np.dot(aa_t, grad['ell2']) -
                           la.inv_mult_chol(mat_ell, grad['ell2']))

    ds2 = 0.5 * np.trace(np.dot(aa_t, grad['s2']) -
                         la.inv_mult_chol(mat_ell, grad['s2']))

    dper = 0.5 * np.trace(np.dot(aa_t, grad['per']) -
                         la.inv_mult_chol(mat_ell, grad['per']))

    dnoise = 0.5 * np.trace(np.dot(aa_t, np.eye(mat_ell.shape[0])) -
                            la.inv_mult_chol(mat_ell, np.eye(mat_ell.shape[0])))
    # Pack gradients
    grad = np.zeros([4, 1])
    grad[0] = dell2 * ell2
    grad[1] = ds2 * s2
    grad[2] = dper * per
    grad[3] = dnoise * noise

    return - grad.flatten()


def learning_pe_iso(hyp0, config):
    return so.minimize(fun=pe_iso_model_obj, x0=hyp0, args=config, method='L-BFGS-B', jac=pe_iso_model_grad)


def get_predictive_for_plots_1d(yp, var_yp, res=1000):
    p = np.zeros([res, yp.shape[0]])
    th = np.linspace(-np.pi, np.pi, res)
    s2 = np.diag(var_yp)
    for ii in xrange(0, yp.shape[0]):
        mu = yp[ii]
        two_s2 = 2 * s2[ii]
        p[:, ii] = np.exp(-(th - mu) ** 2 / two_s2) / np.sqrt(np.pi * two_s2)
    return p, th


def get_predictive_for_wrapped(yp, var_yp, res=1000):
    p = np.zeros([res, yp.shape[0]])
    th = np.linspace(0, 2 * np.pi, res)
    s2 = np.diag(var_yp)
    for ii in xrange(0, yp.shape[0]):
        mu = yp[ii]
        two_s2 = 2 * s2[ii]
        p[:, ii] = np.exp(- (np.mod(th - mu, 2 * np.pi) - np.pi) ** 2 / two_s2) / np.sqrt(np.pi * two_s2)
    return p, th


def get_predictive_for_plots_2d(yp, var_yp, res=1000):
    p_re = np.zeros([res, yp.shape[0]])
    p_im = np.zeros([res, yp.shape[0]])
    th = np.linspace(-np.pi, np.pi, res)
    s2 = np.diag(var_yp)
    for ii in xrange(0, yp.shape[0]):
        two_s2 = 2 * s2[ii]
        mu_re = yp[ii, 0]
        mu_im = yp[ii, 1]
        p_re[:, ii] = np.exp(-(np.cos(th) - mu_re) ** 2 / two_s2) / np.sqrt(np.pi * two_s2)
        p_im[:, ii] = np.exp(-(np.sin(th) - mu_im) ** 2 / two_s2) / np.sqrt(np.pi * two_s2)

    p = np.abs(p_re + 1.j * p_im)
    return p, th