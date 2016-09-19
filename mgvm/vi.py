"""
    vi.py
    Variational inference for the multivariate Generalised von Mises with a factorised approximation.

    This code is designed for proof of concept only, use at your own risk.
"""
import scipy.special as ss
import scipy.optimize as so
import numpy as np
import utils.linalg as la
from utils.special import gv0


def predictive_dist(n_pred, mf_k1, mf_m1, k1, k2, res=1000, pi_shift=False):

    if pi_shift:
        th = np.linspace(0, 2 * np.pi, res)
    else:
        th = np.linspace(-np.pi, np.pi, res)

    step = th[1] - th[0]

    p = np.zeros([res, n_pred])
    for nn in xrange(0, n_pred):
        for rr in xrange(0, res):
            z1 = mf_k1[nn] * np.exp(1.j * mf_m1[nn]) + k1 * np.exp(1.j * th[rr])
            k1_p = np.abs(z1)
            m1_p = np.angle(z1)
            p[rr, nn] = gv0(m1_p, th[rr], k1_p, k2)

        p[:, nn] = p[:, nn] / (np.sum(p[:, nn]) * step)

    return p, th


def inference_model_obj(xin, config):

    # Load parameters
    n_data = config['N_data']
    n_pred = config['N_pred']
    c_data = config['c_data']
    s_data = config['s_data']
    c_2data = config['c_2data']
    s_2data = config['s_2data']
    mat_kin = config['Kinv']  # not ideal, but this way it's all consistent.
    idx_psi_p = config['idx_psi_p']
    idx_mf_k1 = config['idx_mf_k1']
    idx_mf_m1 = config['idx_mf_m1']
    k1 = config['k1']
    k2 = config['k2']

    n_totp = n_data + n_pred

    # Variable unpacking
    psi_p = xin[idx_psi_p]
    mf_k1 = np.exp(xin[idx_mf_k1])
    mf_m1 = xin[idx_mf_m1]

    # Variable reshaping
    psi_p = psi_p.reshape(n_pred, 1)
    mf_k1 = mf_k1.reshape(n_totp, 1)
    mf_m1 = mf_m1.reshape(n_totp, 1)
    k1 = k1.reshape(1, 1)
    k2 = k2.reshape(1, 1)

    # assemble prediction set statistics
    s_psi_p = np.sin(psi_p)
    c_psi_p = np.cos(psi_p)
    s_2psi_p = np.sin(2. * psi_p)
    c_2psi_p = np.cos(2. * psi_p)

    c_psi = np.vstack((c_psi_p, c_data))
    s_psi = np.vstack((s_psi_p, s_data))
    c_2psi = np.vstack((c_2psi_p, c_2data))
    s_2psi = np.vstack((s_2psi_p, s_2data))

    # Getting mean field moments
    div_ive_1_0 = ss.ive(1, mf_k1) / ss.ive(0, mf_k1)
    div_ive_2_0 = ss.ive(2, mf_k1) / ss.ive(0, mf_k1)

    c_x = div_ive_1_0 * np.cos(mf_m1)
    s_x = div_ive_1_0 * np.sin(mf_m1)
    c_2x = div_ive_2_0 * np.cos(2. * mf_m1)
    s_2x = div_ive_2_0 * np.sin(2. * mf_m1)

    cc_x = 0.5 * (1 + c_2x)
    ss_x = 0.5 * (1 - c_2x)
    cs_x = 0.5 * s_2x

    # Building inverse sections
    mat_kin_cc = mat_kin[0:n_totp, 0:n_totp]
    mat_kin_ss = mat_kin[n_totp:, n_totp:]
    mat_kin_cs = mat_kin[0:n_totp, n_totp:]

    diag_mat_kin_cc = np.diag(mat_kin[0:n_totp, 0:n_totp]).reshape(n_totp, 1)
    diag_mat_kin_ss = np.diag(mat_kin[n_totp:, n_totp:]).reshape(n_totp, 1)
    diag_mat_kin_cs = np.diag(mat_kin[0:n_totp, n_totp:]).reshape(n_totp, 1)

    np.fill_diagonal(mat_kin_cc, 0.)
    np.fill_diagonal(mat_kin_ss, 0.)
    np.fill_diagonal(mat_kin_cs, 0.)

    # Free energy assembly point
    # ----------------------------------------------------

    prior = - (la.qform(c_x, mat_kin_cs, s_x) + np.dot(diag_mat_kin_cs.T, cs_x) +
               0.5 * (la.qform(c_x, mat_kin_cc, c_x) + np.dot(diag_mat_kin_cc.T, cc_x) +
                      la.qform(s_x, mat_kin_ss, s_x) + np.dot(diag_mat_kin_ss.T, ss_x)))

    likelihood = k1 * (np.dot(c_psi.T, c_x) + np.dot(s_psi.T, s_x)) - n_totp * (np.log(ss.ive(0, k1)) + k1) +\
                 k2 * (np.dot(c_2psi.T, cc_x - ss_x) + np.dot(s_2psi.T, 2 * cs_x)) - n_totp * (np.log(ss.ive(0, k2)) + k2)

    entropy = np.sum(np.log(ss.ive(0, mf_k1)) + mf_k1 - mf_k1 * ss.ive(1, mf_k1) / ss.ive(0, mf_k1))

    energy = likelihood + prior + entropy

    return - energy.flatten()


def inference_model_grad(xin, config):

    # Load parameters
    n_data = config['N_data']
    n_pred = config['N_pred']
    c_data = config['c_data']
    s_data = config['s_data']
    c_2data = config['c_2data']
    s_2data = config['s_2data']
    mat_kin = config['Kinv']  # not ideal, but this way it's all consistent.
    idx_psi_p = config['idx_psi_p']
    idx_mf_k1 = config['idx_mf_k1']
    idx_mf_m1 = config['idx_mf_m1']
    k1 = config['k1']
    k2 = config['k2']

    n_totp = n_data + n_pred

    # Variable unpacking
    psi_p = xin[idx_psi_p]
    mf_k1 = np.exp(xin[idx_mf_k1])
    mf_m1 = xin[idx_mf_m1]

    # Variable reshaping
    psi_p = psi_p.reshape(n_pred, 1)
    mf_k1 = mf_k1.reshape(n_totp, 1)
    mf_m1 = mf_m1.reshape(n_totp, 1)
    k1 = k1.reshape(1, 1)
    k2 = k2.reshape(1, 1)

    # assemble prediction set statistics
    s_psi_p = np.sin(psi_p)
    c_psi_p = np.cos(psi_p)
    s_2psi_p = np.sin(2. * psi_p)
    c_2psi_p = np.cos(2. * psi_p)

    c_psi = np.vstack((c_psi_p, c_data))
    s_psi = np.vstack((s_psi_p, s_data))
    c_2psi = np.vstack((c_2psi_p, c_2data))
    s_2psi = np.vstack((s_2psi_p, s_2data))

    # Getting mean field moments
    div_ive_1_0 = ss.ive(1, mf_k1) / ss.ive(0, mf_k1)
    div_ive_2_0 = ss.ive(2, mf_k1) / ss.ive(0, mf_k1)
    div_ive_3_0 = ss.ive(3, mf_k1) / ss.ive(0, mf_k1)

    c_x = div_ive_1_0 * np.cos(mf_m1)
    s_x = div_ive_1_0 * np.sin(mf_m1)
    c_2x = div_ive_2_0 * np.cos(2. * mf_m1)
    s_2x = div_ive_2_0 * np.sin(2. * mf_m1)

    cc_x = 0.5 * (1 + c_2x)
    ss_x = 0.5 * (1 - c_2x)
    cs_x = 0.5 * s_2x

    # Building inverse sections
    mat_kin_cc = mat_kin[0:n_totp, 0:n_totp]
    mat_kin_ss = mat_kin[n_totp:, n_totp:]
    mat_kin_cs = mat_kin[0:n_totp, n_totp:]

    diag_mat_kin_cc = np.diag(mat_kin[0:n_totp, 0:n_totp]).reshape(n_totp, 1)
    diag_mat_kin_ss = np.diag(mat_kin[n_totp:, n_totp:]).reshape(n_totp, 1)
    diag_mat_kin_cs = np.diag(mat_kin[0:n_totp, n_totp:]).reshape(n_totp, 1)

    np.fill_diagonal(mat_kin_cc, 0.)
    np.fill_diagonal(mat_kin_ss, 0.)
    np.fill_diagonal(mat_kin_cs, 0.)

    # Gradient assembly point
    # ----------------------------------------------------
    # Prior
    dpri_psi_p = 0.
    dpri_c_x = - (np.dot(mat_kin_cc, c_x) + np.dot(mat_kin_cs, s_x))
    dpri_s_x = - (np.dot(mat_kin_ss, s_x) + np.dot(mat_kin_cs.T, c_x))
    dpri_cc_x = - 0.5 * diag_mat_kin_cc
    dpri_ss_x = - 0.5 * diag_mat_kin_ss
    dpri_cs_x = - diag_mat_kin_cs

    # Likelihood
    ds_psi_psi_p = + np.cos(psi_p)
    dc_psi_psi_p = - np.sin(psi_p)
    ds_2psi_psi_p = + 2. * np.cos(2. * psi_p)
    dc_2psi_psi_p = - 2. * np.sin(2. * psi_p)

    dlik_s_psi = k1 * s_x[0:n_pred]
    dlik_c_psi = k1 * c_x[0:n_pred]
    dlik_s_2psi = k2 * s_2x[0:n_pred]
    dlik_c_2psi = k2 * c_2x[0:n_pred]

    dlik_psi_p = (dlik_s_psi * ds_psi_psi_p + dlik_c_psi * dc_psi_psi_p +
                  dlik_s_2psi * ds_2psi_psi_p + dlik_c_2psi * dc_2psi_psi_p)

    dlik_c_x = +k1 * c_psi
    dlik_s_x = +k1 * s_psi
    dlik_cc_x = +k2 * c_2psi
    dlik_ss_x = -k2 * c_2psi
    dlik_cs_x = +2.0 * k2 * s_2psi

    # Mean field gradients (along with entropy)
    dc_x_mf_k1 = np.cos(mf_m1) * (0.5 * (1. + div_ive_2_0) - div_ive_1_0 ** 2)
    ds_x_mf_k1 = np.sin(mf_m1) * (0.5 * (1. + div_ive_2_0) - div_ive_1_0 ** 2)
    dcc_x_mf_k1 = +0.5 * np.cos(2. * mf_m1) * (0.5 * (div_ive_1_0 + div_ive_3_0) - div_ive_1_0 * div_ive_2_0)
    dss_x_mf_k1 = -0.5 * np.cos(2. * mf_m1) * (0.5 * (div_ive_1_0 + div_ive_3_0) - div_ive_1_0 * div_ive_2_0)
    dcs_x_mf_k1 = +0.5 * np.sin(2. * mf_m1) * (0.5 * (div_ive_1_0 + div_ive_3_0) - div_ive_1_0 * div_ive_2_0)

    dent_mf_k1 = - 0.5 * mf_k1 * (1. + div_ive_2_0) + mf_k1 * div_ive_1_0 ** 2

    dc_x_mf_m1 = - np.sin(mf_m1) * div_ive_1_0
    ds_x_mf_m1 = + np.cos(mf_m1) * div_ive_1_0
    dcc_x_mf_m1 = - np.sin(2. * mf_m1) * div_ive_2_0
    dss_x_mf_m1 = + np.sin(2. * mf_m1) * div_ive_2_0
    dcs_x_mf_m1 = + np.cos(2. * mf_m1) * div_ive_2_0

    dent_mf_m1 = 0.

    dmf_k1 = (dpri_c_x + dlik_c_x) * dc_x_mf_k1 + (dpri_s_x + dlik_s_x) * ds_x_mf_k1 + \
             (dpri_cc_x + dlik_cc_x) * dcc_x_mf_k1 + (dpri_ss_x + dlik_ss_x) * dss_x_mf_k1 + \
             (dpri_cs_x + dlik_cs_x) * dcs_x_mf_k1 + dent_mf_k1

    dmf_m1 = (dpri_c_x + dlik_c_x) * dc_x_mf_m1 + (dpri_s_x + dlik_s_x) * ds_x_mf_m1 + \
             (dpri_cc_x + dlik_cc_x) * dcc_x_mf_m1 + (dpri_ss_x + dlik_ss_x) * dss_x_mf_m1 + \
             (dpri_cs_x + dlik_cs_x) * dcs_x_mf_m1 + dent_mf_m1

    # Final gradient assembly
    dpsi_p = dlik_psi_p + dpri_psi_p

    # Positivy transformation
    dlog_mf_k1 = dmf_k1 * mf_k1

    grad = np.vstack((dpsi_p, dlog_mf_k1, dmf_m1))

    return - grad.flatten()


def inference_model_opt(x0, config):
    return so.minimize(fun=inference_model_obj, x0=x0, args=config, method='L-BFGS-B', jac=inference_model_grad,
                       options={'maxiter':1000, 'disp':False})