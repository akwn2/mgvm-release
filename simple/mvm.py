"""
funcs
functions used in inference with the models
    -> model 1: posterior \propto von Mises * mvM prior
"""
import scipy.special as ss
# import scipy.linalg as la
import scipy.optimize as so
import numpy as np
import numba


# Auxiliary functions


def qform(x, a, y):
    return np.dot(x.T, np.dot(a, y))


def predictive_dist(n_pred, mf_k1, mf_m1, k1, res=1000):

    th = np.linspace(-np.pi, np.pi, res)
    step = th[1] - th[0]

    mf_re1 = mf_k1 * np.cos(mf_m1)
    mf_im1 = mf_m1 * np.sin(mf_m1)

    p = np.zeros([res, n_pred])
    mode = np.zeros([n_pred, 1])
    log_p = np.zeros([res, 1])
    kappa = np.zeros([res, 1])
    for nn in xrange(0, n_pred):
        for rr in xrange(0, res):

            z = mf_re1[nn] + 1.j * mf_im1[nn] + k1 * (np.cos(th[rr]) + 1.j * np.sin(th[rr]))
            kappa[rr] = np.abs(z)
            log_p[rr] = np.log(2. * np.pi * ss.ive(0, kappa[rr]))

        max_kappa = np.max(kappa)
        log_p += kappa - max_kappa
        log_z = np.log(np.sum(np.exp(log_p + np.log(step))))
        p[:, nn] = np.exp(log_p - log_z).flatten()
        # get the mode
        mode_pval = np.max(p[:, nn])
        mode[nn] = np.max(th[p[:, nn] == mode_pval])

    return p, th, mode


# Model 1


def inference_model_obj(xin, config):

    # Load parameters
    n_data = config['N_data']
    n_pred = config['N_pred']
    c_data = config['c_data']
    s_data = config['s_data']
    mat_kin_ss = config['Kinv']  # not ideal, but this way it's all consistent.
    idx_psi_p = config['idx_psi_p']
    idx_mf_k1 = config['idx_mf_k1']
    idx_mf_m1 = config['idx_mf_m1']
    idx_k1 = config['idx_k1']

    n_totp = n_data + n_pred

    # Variable unpacking
    psi_p = xin[idx_psi_p]
    mf_k1 = np.exp(xin[idx_mf_k1])
    mf_m1 = xin[idx_mf_m1]
    k1 = np.exp(xin[idx_k1])

    # Variable reshaping
    psi_p = psi_p.reshape(n_pred, 1)
    mf_k1 = mf_k1.reshape(n_totp, 1)
    mf_m1 = mf_m1.reshape(n_totp, 1)
    k1 = k1.reshape(1, 1)

    # assemble prediction set statistics
    s_psi_p = np.sin(psi_p)
    c_psi_p = np.cos(psi_p)

    c_psi = np.vstack((c_psi_p, c_data))
    s_psi = np.vstack((s_psi_p, s_data))

    # Getting mean field moments
    div_ive_1_0 = ss.ive(1, mf_k1) / ss.ive(0, mf_k1)

    c_x = div_ive_1_0 * np.cos(mf_m1)
    s_x = div_ive_1_0 * np.sin(mf_m1)

    # Building inverse sections

    # Free energy assembly point
    # ----------------------------------------------------

    prior = - 0.5 * qform(s_x, mat_kin_ss, s_x)

    likelihood = k1 * (np.dot(c_psi.T, c_x) + np.dot(s_psi.T, s_x)) - n_totp * (np.log(ss.ive(0, k1)) + k1)

    entropy = np.sum(np.log(ss.ive(0, mf_k1)) + mf_k1 - mf_k1 * ss.ive(1, mf_k1) / ss.ive(0, mf_k1))

    energy = likelihood + prior + entropy

    return - energy.flatten() / n_totp


def inference_model_grad(xin, config):

    # Load parameters
    n_data = config['N_data']
    n_pred = config['N_pred']
    c_data = config['c_data']
    s_data = config['s_data']
    mat_kin_ss = config['Kinv']  # not ideal, but this way it's all consistent.
    idx_psi_p = config['idx_psi_p']
    idx_mf_k1 = config['idx_mf_k1']
    idx_mf_m1 = config['idx_mf_m1']
    idx_k1 = config['idx_k1']

    n_totp = n_data + n_pred

    # Variable unpacking
    psi_p = xin[idx_psi_p]
    mf_k1 = np.exp(xin[idx_mf_k1])
    mf_m1 = xin[idx_mf_m1]
    k1 = np.exp(xin[idx_k1])

    # Variable reshaping
    psi_p = psi_p.reshape(n_pred, 1)
    mf_k1 = mf_k1.reshape(n_totp, 1)
    mf_m1 = mf_m1.reshape(n_totp, 1)
    k1 = k1.reshape(1, 1)

    # assemble prediction set statistics
    s_psi_p = np.sin(psi_p)
    c_psi_p = np.cos(psi_p)

    c_psi = np.vstack((c_psi_p, c_data))
    s_psi = np.vstack((s_psi_p, s_data))

    # Getting mean field moments
    div_ive_1_0 = ss.ive(1, mf_k1) / ss.ive(0, mf_k1)
    div_ive_2_0 = ss.ive(2, mf_k1) / ss.ive(0, mf_k1)

    c_x = div_ive_1_0 * np.cos(mf_m1)
    s_x = div_ive_1_0 * np.sin(mf_m1)

    # Building inverse sections

    # Gradient assembly point
    # ----------------------------------------------------
    # Prior
    dpri_k1 = 0
    dpri_psi_p = 0.
    dpri_c_x = 0.
    dpri_s_x = - np.dot(mat_kin_ss, s_x)

    # Likelihood
    ds_psi_psi_p = + np.cos(psi_p)
    dc_psi_psi_p = - np.sin(psi_p)

    dlik_s_psi = k1 * s_x[0:n_pred]
    dlik_c_psi = k1 * c_x[0:n_pred]

    dlik_psi_p = dlik_s_psi * ds_psi_psi_p + dlik_c_psi * dc_psi_psi_p

    dlik_k1 = (np.dot(c_psi.T, c_x) + np.dot(s_psi.T, s_x)) - n_totp * (ss.ive(1, k1) / ss.ive(0, k1))

    dlik_c_x = k1 * c_psi
    dlik_s_x = k1 * s_psi

    # Mean field gradients (along with entropy)
    dc_x_mf_k1 = np.cos(mf_m1) * (0.5 * (1. + div_ive_2_0) - div_ive_1_0 ** 2)
    ds_x_mf_k1 = np.sin(mf_m1) * (0.5 * (1. + div_ive_2_0) - div_ive_1_0 ** 2)

    dent_mf_k1 = - 0.5 * mf_k1 * (1. + div_ive_2_0) + mf_k1 * div_ive_1_0 ** 2

    dc_x_mf_m1 = - np.sin(mf_m1) * div_ive_1_0
    ds_x_mf_m1 = + np.cos(mf_m1) * div_ive_1_0

    dent_mf_m1 = 0.

    dmf_k1 = (dpri_c_x + dlik_c_x) * dc_x_mf_k1 + (dpri_s_x + dlik_s_x) * ds_x_mf_k1 + dent_mf_k1
    dmf_m1 = (dpri_c_x + dlik_c_x) * dc_x_mf_m1 + (dpri_s_x + dlik_s_x) * ds_x_mf_m1 + dent_mf_m1

    # Final gradient assembly
    dpsi_p = dlik_psi_p + dpri_psi_p

    dk1 = dlik_k1 + dpri_k1

    # Positivy transformation
    dlog_k1 = dk1 * k1
    dlog_mf_k1 = dmf_k1 * mf_k1

    grad = np.vstack((dpsi_p, dlog_mf_k1, dmf_m1, dlog_k1))

    return - grad.flatten() / n_totp


def inference_model_opt(x0, config):
    return so.minimize(fun=inference_model_obj, x0=x0, args=config, method='L-BFGS-B', jac=inference_model_grad,
                       options={'disp': True})