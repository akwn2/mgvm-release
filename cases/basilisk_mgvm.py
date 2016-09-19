from time import time

import numpy as np
import mgvm.vi
import utils.circular as uc
import utils.kernels as kernels
import utils.linalg as ula


def run_case():
    print('--> Load data set')
    training_data = np.load('./datasets/basilisk_ASP_train.npy')
    subset = np.arange(0, 1000, 1)

    x_t = uc.cfix(training_data[subset, 0:2])  # training inputs
    psi_t = uc.cfix(training_data[subset, 2])  # training outputs
    n_data = np.alen(psi_t)

    print('--> Load validation set')
    validation_data = np.load('./datasets/basilisk_ASP_test.npy')
    x_p = uc.cfix(validation_data[subset, 0:2])  # training inputs
    psi_v = uc.cfix(validation_data[subset, 2])  # training outputs

    print('--> Load prediction set')
    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    # Set kernel parameters
    psi_t = psi_t.reshape(n_data, 1)
    x_t = x_t.reshape(n_data, 2)
    x_p = x_p.reshape(n_pred, 2)

    print('--> Calculate kernel')
    # Set kernel parameters
    noise = 1E-4
    params_cc = {
        's2': 300.00,
        'ell2': 2.5E-1 ** 2,
        'per': 6.0
    }
    params_ss = params_cc
    # params_cs = params_cc
    k1 = np.array([[125.0]])
    k2 = np.array([[0.0]])

    x = np.vstack((x_p, x_t))

    # Calculate kernels
    mat_k_cc = kernels.pe_iso(x, x, params_cc)
    mat_k_ss = kernels.pe_iso(x, x, params_ss)
    mat_k_cs = np.zeros([n_totp, n_totp])

    mat_k = np.bmat([[mat_k_cc,   mat_k_cs],
                     [mat_k_cs.T, mat_k_ss]])
    mat_k = np.asarray(mat_k)
    mat_k += noise * np.eye(mat_k.shape[0])

    # Find inverse
    mat_ell, jitter = ula.rchol(mat_k)
    mat_kin = ula.invc(mat_ell)

    print('--> Initialising model variables')

    psi_p = (2 * np.random.rand(n_pred, 1) - 1) * 2
    mf_k1 = np.log(np.random.rand(n_totp, 1) * 20.1)
    mf_m1 = (2 * np.random.rand(n_totp, 1) - 1) * 2

    n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]
    idx = np.arange(0, n_var)

    config = {
        'N_data': n_data,
        'N_pred': n_pred,
        'c_data': np.cos(psi_t),
        's_data': np.sin(psi_t),
        'c_2data': np.cos(2 * psi_t),
        's_2data': np.sin(2 * psi_t),
        'Kinv': mat_kin,
        'idx_psi_p': idx[0:psi_p.shape[0]],
        'idx_mf_k1': idx[psi_p.shape[0]:psi_p.shape[0] + mf_k1.shape[0]],
        'idx_mf_m1': idx[psi_p.shape[0] + mf_k1.shape[0]:psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]],
        'k1': k1,
        'k2': k2,
    }
    xin = np.vstack((psi_p, mf_k1, mf_m1))

    print('--> Starting optimisation')
    t0 = time()
    results = mgvm.vi.inference_model_opt(xin, config)
    tf = time()

    print 'Total elapsed time: ' + str(tf - t0) + ' s'
    print results.message

    # Predictions
    print('--> Saving and displaying results')
    holl_score = 0.
    new_psi_p = uc.cfix(results.x[config['idx_psi_p']])
    for ii in xrange(0, n_pred):
        m1_idx = new_psi_p[ii]
        m2_idx = new_psi_p[ii]
        holl_score += uc.holl(psi_v[ii], m1_idx, m2_idx, k1, k2)

    print 'HOLL score: ' + str(holl_score)
    print('Finished running case!')

if __name__ == '__main__':
    np.random.seed(200)           # fix seed
    run_case()
