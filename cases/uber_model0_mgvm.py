from time import time

import numpy as np
import scipy.io as sio
import scipy.linalg as la

import simple.mgvm_vi
import utils.circular as uc
import utils.kernels as kernels
import utils.plotting as plot


def run_case():
    print('Case: Compact uber dataset example with mGvM model 3')

    print('--> Load training set')
    x_t = sio.loadmat('./datasets/compact_uber.mat')['x_t']  # training inputs
    y_t = sio.loadmat('./datasets/compact_uber.mat')['y_t'] - np.pi  # training outputs
    n_data = np.alen(y_t)

    print('--> Load validation set')
    x_v = sio.loadmat('./datasets/compact_uber.mat')['x_v']  # validation inputs
    y_v = sio.loadmat('./datasets/compact_uber.mat')['y_v'] - np.pi  # validation outputs

    print('--> Load prediction set')
    x_p = sio.loadmat('./datasets/compact_uber.mat')['x_p']  # prediction inputs

    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Calculate kernel')
    # Set kernel parameters
    noise = 1E-1
    params = {
        's2': 250.00,
        'ell2': 5.0E+1 ** 2,
    }
    k1 = np.array([[4.0]])
    k2 = np.array([[7.0]])  # previously 7.0

    y_t = y_t.reshape(n_data, 1)
    x_t = x_t.reshape(n_data, 1)
    x_p = x_p.reshape(n_pred, 1)

    x = np.vstack((x_p, x_t))

    # Calculate kernels
    mat_k_cc = kernels.se_iso(x, x, params)
    mat_k_ss = kernels.se_iso(x, x, params)

    mat_k = np.bmat([[mat_k_cc, np.zeros_like(mat_k_cc)], [np.zeros_like(mat_k_ss), mat_k_ss]])
    mat_k = np.asarray(mat_k)
    mat_k += noise * np.eye(mat_k.shape[0])

    # Find inverse
    mat_ell = la.cholesky(mat_k, lower=True)
    mat_kin = la.solve(mat_ell.T, la.solve(mat_ell, np.eye(mat_ell.shape[0])))

    print('--> Initialising model variables')

    psi_p = (2 * np.random.rand(n_pred, 1) - 1) * 2
    mf_k1 = np.log(np.random.rand(n_totp, 1) * 0.1)
    mf_m1 = (2 * np.random.rand(n_totp, 1) - 1) * 2

    n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]
    idx = np.arange(0, n_var)

    config = {
        'N_data': n_data,
        'N_pred': n_pred,
        'c_data': np.cos(y_t),
        's_data': np.sin(y_t),
        'c_2data': np.cos(2 * y_t),
        's_2data': np.sin(2 * y_t),
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
    results = simple.mgvm_vi.inference_model_0_opt(xin, config)
    tf = time()

    print 'Total elapsed time: ' + str(tf - t0) + ' s'

    print results.message

    # Keep all values between -pi and pi
    new_psi_p = uc.cfix(results.x[config['idx_psi_p']])
    new_mf_k1 = results.x[config['idx_mf_k1']]
    new_mf_m1 = results.x[config['idx_mf_m1']]

    # Predictions
    print('--> Saving and displaying results')

    # First the heatmap in the background
    p, th = simple.mgvm_vi.predictive_dist(n_pred, new_mf_k1, new_mf_m1, k1, k2, res=1000)

    rmse_score = 0.
    holl_score = 0.
    train_ports = x_t
    valid_ports = x_v
    
    for ii in xrange(0, n_pred):
        p_idx = p[:, ii]
        m1_idx = new_psi_p[ii]
        m2_idx = new_psi_p[ii]
        if ii in train_ports:
            y_t_idx = y_t[train_ports == ii]
        else:
            y_t_idx = y_v[valid_ports == ii]
        rmse_score += uc.rmse(y_t_idx, p_idx, th)
        holl_score += uc.holl(y_t_idx, m1_idx, m2_idx, k1, k2, np.alen(y_t_idx))

    print 'RMSE score: ' + str(rmse_score)
    print 'HOLL score: ' + str(holl_score)

    # fig, scaling_x, scaling_y, offset_y = plot.circular_error_bars(th, p)
    # fig.savefig('./results/uber_model3_mgvm.pdf')
    np.save('./results/uber_model0_mgvm', (results, config))
    # fig.show()

    print('Finished running case!')

if __name__ == '__main__':
    # plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
