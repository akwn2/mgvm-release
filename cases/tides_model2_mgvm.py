from time import time

import numpy as np
import scipy.io as sio
import scipy.linalg as la

import simple.mgvm_vi
import utils.circular as uc
import utils.kernels as kernels
import utils.plotting as plot


def run_case():
    print('Case: EasyTide example with mGvM model 3')

    print('--> Load training set')
    x_t = sio.loadmat('./datasets/simpletide.mat')['x_t']  # training inputs
    y_t = sio.loadmat('./datasets/simpletide.mat')['y_t'] - np.pi  # training outputs
    pid_t = sio.loadmat('./datasets/simpletide.mat')['pid_t']  # port ids for training set
    n_data = np.alen(y_t)

    print('--> Load validation set')
    x_v = sio.loadmat('./datasets/simpletide.mat')['x_v']  # validation inputs
    y_v = sio.loadmat('./datasets/simpletide.mat')['y_v'] - np.pi  # validation outputs
    pid_v = sio.loadmat('./datasets/simpletide.mat')['pid_v']  # port ids for validation set

    print('--> Load prediction set')
    x_p = sio.loadmat('./datasets/simpletide.mat')['x_p']  # prediction inputs
    pid_p = sio.loadmat('./datasets/simpletide.mat')['pid_p']  # port ids for prediction set
    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Calcualte kernel')
    # Set kernel parameters
    params = {
        's2': 700.00,
        'ell2': 2.5E-2 ** 2,
    }

    y_t = y_t.reshape(n_data, 1)
    x_t = x_t.reshape(n_data, 2)
    x_p = x_p.reshape(n_pred, 2)

    x = np.vstack((x_p, x_t))

    # Calculate kernels
    mat_k_cc = kernels.se_iso(x, x, params)
    mat_k_ss = kernels.se_iso(x, x, params)

    mat_k = np.bmat([[mat_k_cc, np.zeros_like(mat_k_cc)], [np.zeros_like(mat_k_ss), mat_k_ss]])
    mat_k = np.asarray(mat_k)
    mat_k += 1E-0 * np.eye(mat_k.shape[0])

    # Find inverse
    mat_ell = la.cholesky(mat_k, lower=True)
    mat_kin = la.solve(mat_ell.T, la.solve(mat_ell, np.eye(mat_ell.shape[0])))

    print('--> Initialising model variables')

    psi_p = np.random.rand(n_pred, 1)
    mf_k1 = np.log(np.random.rand(n_totp, 1)) * 5
    mf_m1 = np.random.rand(n_totp, 1) - np.pi
    k2 = np.log(np.random.rand(1, 1)) * 10

    n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + k2.shape[0]
    idx = np.arange(0, n_var)

    config = {
        'N_data': n_data,
        'N_pred': n_pred,
        'c_data': np.cos(y_t),
        's_data': np.sin(y_t),
        'c_2data': np.cos(2 * y_t),
        's_2data': np.sin(2 * y_t),
        'Kinv': mat_kin,
        'idx_2psi_p': idx[0:psi_p.shape[0]],
        'idx_mf_k1': idx[psi_p.shape[0]:psi_p.shape[0] + mf_k1.shape[0]],
        'idx_mf_m1': idx[psi_p.shape[0] + mf_k1.shape[0]:psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]],
        'idx_k2': idx[-1],
    }

    xin = np.vstack((psi_p, mf_k1, mf_m1, k2))

    print('--> Starting optimisation')
    t0 = time()
    results = simple.mgvm_vi.inference_model_2_opt(xin, config)
    tf = time()

    print 'Total elapsed time: ' + str(tf - t0) + ' s'
    print results.message

    # Keep all values between -pi and pi
    new_2psi_p = uc.cfix(results.x[config['idx_2psi_p']])
    new_mf_k1 = results.x[config['idx_mf_k1']]
    new_mf_m1 = results.x[config['idx_mf_m1']]
    new_k2 = np.exp(results.x[config['idx_k2']])

    # Predictions
    print('--> Saving and displaying results')

    # First the heatmap in the background
    p, th, mode_1, mode_2 = simple.mgvm_vi.predictive_dist_model_2(n_pred, new_mf_k1, new_mf_m1, new_k2, res=1000)
    fig = plot.tides(th, p, y_t, y_v, pid_t, pid_v)

    fig.savefig('../results/tides_model2_mgvm.pdf')
    np.save('../results/tides_model2_mgvm', (results, config))
    # fig.show()

    print('Finished running case!')

if __name__ == '__main__':
    #plt.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
