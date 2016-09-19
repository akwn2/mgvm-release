import scipy.linalg as la
import utils.kernels as kernels
import mgvm.vi

import numpy as np
import utils.circular as uc
import utils.plotting as plot
from time import time


def mexhat(t, s, m):
    return (2 / (np.sqrt(3 * s) * np.pi ** 0.25)) * (1 - (t - m) ** 2 / (s ** 2)) * np.exp(-(t - m) ** 2 / (2 * s ** 2))


def toy_fun(x):
    return uc.cfix(15 * mexhat(0.9 * x, 0.15, 0.25) * x ** 2 + 0.75)


def run_case():
    print('--> Create training set')
    x = np.linspace(0, 1, 100)
    y = toy_fun(x)
    x_t = np.array([+0.05, +0.05, +0.17, +0.17, +0.22, +0.30, +0.35, +0.37, +0.52, +0.53, +0.69, +0.70, +0.82, +0.90])
    y_t = np.array([+0.56, +0.65, +0.90, +1.18, +2.39, +3.40, +2.89, +2.64, -2.69, -3.20, -3.40, -2.77, +0.41, +0.35])
    x_v = x
    y_v = y
    n_data = np.alen(x_t)

    print('--> Create prediction set')
    grid_pts = 100
    x_p = np.linspace(0, 1, grid_pts)
    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Calculate kernel')
    # Set kernel parameters
    noise = 1.E-2
    params = {
        's2': 250.00,
        'ell2': 5.50E-2 ** 2,
    }
    k1 = np.array([[10.]])
    k2 = np.array([[0.]])

    y_t = y_t.reshape(n_data, 1)
    x_t = x_t.reshape(n_data, 1)
    x_p = x_p.reshape(grid_pts, 1)

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

    psi_p = (2. * np.random.rand(n_pred, 1) - 1)
    mf_k1 = np.log(np.random.rand(n_totp, 1) * 10)
    mf_m1 = (2. * np.random.rand(n_totp, 1) - 1)

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
        'k2': k2
    }

    xin = np.vstack((psi_p, mf_k1, mf_m1))

    print('--> Starting optimisation')
    t0 = time()
    results = mgvm.vi.inference_model_opt(xin, config)
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
    p, th = mgvm.vi.predictive_dist(n_pred, new_mf_k1, new_mf_m1, k1, 0., pi_shift=True)
    fig, scaling_x, scaling_y, offset_y = plot.circular_error_bars(th, p, True)

    scaled_x_t = x_t * scaling_x
    scaled_y_t = uc.cfix(y_t) * scaling_y + offset_y
    scaled_x_v = x_v * scaling_x
    scaled_y_v = uc.cfix(y_v) * scaling_y + offset_y
    scaled_x_p = x_p * scaling_x
    scaled_y_p = uc.cfix(new_psi_p) * scaling_y + offset_y

    # Now plot the optimised psi's and datapoints
    plot.plot(scaled_x_p, scaled_y_p, 'c.')  # optimised prediction
    plot.plot(scaled_x_t, scaled_y_t, 'xk', mew=2.)  # training set
    plot.plot(scaled_x_v, scaled_y_v, 'ob', fillstyle='none')  # training set

    plot.xticks([0, 20, 40, 60, 80, 100], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    plot.ylabel('Regressed variable $(\psi)$')
    plot.xlabel('Input variable $(x)$')
    plot.tight_layout()

    holl_score = 0.
    for ii in xrange(0, y_v.shape[0]):
        holl_score += uc.holl(y_v[ii], new_psi_p[ii], 0, k1, 0)

    print 'HOLL Score: ' + str(holl_score)
    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)            # fix seed
    run_case()
