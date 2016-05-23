from time import time

import numpy as np
import scipy.io as sio
import scipy.linalg as la

import gp.gp as gp
import utils.circular as uc
import utils.kernels as kernels

import matplotlib.cm as cm
import utils.plotting as plot


def run_case():

    print('Case: Univariate example with mGvM model 1')

    print('--> Load training set')
    x_t = sio.loadmat('./datasets/compact_uber.mat')['x_t']  # training inputs
    psi_t = sio.loadmat('./datasets/compact_uber.mat')['y_t'] - np.pi  # training outputs
    n_data = np.alen(psi_t)

    print('--> Load validation set')
    x_v = sio.loadmat('./datasets/compact_uber.mat')['x_v']  # validation inputs
    psi_v = sio.loadmat('./datasets/compact_uber.mat')['y_v'] - np.pi  # validation outputs

    print('--> Load prediction set')
    x_p = sio.loadmat('./datasets/compact_uber.mat')['x_p']  # prediction inputs

    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Learn GP')
    # Set kernel parameters
    psi_t = psi_t.reshape(n_data, 1)
    y_t = np.hstack((np.cos(psi_t), np.sin(psi_t)))
    x_t = x_t.reshape(n_data, 1)
    x_p = x_p.reshape(n_pred, 1)

    config = {
        'xi': x_t,
        'y': y_t,
    }
    ell2 = 0.15 ** 2
    s2 = 400.
    noise = 1.E-6

    hyp = np.zeros([3, 1])
    hyp[0] = ell2
    hyp[1] = s2
    hyp[2] = noise
    hyp = np.log(hyp.flatten())

    ans = gp.learning_se_iso(hyp, config)
    print ans.message

    hyp = np.exp(ans.x)
    ell2 = hyp[0]
    s2 = hyp[1]
    noise = hyp[2]

    params = {
        'ell2': ell2,
        's2': s2
    }

    print('--> Initialising model variables')
    t0 = time()
    yp, var_yp = gp.predict(y_t, x_t, x_p, params, noise)
    tf = time()

    print 'Total elapsed time in prediction: ' + str(tf - t0) + ' s'

    # Keep all values between -pi and pi
    z_p = yp[:, 0] + 1.j * yp[:, 1]
    s2_p = np.diag(var_yp)

    new_psi_p = np.angle(z_p)
    n_grid = 1000
    p, th = gp.get_predictive_for_wrapped(new_psi_p, var_yp, res=n_grid)

    rmse_score = 0.
    holl_score = 0.
    train_ports = x_t
    valid_ports = x_v
    for ii in xrange(0, n_pred):
        p_idx = p[:, ii]
        if ii in train_ports:
            psi_idx = psi_t[train_ports == ii]
        else:
            psi_idx = psi_v[valid_ports == ii]
            holl_score += np.sum(
                np.exp(-0.5 * (np.cos(psi_v[ii]) - yp[ii, 0]) ** 2 / s2_p) / np.sqrt(np.pi * 2. * s2_p) +
                np.exp(-0.5 * (np.sin(psi_v[ii]) - yp[ii, 1]) ** 2 / s2_p) / np.sqrt(np.pi * 2. * s2_p))

        rmse_score += uc.rmse(psi_idx, p_idx, th)

    print 'RMSE score: ' + str(rmse_score)
    print 'HOLL score: ' + str(holl_score)
    # Predictions
    print('--> Saving and displaying results')

    # fig, scaling_x, scaling_y, offset_y = plot.circular_error_bars(th, p)

    # fig.savefig('./results/uber_sincos_gp.pdf')
    np.save('./results/uber_sincos_gp', config)
    plot.show()

    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
