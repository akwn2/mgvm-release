from time import time

import numpy as np
import scipy.io as sio
import scipy.linalg as la

import gp.gp as gp
import utils.circular as uc
import utils.kernels as kernels
import utils.plotting as plot


def run_case():
    print('Case: EasyTide example with sincos GP')

    print('--> Load training set')
    x_t = sio.loadmat('./datasets/simpletide.mat')['x_t']  # training inputs
    psi_t = sio.loadmat('./datasets/simpletide.mat')['y_t'] - np.pi  # training outputs
    pid_t = sio.loadmat('./datasets/simpletide.mat')['pid_t']  # port ids for training set
    n_data = np.alen(x_t)

    print('--> Load validation set')
    x_v = sio.loadmat('./datasets/simpletide.mat')['x_v']  # validation inputs
    psi_v = sio.loadmat('./datasets/simpletide.mat')['y_v'] - np.pi  # validation outputs
    pid_v = sio.loadmat('./datasets/simpletide.mat')['pid_v']  # port ids for validation set

    print('--> Load prediction set')
    x_p = sio.loadmat('./datasets/simpletide.mat')['x_p']  # prediction inputs
    pid_p = sio.loadmat('./datasets/simpletide.mat')['pid_p']  # port ids for prediction set
    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Learn GP')
    # Set kernel parameters
    psi_t = psi_t.reshape(n_data, 1)
    # psi_v = psi_v.reshape(n_data, 1)

    y_t = np.hstack((np.cos(psi_t), np.sin(psi_t)))

    x_t = x_t.reshape(n_data, 2)
    x_p = x_p.reshape(n_pred, 2)

    config = {
        'xi': x_t,
        'y': y_t,
    }
    ell2 = 1.5 ** 2
    s2 = 100.
    noise = 1.E2

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
    y_p, var_y_p = gp.predict(y_t, x_t, x_p, params, noise)
    tf = time()

    print 'Total elapsed time in prediction: ' + str(tf - t0) + ' s'

    # Keep all values between -pi and pi
    z_p = y_p[:,0] + 1.j * y_p[:, 1]
    s2_p = np.diag(var_y_p)

    new_psi_p = np.angle(z_p)
    n_grid = 1000
    p, th = gp.get_predictive_for_wrapped(new_psi_p, var_y_p, res=n_grid)

    rmse_score = 0.
    holl_score = 0.
    train_ports = pid_t.flatten()
    valid_ports = pid_v.flatten()
    for ii in xrange(0, n_pred):
        p_idx = p[:, ii]
        if ii in train_ports:
            psi_idx = psi_t[train_ports == ii]
        else:
            psi_idx = psi_v[valid_ports == ii]
            holl_score += np.sum(
                np.exp(-0.5 * (np.cos(psi_v[ii]) - y_p[ii, 0]) ** 2 / s2_p) / np.sqrt(np.pi * 2. * s2_p) +
                np.exp(-0.5 * (np.sin(psi_v[ii]) - y_p[ii, 1]) ** 2 / s2_p) / np.sqrt(np.pi * 2. * s2_p))

        rmse_score += uc.rmse(psi_idx, p_idx, th)

    print 'RMSE score: ' + str(rmse_score)
    print 'HOLL score: ' + str(holl_score)
    # Predictions
    print('--> Saving and displaying results')

    # First the heatmap in the background
    fig = plot.tides(th, p, psi_t, psi_v, pid_t, pid_v)

    fig.savefig('./results/tides_sincos_gp.svg')
    np.save('./results/tides_sincos_gp', (params, noise, config))
    # fig.show()

    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
