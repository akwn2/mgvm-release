from time import time

import numpy as np
import scipy.io as sio
import scipy.linalg as la

import gp.gp as gp
import utils.circular as uc
import utils.kernels as kernels
import utils.plotting as plot


def run_case():
    print('Case: EasyTide example with naive GP')

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

    print('--> Learn GP')
    # Set kernel parameters
    y_t = y_t.reshape(n_data, 1)
    x_t = x_t.reshape(n_data, 2)
    x_p = x_p.reshape(n_pred, 2)

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
    y_p, var_y_p = gp.predict(y_t, x_t, x_p, params, noise)
    tf = time()
    s2_p = np.diag(var_y_p)
    print 'Total elapsed time in prediction: ' + str(tf - t0) + ' s'

    # Keep all values between -pi and pi
    new_psi_p = uc.cfix(y_p)

    n_grid = 1000
    p, th = gp.get_predictive_for_plots_1d(y_p, var_y_p, res=1000)

    rmse_score = 0.
    holl_score = 0.
    train_ports = pid_t.flatten()
    valid_ports = pid_v.flatten()
    for ii in xrange(0, n_pred):
        p_idx = p[:, ii]
        if ii in train_ports:
            y_t_idx = y_t[train_ports == ii]
        else:
            y_t_idx = y_v[valid_ports == ii]
            holl_score += np.sum(-0.5 * (y_p[ii] - y_t_idx) ** 2 / s2_p[ii] - 0.5 * np.log(s2_p[ii] * 2. * np.pi))

        rmse_score += uc.rmse(y_t_idx, p_idx, th)

    print 'RMSE score: ' + str(rmse_score)
    print 'HOLL score: ' + str(holl_score)
    # Predictions
    print('--> Saving and displaying results')

    # First the heatmap in the background
    fig = plot.tides(th, p, y_t, y_v, pid_t, pid_v)

    fig.savefig('./results/tides_naive_gp.pdf')
    np.save('./results/tides_naive_gp', (params, noise, config))
    # fig.show()

    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
