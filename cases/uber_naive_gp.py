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
    y_t = sio.loadmat('./datasets/compact_uber.mat')['y_t'] - np.pi  # training outputs
    n_data = np.alen(y_t)

    print('--> Load validation set')
    x_v = sio.loadmat('./datasets/compact_uber.mat')['x_v']  # validation inputs
    y_v = sio.loadmat('./datasets/compact_uber.mat')['y_v'] - np.pi  # validation outputs

    print('--> Load prediction set')
    x_p = sio.loadmat('./datasets/compact_uber.mat')['x_p']  # prediction inputs

    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Learn GP')
    # Set kernel parameters
    y_t = y_t.reshape(n_data, 1)
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

    new_psi_p = yp
    s2_p = np.diag(var_yp)

    n_grid = 1000
    p, th = gp.get_predictive_for_plots_1d(yp, var_yp, res=n_grid)

    rmse_score = 0.
    holl_score = 0.
    train_ports = x_t
    valid_ports = x_v
    for ii in xrange(0, n_pred):
        p_idx = p[:, ii]
        if ii in train_ports:
            y_t_idx = y_t[train_ports == ii]
        else:
            y_t_idx = y_v[valid_ports == ii]
            holl_score += np.sum(-0.5 * (yp[ii] - y_t_idx) ** 2 / s2_p[ii] - 0.5 * np.log(s2_p[ii] * 2. * np.pi))

        rmse_score += uc.rmse(y_t_idx, p_idx, th)

    print 'RMSE score: ' + str(rmse_score)
    print 'HOLL score: ' + str(holl_score)

    # Predictions
    print('--> Saving and displaying results')

    # fig, scaling_x, scaling_y, offset_y = plot.circular_error_bars(th, p)
    # # Then scale the predicted and training sets to match the dimensions of the heatmap
    # scaled_x_t = x_t * scaling_x
    # scaled_y_t = y_t * scaling_y
    # scaled_x_p = x_p * scaling_x
    # scaled_y_p = new_psi_p * scaling_y
    #
    # # Now plot the optimised psi's and datapoints
    # # plt.plot(scaled_mode, 'go', ms=5.0, mew=0.1)  # mode of predictive
    # plot.plot(scaled_x_p, scaled_y_p, 'c*', ms=10.0, mew=0.1)  # optimised prediction
    # plot.plot(scaled_x_t, scaled_y_t, 'xk', ms=10.0, mew=2.0)  # training set
    #
    # plot.ylabel('Regressed variable $(\psi)$')
    # plot.xlabel('Input variable $(x)$')

    # fig.savefig('./results/uber_naive_gp.pdf')
    np.save('./results/uber_naive_gp', config)
    plot.show()

    print('Finished running case!')

if __name__ == '__main__':
    # plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
