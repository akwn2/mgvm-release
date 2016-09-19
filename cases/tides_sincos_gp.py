from time import time

import numpy as np
import scipy.io as sio

import gp.gp as gp
import utils.circular as uc
import utils.plotting as plot


def run_case():
    print('--> Load training set')
    x_t = sio.loadmat('../datasets/simpletide.mat')['x_t']  # training inputs
    psi_t = sio.loadmat('../datasets/simpletide.mat')['y_t'] - np.pi  # training outputs
    pid_t = sio.loadmat('../datasets/simpletide.mat')['pid_t']  # port ids for training set
    n_data = np.alen(x_t)

    print('--> Load validation set')
    x_v = sio.loadmat('../datasets/simpletide.mat')['x_v']  # validation inputs
    psi_v = sio.loadmat('../datasets/simpletide.mat')['y_v'] - np.pi  # validation outputs
    pid_v = sio.loadmat('../datasets/simpletide.mat')['pid_v']  # port ids for validation set

    print('--> Load prediction set')
    x_p = sio.loadmat('../datasets/simpletide.mat')['x_p']  # prediction inputs
    pid_p = sio.loadmat('../datasets/simpletide.mat')['pid_p']  # port ids for prediction set
    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Learn GP')
    # Set kernel parameters
    psi_t = psi_t.reshape(n_data, 1)

    y_t = np.hstack((np.cos(psi_t), np.sin(psi_t)))

    x_t = x_t.reshape(n_data, 2)
    x_p = x_p.reshape(n_pred, 2)

    config = {
        'xi': x_t,
        'y': y_t,
    }
    ell2 = 2.5 ** 2
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
    yp, var_y_p = gp.predict_se_iso(y_t, x_t, x_p, params, noise)
    tf = time()

    print 'Total elapsed time in prediction: ' + str(tf - t0) + ' s'

    # Keep all values between -pi and pi
    z_p = yp[:,0] + 1.j * yp[:, 1]
    s2_p = np.diag(var_y_p)

    print('--> Calculating scores')
    holl_score = 0.
    for ii in xrange(0, n_pred):
        holl_score += uc.loglik_gp2circle(psi_v[ii, 0], z_p[ii], s2_p[ii])

    print 'HOLL score: ' + str(holl_score)

    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
