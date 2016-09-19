from time import time
import csv
import numpy as np
import gp.gp as gp
import utils.plotting as plot


def run_case():
    print('--> Load data set')
    data = list()
    with open('../datasets/Spellman-alpha.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    data = np.array(data[1:]).astype(np.float_)

    train_idx = np.arange(0, 18, 2)
    valid_idx = np.arange(1, 18, 2)

    print('--> Preparing training set')
    x_t = data[train_idx, 1:]  # training inputs
    psi_t = data[train_idx, 0]  # training outputs
    n_data = np.alen(psi_t)
    d_input = x_t.shape[1]

    print('--> Preparing validation set')
    x_v = data[valid_idx, 1:]  # validation inputs
    psi_v = data[valid_idx, 0].reshape(len(valid_idx), 1)  # validation outputs

    print('--> Load prediction set')
    x_p = x_v # prediction inputs
    n_pred = x_p.shape[0]

    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Learn GP')
    # Set kernel parameters
    psi_t = psi_t.reshape(n_data, 1)
    y_t = np.hstack((np.cos(psi_t), np.sin(psi_t)))
    x_t = x_t.reshape(n_data, d_input)
    x_p = x_p.reshape(n_pred, d_input)

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
    yp, var_yp = gp.predict_se_iso(y_t, x_t, x_p, params, noise)
    tf = time()

    print 'Total elapsed time in prediction: ' + str(tf - t0) + ' s'
    s2_p = np.diag(var_yp)
    holl_score = 0.
    for ii in xrange(0, n_pred):
        holl_score += np.sum(-0.5 * (yp[ii] - psi_v[ii]) ** 2 / s2_p[ii] - 0.5 * np.log(s2_p[ii] * 2. * np.pi))
    print 'HOLL score: ' + str(holl_score)
    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
