from time import time
import numpy as np
import gp.gp as gp
import utils.circular as uc


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
    ell2 = 1.00 ** 2
    s2 = 200.
    per = 0.5
    noise = 1.E-4

    hyp = np.zeros([4, 1])
    hyp[0] = ell2
    hyp[1] = s2
    hyp[2] = per
    hyp[3] = noise
    hyp = np.log(hyp.flatten())

    ans = gp.learning_pe_iso(hyp, config)
    print ans.message

    hyp = np.exp(ans.x)
    ell2 = hyp[0]
    s2 = hyp[1]
    per = hyp[2]
    noise = hyp[3]

    params = {
        'ell2': ell2,
        's2': s2,
        'per': per
    }

    print('--> Initialising model variables')
    t0 = time()
    new_psi_p, var_psi_p = gp.predict_pe_iso(y_t, x_t, x_p, params, noise)
    tf = time()

    print 'Total elapsed time in prediction: ' + str(tf - t0) + ' s'

    # Keep all values between -pi and pi
    s2_p = np.diag(var_psi_p)
    holl_score = 0.
    for ii in xrange(0, n_pred):
        holl_score += uc.loglik_gp2circle(psi_v[ii], new_psi_p[ii], s2_p[ii])

    print 'HOLL score: ' + str(holl_score)
    print('Finished running case!')

if __name__ == '__main__':
    np.random.seed(0)           # fix seed
    run_case()
