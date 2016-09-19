import gp.gp as gp
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
    y_t = uc.cfix(y_t)
    y_v = uc.cfix(y_v)
    n_data = np.alen(x_t)

    print('--> Create prediction set')
    grid_pts = 100
    x_p = np.linspace(0, 1, grid_pts)
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
    ell2 = 0.5 ** 2
    s2 = 200.
    noise = 1.E-4

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

    new_psi_p = yp
    s2_p = np.diag(var_yp)

    p, th = gp.get_predictive_for_plots_1d(yp, var_yp, res=1000)

    # Predictions
    print('--> Saving and displaying results')

    # First the heatmap in the background
    fig, scaling_x, scaling_y, offset_y = plot.circular_error_bars(th, p, True)

    # Then scale the predicted and training sets to match the dimensions of the heatmap
    scaled_x_t = x_t * scaling_x
    scaled_y_t = uc.cfix(y_t) * scaling_y + offset_y
    scaled_x_v = x_v * scaling_x
    scaled_y_v = uc.cfix(y_v) * scaling_y + offset_y
    scaled_x_p = x_p * scaling_x
    scaled_y_p = new_psi_p * scaling_y + offset_y
    # scaled_mode = (mode + np.pi) * 1000 / (2 * np.pi)

    # Now plot the optimised psi's and datapoints
    plot.plot(scaled_x_p, scaled_y_p, 'c.')  # optimised prediction
    plot.plot(scaled_x_t, scaled_y_t, 'xk', mew=2.0)  # training set
    plot.plot(scaled_x_v, scaled_y_v, 'ob', fillstyle='none')  # training set

    plot.ylabel('Regressed variable $(\psi)$')
    plot.xlabel('Input variable $(x)$')
    plot.tight_layout()

    holl_score = 0.
    for ii in xrange(0, y_v.shape[0]):
        holl_score += - np.sum(0.5 * (yp[ii] - y_v[ii]) ** 2 / s2_p[ii] - 0.5 * np.log(s2_p[ii] * 2. * np.pi))
    print 'HOLL Score: ' + str(holl_score)
    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)            # fix seed
    run_case()
