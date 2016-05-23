from time import time
import scipy.linalg as la
import utils.kernels as kernels
import simple.mgvm_vi
import numpy as np
import utils.plotting as plot
import utils.circular as uc


def run_case():
    print('Case: Univariate example with mvM model 1')

    print('--> Create training set')
    x_t = np.array([+0.05, +0.05, +0.15, +0.15, +0.20, +0.25, +0.30, +0.32, +0.58, +0.60, +0.69, +0.70, +0.82, +0.90])
    y_t = np.array([+0.56, +0.65, +0.90, +1.18, +2.39, +3.40, +2.89, +2.64, -2.71, -3.20, -3.40, -2.77, +0.84, +0.25])
    n_data = np.alen(x_t)

    print('--> Create prediction set')
    grid_pts = 100
    x_p = np.linspace(0, 1, grid_pts)
    n_pred = np.alen(x_p)
    n_totp = n_data + n_pred

    print('--> Calcualte kernel')
    # Set kernel parameters
    noise = 1E+1
    y_t = y_t.reshape(n_data, 1)
    x_t = x_t.reshape(n_data, 1)
    x_p = x_p.reshape(grid_pts, 1)

    x = np.vstack((x_p, x_t))
    mat_k = noise * np.eye(2 * x.shape[0])

    # Find inverse
    mat_ell = la.cholesky(mat_k, lower=True)
    mat_kin = la.solve(mat_ell.T, la.solve(mat_ell, np.eye(mat_ell.shape[0])))

    # Impose inverse covariance mvM constraints
    diag_mat_kin = np.diag(mat_kin)
    mat_kin = np.zeros_like(mat_kin)
    np.fill_diagonal(mat_kin, diag_mat_kin)

    print('--> Initialising model variables')
    # here we need to help this to converge to a sensible solution
    psi_p = np.random.rand(n_pred, 1)
    mf_k1 = np.log(np.random.rand(n_totp, 1) * 20)
    mf_m1 = np.random.rand(n_totp, 1)
    k1 = np.log(np.random.rand(1, 1) * 10)

    n_var = psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0] + k1.shape[0]
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
        'idx_k1': idx[psi_p.shape[0] + mf_k1.shape[0] + mf_m1.shape[0]:n_var],
    }

    xin = np.vstack((psi_p, mf_k1, mf_m1, k1))

    print('--> Starting optimisation')
    t0 = time()
    results = simple.mgvm_vi.inference_model_1_opt(xin, config)
    tf = time()

    print 'Total elapsed time: ' + str(tf - t0) + ' s'

    print results.message

    # Keep all values between -pi and pi
    new_psi_p = uc.cfix(results.x[config['idx_psi_p']])
    new_mf_k1 = results.x[config['idx_mf_k1']]
    new_mf_m1 = results.x[config['idx_mf_m1']]
    new_k1 = np.exp(results.x[config['idx_k1']])

    # Predictions
    print('--> Saving and displaying results')

    # First the heatmap in the background
    p, th = simple.mgvm_vi.predictive_dist(n_pred,new_mf_k1, new_mf_m1, new_k1, 0.)
    fig, scaling_x, scaling_y, offset_y = plot.circular_error_bars(th, p)

    scaled_x_t = x_t * scaling_x
    scaled_y_t = uc.cfix(y_t) * scaling_y + offset_y
    scaled_x_p = x_p * scaling_x
    scaled_y_p = uc.cfix(new_psi_p) * scaling_y + offset_y

    # Now plot the optimised psi's and datapoints
    # plt.plot(scaled_mode, 'go', ms=5.0, mew=0.1)  # mode of predictive
    plot.plot(scaled_x_p, scaled_y_p, 'c*', ms=10.0, mew=0.1)  # optimised prediction
    plot.plot(scaled_x_t, scaled_y_t, 'xk', ms=10.0, mew=2.0)  # training set
    plot.tight_layout()

    plot.xticks([0, 20, 40, 60, 80, 100], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    plot.ylabel('Regressed variable $(\psi)$')
    plot.xlabel('Input variable $(x)$')
    axes = plot.gca()
    axes.set_xlim([0, p.shape[1]])

    fig.savefig('./results/uni_model1_mvm.pdf')
    np.save('./results/uni_model1_mvm', (results, config))
    plot.show()

    print('Finished running case!')

if __name__ == '__main__':
    plot.switch_backend('agg')   # no display
    np.random.seed(0)           # fix seed
    run_case()
