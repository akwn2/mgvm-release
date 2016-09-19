"""
cutil.py
circular stats utility functions
"""
import numpy as np
import linalg as ula
from scipy.special import ive
from scipy.stats import norm as gaussian
from scipy.stats import multivariate_normal as mgaussian


# General auxiliary functions
def cmean(x, axis=None):
    """
    Calculates the circular mean of a vector x
    :param x:
    :param axis:
    :return:
    """
    x = x.flatten()
    exp_ix = np.exp(1.j * x)
    return np.angle(np.sum(exp_ix, axis=axis))


# General auxiliary functions
def mod2pi(x, axis=None):
    """
    Calculates the modulo 2 pi
    :param x:
    :param axis:
    :return:
    """
    return np.mod(x, 2 * np.pi)


def cfix(x, axis=None):
    """
    Fix angle to be between -pi and pi
    :param x:
    :param axis:
    :return:
    """
    return mod2pi(x + np.pi) - np.pi


def dive(nu, z):
    if nu == 0:
        return ive(1, z)
    else:
        return 0.5 * (ive(nu - 1, z) + ive(nu + 1, z))


def log_iv(nu, z):
    """
    calculates the log of a modified bessel function
    :param nu: modified bessel function order
    :param z: modified bessel function argument
    :return:
    """
    return z + np.log(ive(nu, z))


def inv_log_iv(nu, log_val):
    """
    Calculates the inverse of the log of a modified bessel function of first kind and order nu using Newton's method.
    :param nu: order of the modified bessel function
    :param log_val: log of the inverse bessel function
    :return:
    """
    tol = 1.E-7
    max_z = 1.E+6
    max_val = np.log(ive(nu, max_z)) + max_z

    f = lambda z: np.log(ive(nu, z)) + z - log_val
    if nu == 0:
        df = lambda z: ive(1, z) / ive(0, z) * np.sign(z)
    else:
        df = lambda z: 0.5 * (ive(nu - 1, z) + ive(nu + 1, z)) / ive(nu, z)

    if log_val < max_val:
        # Newton's method for root finding
        converged = False
        max_iter = 100
        ii = 0
        z = np.random.rand(1, 1)
        while not converged and ii < max_iter:
            z_old = np.copy(z)
            df_z_old = df(z_old)

            # Avoid zero division errors
            if np.abs(df_z_old) > 1.E-8:
                z = z_old - f(z_old) / df_z_old
            else:
                z = z_old - f(z_old) / (df_z_old + 1.E-8)

            converged = np.abs(z - z_old) < tol
            ii += 1

        if not converged:
            print '!!! Warning: Did not converge after ' + str(max_iter) + ' Newton iterations.'
        return z
    else:
        print('!!! Warning: Value supplied greater than ' + str(max_z) + '.')
        return max_z


def quartic(a, b, c, d, e):
    return np.roots([a, b, c, d, e])


def cmse(y, m):
    # Circular-mean-squared error
    return 1 - np.cos(cmean(y - m))


def mce(y, m):
    # Mean cosine error
    return (np.cos(y) - np.cos(m)) ** 2 + (np.sin(y) - np.sin(m)) ** 2


def holl(psi, m1, m2, k1, k2, N=1):

    score = N * (np.log(2. * np.pi * ive(0, k1)) + np.log(2. * np.pi * ive(0, k2)) + k1 + k2) + \
            np.sum(k1 * np.cos(psi - m1) + k2 * np.cos(2. * (psi - m2)))

    return score


def gvm_sample(m1, m2, k1, k2, res=5000, max_reject=250):
    # Sampling from a gvm using rejection sampling

    # Find the supremum of the unnormalised density to create a bounding box
    omega = np.linspace(-np.pi, np.pi, res).reshape(res, 1)
    g = k1 * np.cos(omega - m1) + k2 * np.cos(2. * (omega - m2))
    max_g = np.max(g)

    accepted = False
    rejections = 0
    th = 0.
    while not accepted and rejections < max_reject:
        rejections += 1

        th = 2. * np.pi * np.random.rand(1, 1) - np.pi # pick a point in the unit circle
        w = k1 * np.cos(th - m1) + k2 * np.cos(2. * (th - m2)) # height of the unnormalized density
        log_u = max_g + np.log(np.random.rand(1, 1) + 1E-12)  # height of the bounding box w/ offset to avoid log(0)

        accepted = log_u <= w  # here we took the log of both sides

    if rejections > max_reject:
        print 'Exceeded maximum number of rejections'
        raise ValueError
    else:
        return th


def loglik_gp2circle(psi, psi_p, s2):
    """
    log likelihood for a GP in 2 dimensions projected back to the unit circle (projected normal)
    :param psi: data
    :param psi_p: predictions
    :param s2: variance
    :return:
    """

    inv_cov = np.array([[1. / s2, 0.],
                        [0., 1. / s2]])
    root_inv_cov_det = s2
    v_psi = np.array([[np.cos(psi)],
                      [np.sin(psi)]])
    mu = np.array([[psi_p[0]],
                   [psi_p[1]]])

    norm = ula.qform(v_psi, inv_cov, v_psi) ** 0.5
    d_term = ula.qform(mu, inv_cov, v_psi) / norm
    mu_exterior_x = mu[0] * v_psi[1] - mu[1] * v_psi[0]

    std_cdf_term = gaussian.cdf(d_term)
    cov_pdf_term = np.exp(ula.qform(mu, inv_cov, mu)) / (2. * np.pi * s2)
    std_pdf_term = np.exp(- 0.5 * (mu_exterior_x * root_inv_cov_det / norm) ** 2) / (2 * np.pi) ** 0.5

    log_lik = np.log(cov_pdf_term + root_inv_cov_det * d_term * std_cdf_term * std_pdf_term) - 2. * np.log(norm)

    return log_lik
