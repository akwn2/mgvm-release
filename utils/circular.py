"""
cutil.py
circular stats utility functions
"""
import numpy as np
import scipy as sp
from scipy.special import ive


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
    return z + sp.log(ive(nu, z))


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


def rmse(y, p, th, n_bins=20):

    hist, bins = np.histogram(y, bins=n_bins, range=(0, 2 * np.pi))
    scale_factor = np.max(hist) / np.max(p)

    # Compute RMSE
    aux = 0.
    for nn in xrange(0, bins.shape[0] - 1):
        idx = np.logical_and(th >= bins[nn], th <= bins[nn + 1]).flatten()
        if np.alen(p[idx]) > 0:
            aux += np.sum((hist[nn] - p[idx] * scale_factor) ** 2)

    return np.sqrt(aux / bins.shape[0])


def holl(psi, m1, m2, k1, k2, N=1):

    score = N * (np.log(2. * np.pi * ive(0, k1)) + np.log(2. * np.pi * ive(0, k2)) + k1 + k2) + \
            np.sum(k1 * np.cos(psi - m1) + k2 * np.cos(2. * (psi - m2)))

    return score