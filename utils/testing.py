"""
tutil.py
testing utility functions
"""
import numpy as np


def assert_real(x):
    if np.any(np.isnan(x)):
        raise AssertionError('!!! ERROR: NaN token found.')
    elif np.any(np.isinf(x)):
        raise AssertionError('!!! ERROR: Inf token found.')
    elif np.any(np.iscomplex(x)):
        raise AssertionError('!!! ERROR: Complex number found when result should be real.')


def fd_array_func(f, x, y, precision=6, delta=1.e-7):
    """
    checks if a computed gradient y at a point x for a function f coincides with finite differences
    :param f: function one wants to get the gradients of
    :param x: input array
    :param y: analytic gradient at x
    :param precision: decimal places to consider y == x.
    :param delta: perturbation magnitude
    :return:
    """
    grad_fd = np.zeros(x.shape)
    for ii in xrange(0, x.size):
        pert = np.zeros(x.shape)
        pert[ii] = delta

        grad_p = f(x + pert)
        grad_m = f(x - pert)

        grad_fd[ii] = (grad_p - grad_m) / (2. * delta)

    np.testing.assert_array_almost_equal(grad_fd, y, decimal=precision)
