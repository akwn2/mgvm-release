import numba
import numpy as np
from scipy.special import ive


def logive(nu, kappa):
    return np.log(ive(nu, kappa))

@numba.jit(cache=True)
def gve(nu, m1, m2, k1, k2):
    # set parameters parameters
    res = 5000
    dim = m1.shape[0]

    # preallocate
    g_val = np.zeros(m1.shape, dtype=np.complex)
    th = np.linspace(-np.pi, np.pi, res, endpoint=False)
    step = th[1] - th[0]

    # compute energies
    for ee in xrange(0, dim):
        g_val[ee] = np.sum(np.exp(k1[ee] * (np.cos(th - m1[ee]) - 1) + k2[ee] * (np.cos(2. * (th - m2[ee])) - 1) +
                                  1.j * nu * th)) * step

    return g_val


@numba.jit(cache=True)
def grad_gve(nu, m1, m2, k1, k2):
    # set parameters parameters
    res = 1000
    dim = m1.shape[0]

    # preallocate
    dm1 = np.zeros_like(m1)
    dm2 = np.zeros_like(m2)
    dk1 = np.zeros_like(k1)
    dk2 = np.zeros_like(k2)

    th = np.linspace(-np.pi, np.pi, res, False)
    step = th[1] - th[0]

    # compute moments
    for ee in xrange(0, dim):
        expterm = np.exp(k1[ee] * (np.cos(th - m1[ee]) - 1) + k2[ee] * (np.cos(2. * (th - m2[ee])) - 1) + 1.j * nu * th)
        dm1[ee] = - np.sum(expterm * np.sin(m1[ee] - th)) * step * k1[ee]
        dm2[ee] = - np.sum(expterm * np.sin(2 * (m2[ee] - th))) * step * 2 * k2[ee]
        dk1[ee] = + np.sum(expterm * np.cos(th - m1[ee])) * step
        dk2[ee] = + np.sum(expterm * np.cos(2 * (th - m2[ee]))) * step

    return dm1, dm2, dk1, dk2


@numba.jit(cache=True)
def gv0(m1, m2, k1, k2, res=1000):

    th = np.linspace(-np.pi, np.pi, res)
    step = th[1] - th[0]

    energy = k1 * np.cos(th - m1) + k2 * np.cos(2. * (th - m2))
    weight = k1 + k2

    return (np.sum(np.exp(energy - weight)) * step).flatten()


@numba.jit(cache=True)
def gv(nu, m1, m2, k1, k2):
    # set parameters parameters
    res = 5000
    dim = m1.shape[0]

    # preallocate
    g_val = np.zeros_like(m1)
    th = np.linspace(-np.pi, np.pi, res, endpoint=False)
    step = th[1] - th[0]

    # compute energies
    for ee in xrange(0, dim):
        g_val[ee] = step * np.sum(np.exp(k1[ee] * np.cos(th - m1[ee]) + k2[ee] * np.cos(2. * (th - m2[ee])) +
                                         1.j * nu * th))

    return g_val

