import numpy as np
import scipy.spatial.distance as spdist
from numba import jit


@jit(cache=True)
def sd_iso(x, xp, params, get_gradients=False):
    """
    compute a scaled diagonal matrix
    :param x:
    :param xp:
    :param params:
    :param get_gradients:
    :return:
    """
    n = x.shape[0]
    m = xp.shape[0]
    K = np.eye(n, m) * params['s2']
    if get_gradients:
        grad = dict()
        grad['s2'] = np.eye(n, m)
        return K, grad
    else:
        return K


@jit(cache=True)
def sd_ard(x, xp, params, get_gradients=False):
    """
    compute a scaled diagonal matrix with different diagonals
    :param x:
    :param xp:
    :param params:
    :param get_gradients:
    :return:
    """
    n = x.shape[0]
    m = xp.shape[0]
    K = np.eye(n, m) * np.diag(params['s2'].flatten)
    if get_gradients:
        grad = dict()
        grad['s2'] = np.ones_like(params['s2'])
        return K, grad
    else:
        return K


@jit(cache=True)
def se_iso(x, xp, params, get_gradients=False):
    """
    compute a (isometric) squared exponential kernel
    :param x: first kernel input
    :param xp: second kernel input
    :param params: dictionary with the kernel parameters s2 and ell2
    :param get_gradients: option to yield gradients
    :return:
    """
    norm_sq = spdist.cdist(x, xp) ** 2

    K = np.exp(-0.5 * norm_sq / params['ell2'])
    if get_gradients:
        grad = dict()
        grad['s2'] = K
        grad['ell2'] = 0.5 * params['s2'] * norm_sq * K / (params['ell2'] ** 2)
        return params['s2'] * K, grad
    else:
        return params['s2'] * K


@jit(cache=True)
def pe_iso(x, xp, params, get_gradients=False):
    """
    compute a (isometric) periodic kernel
    :param x: first kernel input
    :param xp: second kernel input
    :param params: dictionary with the kernel parameters s2 and ell2
    :param get_gradients: option to yield gradients
    :return:
    """
    dist = spdist.cdist(x, xp)
    norm = 0.5 - 0.5 * np.cos(2. * dist / params['per']) ** 2

    K = np.exp(- 0.5 * norm / params['ell2'])
    if get_gradients:
        grad = dict()
        grad['s2'] = K
        grad['ell2'] = params['s2'] * norm * K / (params['ell2'] ** 2)
        grad['per'] = params['s2'] * norm * K / (2. * params['ell2'] * params['per'] ** 2) * np.sin(2. * dist /
                                                                                                    params['per'])
        return params['s2'] * K, grad
    else:
        return params['s2'] * K


@jit(cache=True)
def se_ard(x, xp, params, get_gradients=False):
    """
    compute an automatic relevance determination squared exponential kernel
    :param x: first kernel input
    :param xp: second kernel input
    :param params: dictionary with the kernel parameters s2 and ell2
    :param get_gradients: option to yield gradients
    :return:
    """
    norm_sq = spdist.cdist(x, xp) ** 2
    K = np.exp(- 0.5 * norm_sq * np.diag(1 / params['ell2'].flatten()))

    if get_gradients:
        grad = dict()
        grad['s2'] = K
        grad['ell2'] = 0.5 * params['s2'] * norm_sq * K * np.diag(1 / params['ell2'].flatten() ** 2)
        return params['s2'] * K, grad
    else:
        return params['s2'] * K


@jit(cache=True)
def rq_iso(x, xp, params, get_gradients=False):
    """
    compute a (isometric) rational quadratic kernel
    :param x: first kernel input
    :param xp: second kernel input
    :param params: dictionary with the kernel parameters alpha, ell2 and s2
    :param get_gradients: option to yield gradients
    :return:
    """
    norm_sq = spdist.cdist(x, xp) ** 2
    aux = (1 + norm_sq / (2. * params['alpha'] * params['ell2']))
    K = aux ** (-params['alpha'])

    if get_gradients:
        grad = dict()
        grad['s2'] = K
        grad['ell2'] = -params['s2'] * aux ** (-params['alpha'] - 1) / (2 * params['ell2'] ** 2)
        grad['alpha'] = K * (-np.log(aux) + norm_sq / (2 * params['alpha'] * params['ell'] * aux))
        return params['s2'] * K, grad
    else:
        return params['s2'] * K


@jit(cache=True)
def rq_ard(x, xp, params, get_gradients=False):
    """
    compute an automatic relevance detection rational quadratic kernel
    :param x: first kernel input
    :param xp: second kernel input
    :param params: dictionary with the kernel parameters alpha, ell2 and s2
    :param get_gradients: option to yield gradients
    :return:
    """
    norm_sq = spdist.cdist(x, xp) ** 2
    aux = (1 + norm_sq / (2. * params['alpha']) * np.diag(1 / params['ell2']).flatten())
    K = aux ** (-params['alpha'])

    if get_gradients:
        grad = dict()
        grad['s2'] = K
        grad['ell2'] = -params['s2'] * aux ** (-params['alpha'] - 1) / 2 * np.diag(params['ell2'].flatten() ** -2)
        grad['alpha'] = K * (-np.log(aux) + norm_sq / (2 * params['alpha'] * params['ell'] * aux))
        return params['s2'] * K, grad
    else:
        return params['s2'] * K
