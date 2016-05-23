"""
mutil.py
Matrix utility functions
"""
import numpy as np
import scipy as sp
import scipy.linalg as sla


def rchol(x, attempts=6):
    """
    robust cholesky
    :param x: the matrix to be factorised
    :param attempts: maximum number of attempts
    :return:
    """
    jitter = 0
    try:
        L = sla.cholesky(x, lower=True, check_finite=False)
        return L, jitter
    except sp.linalg.LinAlgError:
        a = 0
        jitter = 1e-6 * sp.mean(sp.diag(x))
        while a < attempts:
            try:
                L = sla.cholesky(x + jitter * sp.eye(x.shape[0]), lower=True, check_finite=False)
                return L, jitter
            except sp.linalg.LinAlgError:
                jitter *= 10.

    raise sp.linalg.LinAlgError


def solve_chol(L, A):
    """
    solve Lx = A
    :param L: Cholesky factor
    :param X: RHS matrix
    :return:
    """
    LiA = sla.solve_triangular(L, A, lower=True, check_finite=False)
    return LiA


def inv_mult_chol(L, X):
    """
    calculate dot(K_inv, X) for K_inv the inverse of K = L * L'
    :param L: cholesky factor of K
    :param X: matrix or vector to be multiplied by the inverse of K
    :return:
    """
    LiX = sla.solve_triangular(L, X, lower=True, check_finite=False)
    KiX = sla.solve_triangular(L.T, LiX, lower=False, check_finite=False)
    return KiX


def invc(L):
    """
    find the inverse of K = L * L' using cholesky factor L
    :param L: cholesky factor of K
    :return:
    """
    return inv_mult_chol(L, sp.eye(L.shape[0]))


def invx(X):
    """
    calculates the inverse for X and the determinant of X using SVD decomposition.
    This was used to sanity check the robust cholesky implementation.
    :param X:
    :return:
    """
    U, S, Vh = sp.linalg.svd(X)

    if min(S) < 1.e-6:
        delta = 1.e-6 - min(S)
        S = S + delta

    Sinv = sp.diag(1. / S)

    logdet = np.sum(np.log(S))

    return sp.dot(U, sp.dot(Sinv, Vh)), logdet


def logdetK(L):
    """
    numerical stable way to calculates the determinant of K using its cholesky factor L
    :param L: cholesky factor of K
    :return: log of determinant of K
    """
    return 2. * sum(np.log(np.diag(L)))


def s_abs(x):
    """
    smoothed absolute value
    :param x: input value
    :return:
    """
    c = 1.e-2
    return np.sqrt(x ** 2 + c)


def ds_abs(x):
    """
    derivative of the smoothed absolute value
    :param x: input value
    :return:
    """
    c = 1.e-2
    return x / np.sqrt(x ** 2 + c)


def blockm(A, B, C, D):
    """
    creates a block matrix [[A, B],[C, D]] as a 2D array
    :param A: upper left block
    :param B: upper right block
    :param C: lower left block
    :param D: lower right block
    :return:
    """

    return np.asarray(np.bmat([[A, B], [C, D]]))


def get_sym_blocks(K):
    """
    gets the N x N blocks of a 2N x 2N symmetric matrix K
    :param K: 2N x 2N matrix
    :return:
    """
    N = K.shape[0] / 2

    return K[0:N, 0:N], K[0:N, N:], K[N:, N:]
