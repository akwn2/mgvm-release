# fast_gvm
# Just-in-time compiled fast GvM moment calculations using trapezoidal integration.
#
# 5 point-quadrature for an interval [a, b]
# def _quad5(b, a, f):
#
#     d = 0.5 * (b - a)
#     c = 0.5 * (b + a)
#
#     # Apply quadrature at collocation points
#     w1f1 = 0.23692688505618908 * f(-0.90617984593866396 * d + c)
#     w2f2 = 0.47862867049936647 * f(-0.53846931010568311 * d + c)
#     w3f3 = 0.50196078431372550 * f(c)
#     w4f4 = 0.47862867049936647 * f(+0.53846931010568311 * d + c)
#     w5f5 = 0.23692688505618908 * f(+0.90617984593866396 * d + c)
#
#     # sum and scale
#     return 0.5 * (b - a) * (w1f1 + w2f2 + w3f3 + w4f4 + w5f5)
import cmath
import math

import numba
import numpy as np


# Python-ready C-compiled functions
# -----------------------------------------
@numba.jit(nopython=True, cache=True)
def get_trig_mom(ord, th, k1, k2, m1, m2):
    """
    calculates the trigonometric moments of the mgvm using trapezoidal integration
    :param ord: moment order (integer)
    :param th: Grid where the integration will be carried out (1D-array)
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param m1: First location parameter
    :param m2: Second location parameter
    :return: Matrix of real and imaginary moments (double-precision complex number)
    """
    N = th.shape[0] - 1
    R, C = k1.shape
    T_re = k1 + k2
    T_im = k1 + k2

    for rr in range(0, R):
        for cc in range(0, C):
            # Sum ends of the integration interval
            aux_0 = math.exp(k1[rr, cc] * math.cos(th[0] - m1[rr, cc]) +
                             k2[rr, cc] * math.cos(2 * (th[0] - m2[rr, cc])) -
                             k1[rr, cc] - k2[rr, cc])
            aux_N = math.exp(k1[rr, cc] * math.cos(th[N] - m1[rr, cc]) +
                             k2[rr, cc] * math.cos(2 * (th[N] - m2[rr, cc])) -
                             k1[rr, cc] - k2[rr, cc])

            aux_zz = 0.5 * (aux_0 + aux_N)
            aux_re = 0.5 * (math.cos(ord * th[0]) * aux_0 + math.cos(ord * th[N]) * aux_N)
            aux_im = 0.5 * (math.sin(ord * th[0]) * aux_0 + math.sin(ord * th[N]) * aux_N)

            # Sum all inner point evaluation
            for jj in range(1, N - 1):
                aux_jj = math.exp(k1[rr, cc] * math.cos(th[jj] - m1[rr, cc]) +
                                  k2[rr, cc] * math.cos(2 * (th[jj] - m2[rr, cc])) -
                                  k1[rr, cc] - k2[rr, cc])

                aux_zz += aux_jj
                aux_re += math.cos(ord * th[jj]) * aux_jj
                aux_im += math.sin(ord * th[jj]) * aux_jj

            # We can omit the rescaling:
            #     -> aux_zz *= 2. * math.pi / N
            #     -> aux_re *= 2. * math.pi / N
            #     -> aux_im *= 2. * math.pi / N
            # because these factors are all equal, and cancel in the division.
            #
            # The same principle is valid for the exponential weighting term exp(-k1 - k2).

            T_re[rr, cc] = aux_re / aux_zz
            T_im[rr, cc] = aux_im / aux_zz

    return T_re + 1.j * T_im


# Python wrappers for C-compiled functions
# -----------------------------------------

def get_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2):
    """
    Log normalizer of the GvM with additional constraints
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter cosine
    :param sin_m1: First location parameter sine
    :param cos_2m2: Second location parameter cosine
    :param sin_2m2: Second location parameter sine
    :return:
    """
    # if method == 'trapz':
    #    return _log_z_lik_trapz(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2)
    # elif method == 'euler':
    return _log_z_lik_euler(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2)


def get_grad_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2):
    """
    python wrapper for the underlying c-compiled function
    gets the derivative of a trigonometric moment of a GvM with k1, k2, m1 and m2 using the compiled functions
    calculates the trigonometric moments of the mgvm using trapezoidal integration
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter
    :param sin_m1: First location parameter
    :param cos_2m2: Second location parameter
    :param sin_2m2: Second location parameter
    :return: tuple with:
        dlogz_k1: Gradient of first concentration parameter (array where the answer will be stored)
        dlogz_k2: Gradient of second concentration parameter (array where the answer will be stored)
        dlogz_cos_m1: Gradient of first location parameter cos (array where the answer will be stored)
        dlogz_sin_m1: Gradient of first location parameter sin (array where the answer will be stored)
        dlogz_cos_2m2: Gradient of second location parameter cos (array where the answer will be stored)
        dlogz_sin_2m2: Gradient of second location parameter sin (array where the answer will be stored)
        dlogz_psi: Gradient of the angular variable psi
    """
    # Allocate arrays to be passed to compiled function
    R, C = cos_m1.shape
    grad_log_z = np.zeros([R, C, 7], dtype=np.complex128)

    # Call c-compiled function
    # if method == 'euler':
    grad_log_z = _grad_log_z_lik_euler(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, grad_log_z)

    dlogz_k1 = grad_log_z[:, :, 0].real
    dlogz_k2 = grad_log_z[:, :, 1].real
    dlogz_cos_m1 = grad_log_z[:, :, 2].real
    dlogz_sin_m1 = grad_log_z[:, :, 3].real
    dlogz_cos_2m2 = grad_log_z[:, :, 4].real
    dlogz_sin_2m2 = grad_log_z[:, :, 5].real
    dlogz_psi = grad_log_z[:, :, 6].real

    return dlogz_k1, dlogz_k2, dlogz_cos_m1, dlogz_sin_m1, dlogz_cos_2m2, dlogz_sin_2m2, dlogz_psi


@numba.jit(nopython=False, cache=True)
def get_grad_u_trig_m(ord, th, k1, k2, m1, m2):
    """
    wrapper for the c-compiled function
    gets the derivative of a trigonometric moment of a GvM with k1, k2, m1 and m2 using the compiled functions
    calculates the trigonometric moments of the mgvm using trapezoidal integration
    :param ord: moment order (integer)
    :param th: Grid where the integration will be carried out (1D-array)
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param m1: First location parameter
    :param m2: Second location parameter
    :return tuple with:
        dfk1: Gradient of first concentration parameter (array where the answer will be stored)
        dfk2: Gradient of second concentration parameter (array where the answer will be stored)
        dfm1: Gradient of first location parameter (array where the answer will be stored)
        dfm2: Gradient of second location parameter (array where the answer will be stored)
    """
    dfk1 = np.zeros(k1.shape, dtype=np.complex128)
    dfk2 = np.zeros(k2.shape, dtype=np.complex128)
    dfm1 = np.zeros(m1.shape, dtype=np.complex128)
    dfm2 = np.zeros(m2.shape, dtype=np.complex128)

    _get_grad_u_trig_m(ord, th, k1, k2, m1, m2, dfk1, dfk2, dfm1, dfm2)

    return dfk1, dfk2, dfm1, dfm2


# C-compiled functions
# -----------------------------------------
@numba.jit(nopython=True, cache=True)
def _get_grad_u_trig_m(ord, th, k1, k2, m1, m2, dfk1, dfk2, dfm1, dfm2):
    """
    c-compiled function
    gets the derivative of a trigonometric moment of a GvM with k1, k2, m1 and m2 using the compiled functions
    calculates the trigonometric moments of the mgvm using trapezoidal integration
    :param ord: moment order (integer)
    :param th: Grid where the integration will be carried out (1D-array)
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param m1: First location parameter
    :param m2: Second location parameter
    :param dfk1: Gradient of first concentration parameter (array where the answer will be stored)
    :param dfk2: Gradient of second concentration parameter (array where the answer will be stored)
    :param dfm1: Gradient of first location parameter (array where the answer will be stored)
    :param dfm2: Gradient of second location parameter (array where the answer will be stored)
    """

    N = th.shape[0] - 1
    R, C = k1.shape

    ord_m2 = ord - 2
    ord_m1 = ord - 1
    ord_p1 = ord + 1
    ord_p2 = ord + 2

    # Allocate these complex arrays
    exp_m2jm2 = dfm1 + dfm2
    exp_m1jm1 = dfm1 + dfm2
    exp_p1jm1 = dfm1 + dfm2
    exp_p2jm2 = dfm1 + dfm2

    for rr in range(0, R):
        for cc in range(0, C):
            # Sum ends of the integration interval
            aux_0 = math.exp(k1[rr, cc] * math.cos(th[0] - m1[rr, cc]) +
                             k2[rr, cc] * math.cos(2 * (th[0] - m2[rr, cc])) -
                             k1[rr, cc] - k2[rr, cc])
            aux_N = math.exp(k1[rr, cc] * math.cos(th[N] - m1[rr, cc]) +
                             k2[rr, cc] * math.cos(2 * (th[N] - m2[rr, cc])) -
                             k1[rr, cc] - k2[rr, cc])

            aux_ord_m2 = 0.5 * (cmath.exp(1.j * ord_m2 * th[0]) * aux_0 + cmath.exp(1.j * ord_m2 * th[N]) * aux_N)
            aux_ord_m1 = 0.5 * (cmath.exp(1.j * ord_m1 * th[0]) * aux_0 + cmath.exp(1.j * ord_m1 * th[N]) * aux_N)
            aux_ord = 0.5 * (cmath.exp(1.j * ord * th[0]) * aux_0 + cmath.exp(1.j * ord * th[N]) * aux_N)
            aux_ord_p1 = 0.5 * (cmath.exp(1.j * ord_p1 * th[0]) * aux_0 + cmath.exp(1.j * ord_p1 * th[N]) * aux_N)
            aux_ord_p2 = 0.5 * (cmath.exp(1.j * ord_p2 * th[0]) * aux_0 + cmath.exp(1.j * ord_p2 * th[N]) * aux_N)

            # Sum all inner point evaluation
            for jj in range(1, N - 1):
                aux_jj = math.exp(k1[rr, cc] * math.cos(th[jj] - m1[rr, cc]) +
                                  k2[rr, cc] * math.cos(2 * (th[jj] - m2[rr, cc])) -
                                  k1[rr, cc] - k2[rr, cc])

                aux_ord_m2 += cmath.exp(1.j * ord_m2 * th[jj]) * aux_jj
                aux_ord_m1 += cmath.exp(1.j * ord_m1 * th[jj]) * aux_jj
                aux_ord += cmath.exp(1.j * ord * th[jj]) * aux_jj
                aux_ord_p1 += cmath.exp(1.j * ord_p1 * th[jj]) * aux_jj
                aux_ord_p2 += cmath.exp(1.j * ord_p2 * th[jj]) * aux_jj

            exp_m2jm2[rr, cc] = cmath.exp(-2.j * m2[rr, cc])
            exp_m1jm1[rr, cc] = cmath.exp(-1.j * m1[rr, cc])
            exp_p1jm1[rr, cc] = cmath.exp(+1.j * m1[rr, cc])
            exp_p2jm2[rr, cc] = cmath.exp(+2.j * m2[rr, cc])

    dfk1 = 0.5 * (exp_m1jm1 * aux_ord_p1 + exp_p1jm1 * aux_ord_m1) - aux_ord
    dfk2 = 0.5 * (exp_m2jm2 * aux_ord_p2 + exp_p2jm2 * aux_ord_m2) - aux_ord
    dfm1 = -k1 * 0.5j * (exp_m1jm1 * aux_ord_p1 - exp_p1jm1 * aux_ord_m1)
    dfm2 = -k2 * 1.0j * (exp_m2jm2 * aux_ord_p2 - exp_p2jm2 * aux_ord_m2)


@numba.jit(nopython=True, cache=True)
def _log_z_lik_trapz(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2):
    """
    Log normalizer of the GvM by trapezoidal integration
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter cosine
    :param sin_m1: First location parameter sine
    :param cos_2m2: Second location parameter cosine
    :param sin_2m2: Second location parameter sine
    :return:
    """
    N = 1000
    delta = 2 * math.pi / N
    R, C = k1.shape
    T = k1 + k2

    for rr in range(0, R):
        for cc in range(0, C):
            w = k1[rr, cc] + k2[rr, cc]

            # Sum ends of the integration interval
            aux = 0.5 * (math.exp(-k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] - w) +
                         math.exp(-k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] - w))

            # Sum all inner point evaluations
            for jj in range(1, N - 1):
                aux += math.exp(k1[rr, cc] * (math.cos(jj * delta - math.pi) * cos_m1[rr, cc] +
                                              math.sin(jj * delta - math.pi) * sin_m1[rr, cc]) +
                                k2[rr, cc] * (math.cos(2 * (jj * delta - math.pi)) * cos_2m2[rr, cc] +
                                              math.sin(2 * (jj * delta - math.pi)) * sin_2m2[rr, cc]) - w)

            T[rr, cc] += math.log(2. * math.pi / N * aux)

    return T


@numba.jit(nopython=True, cache=True)
def _grad_log_z_lik_trapz(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, grad_log_z):
    """
    underlying c-compiled function
    gets the derivative of a trigonometric moment of a GvM with k1, k2, m1 and m2 using the compiled functions
    calculates the trigonometric moments of the mgvm using trapezoidal integration
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter
    :param sin_m1: First location parameter
    :param cos_2m2: Second location parameter
    :param sin_2m2: Second location parameter
    :param grad_log_z: vector where we store
        dlogz_k1: Gradient of first concentration parameter (array where the answer will be stored)
        dlogz_k2: Gradient of second concentration parameter (array where the answer will be stored)
        dlogz_cos_m1: Gradient of first location parameter cos (array where the answer will be stored)
        dlogz_sin_m1: Gradient of first location parameter sin (array where the answer will be stored)
        dlogz_cos_2m2: Gradient of second location parameter cos (array where the answer will be stored)
        dlogz_sin_2m2: Gradient of second location parameter sin (array where the answer will be stored)
    """

    N = 1000
    delta = 2 * math.pi / N
    R, C = k1.shape

    for rr in range(0, R):
        for cc in range(0, C):
            w = k1[rr, cc] + k2[rr, cc]

            # Sum ends of the integration interval
            aux_0 = math.exp(- k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] - w)
            aux_N = math.exp(- k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] - w)

            z_0 = 0.5 * (aux_0 + aux_N)
            z_1 = 0.5 * (cmath.exp(+1.j * math.pi) * aux_0 + cmath.exp(+1.j * 2 * math.pi) * aux_N)
            z_2 = 0.5 * (cmath.exp(+2.j * math.pi) * aux_0 + cmath.exp(+2.j * 2 * math.pi) * aux_N)

            # Sum all inner point evaluation
            for jj in range(1, N - 1):
                aux_jj = math.exp(k1[rr, cc] * (math.cos(jj * delta - math.pi) * cos_m1[rr, cc] +
                                                math.sin(jj * delta - math.pi) * sin_m1[rr, cc]) +
                                  k2[rr, cc] * (math.cos(2 * (jj * delta - math.pi)) * cos_2m2[rr, cc] +
                                                math.sin(2 * (jj * delta - math.pi)) * sin_2m2[rr, cc]) - w)

                z_0 += aux_jj
                z_1 += cmath.exp(+1.j * (jj * delta - math.pi)) * aux_jj
                z_2 += cmath.exp(+2.j * (jj * delta - math.pi)) * aux_jj

            mc_psi = z_1.real / z_0
            ms_psi = z_1.imag / z_0
            mc_2psi = z_2.real / z_0
            ms_2psi = z_2.imag / z_0

            grad_log_z[rr, cc, 0] = cos_m1[rr, cc] * mc_psi + sin_m1[rr, cc] * ms_psi
            grad_log_z[rr, cc, 1] = cos_2m2[rr, cc] * mc_2psi + sin_2m2[rr, cc] * ms_2psi
            grad_log_z[rr, cc, 2] = k1[rr, cc] * mc_psi
            grad_log_z[rr, cc, 3] = k1[rr, cc] * ms_psi
            grad_log_z[rr, cc, 4] = k2[rr, cc] * mc_2psi
            grad_log_z[rr, cc, 5] = k2[rr, cc] * ms_2psi

    return grad_log_z


@numba.jit(nopython=True, cache=True)
def _log_z_lik_euler(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2):
    """
    Log normalizer of the GvM by forward euler integration
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter cosine
    :param sin_m1: First location parameter sine
    :param cos_2m2: Second location parameter cosine
    :param sin_2m2: Second location parameter sine
    :return:
    """
    N = 1000
    delta = 2 * math.pi / N
    R, C = k1.shape
    T = k1 + k2

    for rr in range(0, R):
        for cc in range(0, C):
            w = k1[rr, cc] + k2[rr, cc]
            aux = 0.
            # Sum all inner point evaluations
            for jj in range(0, N):
                aux += math.exp(k1[rr, cc] * (math.cos(jj * delta - math.pi) * cos_m1[rr, cc] +
                                              math.sin(jj * delta - math.pi) * sin_m1[rr, cc]) +
                                k2[rr, cc] * (math.cos(2 * (jj * delta - math.pi)) * cos_2m2[rr, cc] +
                                              math.sin(2 * (jj * delta - math.pi)) * sin_2m2[rr, cc]) - w)

            T[rr, cc] += math.log(delta * aux)

    return T


@numba.jit(nopython=True, cache=True)
def _grad_log_z_lik_euler(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, grad_log_z):
    """
    underlying c-compiled function
    gets the derivative of a trigonometric moment of a GvM with k1, k2, m1 and m2 using the compiled functions
    calculates the trigonometric moments of the mgvm using forward euler
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter
    :param sin_m1: First location parameter
    :param cos_2m2: Second location parameter
    :param sin_2m2: Second location parameter
    :param grad_log_z: vector where we store
        dlogz_k1: Gradient of first concentration parameter (array where the answer will be stored)
        dlogz_k2: Gradient of second concentration parameter (array where the answer will be stored)
        dlogz_cos_m1: Gradient of first location parameter cos (array where the answer will be stored)
        dlogz_sin_m1: Gradient of first location parameter sin (array where the answer will be stored)
        dlogz_cos_2m2: Gradient of second location parameter cos (array where the answer will be stored)
        dlogz_sin_2m2: Gradient of second location parameter sin (array where the answer will be stored)
        dlogz_psi_p: Gradient of predicted entries psi
    """

    N = 1000
    delta = 2 * math.pi / N
    R, C = k1.shape

    for rr in range(0, R):
        for cc in range(0, C):
            w = k1[rr, cc] + k2[rr, cc]
            z_0 = 0.
            z_1 = 0.
            z_2 = 0.
            # Sum all inner point evaluation
            for jj in range(0, N):
                aux_jj = math.exp(k1[rr, cc] * (math.cos(jj * delta - math.pi) * cos_m1[rr, cc] +
                                                math.sin(jj * delta - math.pi) * sin_m1[rr, cc]) +
                                  k2[rr, cc] * (math.cos(2 * (jj * delta - math.pi)) * cos_2m2[rr, cc] +
                                                math.sin(2 * (jj * delta - math.pi)) * sin_2m2[rr, cc]) - w)

                z_0 += aux_jj
                z_1 += cmath.exp(+1.j * (jj * delta - math.pi)) * aux_jj
                z_2 += cmath.exp(+2.j * (jj * delta - math.pi)) * aux_jj

            mc_psi = z_1.real / z_0
            ms_psi = z_1.imag / z_0
            mc_2psi = z_2.real / z_0
            ms_2psi = z_2.imag / z_0

            grad_log_z[rr, cc, 0] = cos_m1[rr, cc] * mc_psi + sin_m1[rr, cc] * ms_psi
            grad_log_z[rr, cc, 1] = cos_2m2[rr, cc] * mc_2psi + sin_2m2[rr, cc] * ms_2psi
            grad_log_z[rr, cc, 2] = k1[rr, cc] * mc_psi
            grad_log_z[rr, cc, 3] = k1[rr, cc] * ms_psi
            grad_log_z[rr, cc, 4] = k2[rr, cc] * mc_2psi
            grad_log_z[rr, cc, 5] = k2[rr, cc] * ms_2psi
            grad_log_z[rr, cc, 6] = k1[rr, cc] * (sin_m1[rr, cc] * mc_psi - cos_m1[rr, cc] * ms_psi) + \
                                    2. * k2[rr, cc] * (sin_2m2[rr, cc] * mc_2psi - cos_2m2[rr, cc] * ms_2psi)

    return grad_log_z


@numba.jit(nopython=True, cache=True)
def _log_z_lik_quad5(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2):
    """
    Log normalizer of the GvM by 5-point quadrature integration
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter cosine
    :param sin_m1: First location parameter sine
    :param cos_2m2: Second location parameter cosine
    :param sin_2m2: Second location parameter sine
    :return:
    """
    N = 1000
    delta = 2 * math.pi / N
    R, C = k1.shape
    T = k1 + k2

    for rr in range(0, R):
        for cc in range(0, C):
            aux = 0.
            for jj in range(0, N - 1):
                a = jj * delta - math.pi
                b = (jj + 1) * delta - math.pi

                d = 0.5 * (b - a)
                c = 0.5 * (b + a)

                # Collocation points (Gauss-Legendre)
                x1 = -0.90617984593866396 * d + c
                x2 = -0.53846931010568311 * d + c
                x3 = c
                x4 = +0.53846931010568311 * d + c
                x5 = +0.90617984593866396 * d + c

                # Associated weights
                w1 = 0.2369268850561890
                w2 = 0.4786286704993665
                w3 = 0.5019607843137255
                w4 = 0.4786286704993665
                w5 = 0.2369268850561890

                # # Collocation points (Gauss-Lobatto)
                # x1 = -d + c
                # x2 = -0.65465367070797709 * d + c
                # x3 = c
                # x4 = +0.65465367070797709 * d + c
                # x5 = d + c
                #
                # # Associated weights
                # w1 = 0.10
                # w2 = 0.5444444444444444
                # w3 = 0.71111111111111111
                # w4 = 0.5444444444444444
                # w5 = 0.10


                # function @ collocation points
                f1 = math.exp(k1[rr, cc] * (math.cos(x1) * cos_m1[rr, cc] + math.sin(x1) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x1) * cos_2m2[rr, cc] + math.sin(x1) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f2 = math.exp(k1[rr, cc] * (math.cos(x2) * cos_m1[rr, cc] + math.sin(x2) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x2) * cos_2m2[rr, cc] + math.sin(x2) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f3 = math.exp(k1[rr, cc] * (math.cos(x3) * cos_m1[rr, cc] + math.sin(x3) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x3) * cos_2m2[rr, cc] + math.sin(x3) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f4 = math.exp(k1[rr, cc] * (math.cos(x4) * cos_m1[rr, cc] + math.sin(x4) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x4) * cos_2m2[rr, cc] + math.sin(x4) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f5 = math.exp(k1[rr, cc] * (math.cos(x5) * cos_m1[rr, cc] + math.sin(x5) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x5) * cos_2m2[rr, cc] + math.sin(x5) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                # Sum and scale
                aux += d * (w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5)

            T[rr, cc] += math.log(aux)

    return T


@numba.jit(nopython=True, cache=True)
def _grad_log_z_lik_quad5(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, grad_log_z):
    """
    underlying c-compiled function
    gets the derivative of a trigonometric moment of a GvM with k1, k2, m1 and m2 using the compiled functions
    calculates the trigonometric moments of the mgvm using 5-point quadrature
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter
    :param sin_m1: First location parameter
    :param cos_2m2: Second location parameter
    :param sin_2m2: Second location parameter
    :param grad_log_z: vector where we store
        dlogz_k1: Gradient of first concentration parameter (array where the answer will be stored)
        dlogz_k2: Gradient of second concentration parameter (array where the answer will be stored)
        dlogz_cos_m1: Gradient of first location parameter cos (array where the answer will be stored)
        dlogz_sin_m1: Gradient of first location parameter sin (array where the answer will be stored)
        dlogz_cos_2m2: Gradient of second location parameter cos (array where the answer will be stored)
        dlogz_sin_2m2: Gradient of second location parameter sin (array where the answer will be stored)
    """

    N = 1000
    delta = 2 * math.pi / N
    R, C = k1.shape

    for rr in range(0, R):
        for cc in range(0, C):
            z_0 = 0.
            z_1 = 0.
            z_2 = 0.

            # Sum all inner point evaluation
            for jj in range(0, N - 1):
                a = jj * delta - math.pi
                b = (jj + 1) * delta - math.pi

                d = 0.5 * (b - a)
                c = 0.5 * (b + a)

                # Collocation points (Gauss-Legendre)
                x1 = -0.90617984593866396 * d + c
                x2 = -0.53846931010568311 * d + c
                x3 = c
                x4 = +0.53846931010568311 * d + c
                x5 = +0.90617984593866396 * d + c

                # Associated weights
                w1 = 0.2369268850561890
                w2 = 0.4786286704993665
                w3 = 0.5019607843137255
                w4 = 0.4786286704993665
                w5 = 0.2369268850561890

                # # Collocation points (Gauss-Lobatto)
                # x1 = -d + c
                # x2 = -0.65465367070797709 * d + c
                # x3 = c
                # x4 = +0.65465367070797709 * d + c
                # x5 = d + c
                #
                # # Associated weights
                # w1 = 0.10
                # w2 = 0.5444444444444444
                # w3 = 0.71111111111111111
                # w4 = 0.5444444444444444
                # w5 = 0.10


                # function @ collocation points
                f1 = math.exp(k1[rr, cc] * (math.cos(x1) * cos_m1[rr, cc] + math.sin(x1) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x1) * cos_2m2[rr, cc] + math.sin(x1) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f2 = math.exp(k1[rr, cc] * (math.cos(x2) * cos_m1[rr, cc] + math.sin(x2) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x2) * cos_2m2[rr, cc] + math.sin(x2) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f3 = math.exp(k1[rr, cc] * (math.cos(x3) * cos_m1[rr, cc] + math.sin(x3) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x3) * cos_2m2[rr, cc] + math.sin(x3) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f4 = math.exp(k1[rr, cc] * (math.cos(x4) * cos_m1[rr, cc] + math.sin(x4) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x4) * cos_2m2[rr, cc] + math.sin(x4) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                f5 = math.exp(k1[rr, cc] * (math.cos(x5) * cos_m1[rr, cc] + math.sin(x5) * sin_m1[rr, cc]) +
                              k2[rr, cc] * (math.cos(x5) * cos_2m2[rr, cc] + math.sin(x5) * sin_2m2[rr, cc]) -
                              k1[rr, cc] - k2[rr, cc])

                # Sum and scale
                z_0 += d * (w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5)

                z_1 += d * (w1 * f1 * cmath.exp(+1.j * x1) +
                            w2 * f2 * cmath.exp(+1.j * x2) +
                            w3 * f3 * cmath.exp(+1.j * x3) +
                            w4 * f4 * cmath.exp(+1.j * x4) +
                            w5 * f5 * cmath.exp(+1.j * x5))

                z_2 += d * (w1 * f1 * cmath.exp(+2.j * x1) +
                            w2 * f2 * cmath.exp(+2.j * x2) +
                            w3 * f3 * cmath.exp(+2.j * x3) +
                            w4 * f4 * cmath.exp(+2.j * x4) +
                            w5 * f5 * cmath.exp(+2.j * x5))

            grad_log_z[rr, cc, 0] = (cos_m1[rr, cc] * z_1.real + sin_m1[rr, cc] * z_1.imag) / z_0
            grad_log_z[rr, cc, 1] = (cos_2m2[rr, cc] * z_2.real + sin_2m2[rr, cc] * z_2.imag) / z_0
            grad_log_z[rr, cc, 2] = (k1[rr, cc] * z_1.real) / z_0
            grad_log_z[rr, cc, 3] = (k1[rr, cc] * z_1.imag) / z_0
            grad_log_z[rr, cc, 4] = (k2[rr, cc] * z_2.real) / z_0
            grad_log_z[rr, cc, 5] = (k2[rr, cc] * z_2.imag) / z_0

    return grad_log_z


@numba.jit(nopython=True, cache=True)
def _log_z_lik_gt(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2):
    """
    Log normalizer of the GvM by forward euler integration with very small step sizes (to serve as ground truth)
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter cosine
    :param sin_m1: First location parameter sine
    :param cos_2m2: Second location parameter cosine
    :param sin_2m2: Second location parameter sine
    :return:
    """
    N = 1000000
    delta = 2 * math.pi / N
    R, C = k1.shape
    T = k1 + k2

    for rr in range(0, R):
        for cc in range(0, C):
            w = k1[rr, cc] + k2[rr, cc]
            aux = 0.
            # Sum all inner point evaluations
            for jj in range(0, N):
                aux += math.exp(k1[rr, cc] * (math.cos(jj * delta - math.pi) * cos_m1[rr, cc] +
                                              math.sin(jj * delta - math.pi) * sin_m1[rr, cc]) +
                                k2[rr, cc] * (math.cos(2 * (jj * delta - math.pi)) * cos_2m2[rr, cc] +
                                              math.sin(2 * (jj * delta - math.pi)) * sin_2m2[rr, cc]) - w)

            T[rr, cc] += math.log(2. * math.pi / N * aux)

    return T
