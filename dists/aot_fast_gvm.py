# source_fast_gvm
# Source code for compiled fast GvM moment calculations using trapezoidal integration.
import cmath
import math

from numba.pycc import CC

cc = CC('fast_gvm')
cc.verbose = True


# Syntax example for compiling with void returns
# @cc.export('func', 'void(f8[:,:,:], f8[:,:,:])')
# def func(input, output):
#     output = input ** 2

@cc.export('get_log_z', 'f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8[:, :])')
def get_log_z(k1, k2, m1, m2):
    """
    Log normalizer of the GvM with additional constraints
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param m1: First location parameter
    :param m2: Second location parameter
    :return:
    """
    N = 100
    delta = 2 * math.pi / N
    R, C = k1.shape
    T = k1 + k2

    for rr in range(0, R):
        for cc in range(0, C):
            # Sum ends of the integration interval
            aux = 0.5 * (math.exp(k1[rr, cc] * math.cos(-math.pi - m1[rr, cc]) +
                                  k2[rr, cc] * math.cos(2 * (-math.pi - m2[rr, cc])) - k1[rr, cc] - k2[rr, cc]) +
                         math.exp(k1[rr, cc] * math.cos(math.pi - m1[rr, cc]) +
                                  k2[rr, cc] * math.cos(2 * (math.pi - m2[rr, cc])) - k1[rr, cc] - k2[rr, cc]))

            # Sum all inner point evaluations
            for jj in range(1, N - 1):
                aux += math.exp(k1[rr, cc] * math.cos(jj * delta - math.pi - m1[rr, cc]) +
                                k2[rr, cc] * math.cos(2 * (jj * delta - math.pi - m2[rr, cc])) -
                                k1[rr, cc] - k2[rr, cc])

            T[rr, cc] += math.log(2. * math.pi / N * aux)

    return T


@cc.export('get_log_z_lik', 'f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :])')
def get_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2):
    """
    Log normalizer of the GvM with additional constraints
    :param th: Grid where the integration will be carried out (1D-array)
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter cosine
    :param sin_m1: First location parameter sine
    :param cos_2m2: Second location parameter cosine
    :param sin_2m2: Second location parameter sine
    :return:
    """
    N = 100
    delta = 2 * math.pi / N
    R, C = k1.shape
    T = k1 + k2

    for rr in range(0, R):
        for cc in range(0, C):
            # Sum ends of the integration interval
            aux = 0.5 * (math.exp(-k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] -
                                  k1[rr, cc] - k2[rr, cc]) +
                         math.exp(-k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] -
                                  k1[rr, cc] - k2[rr, cc]))

            # Sum all inner point evaluations
            for jj in range(1, N - 1):
                aux += math.exp(k1[rr, cc] * (math.cos(jj * delta - math.pi) * cos_m1[rr, cc] +
                                              math.sin(jj * delta - math.pi) * sin_m1[rr, cc]) +
                                k2[rr, cc] * (math.cos(2 * (jj * delta - math.pi)) * cos_2m2[rr, cc] +
                                              math.sin(2 * (jj * delta - math.pi)) * sin_2m2[rr, cc]) -
                                k1[rr, cc] - k2[rr, cc])

            T[rr, cc] += math.log(2. * math.pi / N * aux)

    return T


@cc.export('get_grad_log_z_lik', 'void(f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :],' +
           'f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :], f8[:, :])')
def get_grad_log_z_lik(k1, k2, cos_m1, sin_m1, cos_2m2, sin_2m2, dlogz_k1, dlogz_k2, dlogz_cos_m1, dlogz_sin_m1,
                       dlogz_cos_2m2, dlogz_sin_2m2):
    """
    gets the derivative of a trigonometric moment of a GvM with k1, k2, m1 and m2 using the compiled functions
    calculates the trigonometric moments of the mgvm using trapezoidal integration
    :param th: Grid where the integration will be carried out (1D-array)
    :param k1: First concentration parameter
    :param k2: Second concentration parameter
    :param cos_m1: First location parameter
    :param sin_m1: First location parameter
    :param cos_2m2: Second location parameter
    :param sin_2m2: Second location parameter
    :param dlogz_k1: Gradient of first concentration parameter (array where the answer will be stored)
    :param dlogz_k2: Gradient of second concentration parameter (array where the answer will be stored)
    :param dlogz_cos_m1: Gradient of first location parameter cos (array where the answer will be stored)
    :param dlogz_sin_m1: Gradient of first location parameter sin (array where the answer will be stored)
    :param dlogz_cos_2m2: Gradient of second location parameter cos (array where the answer will be stored)
    :param dlogz_sin_2m2: Gradient of second location parameter sin (array where the answer will be stored)
    """

    N = 100
    delta = 2 * math.pi / N
    R, C = k1.shape

    for rr in range(0, R):
        for cc in range(0, C):
            # Sum ends of the integration interval
            aux_0 = math.exp(- k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] -
                             k1[rr, cc] - k2[rr, cc])
            aux_N = math.exp(- k1[rr, cc] * cos_m1[rr, cc] + k2[rr, cc] * cos_2m2[rr, cc] -
                             k1[rr, cc] - k2[rr, cc])

            z_0 = 0.5 * (aux_0 + aux_N)
            z_1 = 0.5 * (cmath.exp(+1.j * math.pi) * aux_0 + cmath.exp(+1.j * 2 * math.pi) * aux_N)
            z_2 = 0.5 * (cmath.exp(+2.j * math.pi) * aux_0 + cmath.exp(+2.j * 2 * math.pi) * aux_N)

            # Sum all inner point evaluation
            for jj in range(1, N - 1):
                aux_jj = math.exp(k1[rr, cc] * (math.cos(jj * delta - math.pi) * cos_m1[rr, cc] +
                                                math.sin(jj * delta - math.pi) * sin_m1[rr, cc]) +
                                  k2[rr, cc] * (math.cos(2 * (jj * delta - math.pi)) * cos_2m2[rr, cc] +
                                                math.sin(2 * (jj * delta - math.pi)) * sin_2m2[rr, cc]) -
                                  k1[rr, cc] - k2[rr, cc])

                z_0 += aux_jj
                z_1 += cmath.exp(+1.j * (jj * delta - math.pi)) * aux_jj
                z_2 += cmath.exp(+2.j * (jj * delta - math.pi)) * aux_jj

    dlogz_k1 = (cos_m1 * z_1.real + sin_m1 * z_1.imag) / z_0
    dlogz_k2 = (cos_2m2 * z_2.real + sin_2m2 * z_2.imag) / z_0
    dlogz_cos_m1 = (k1 * z_1.real) / z_0
    dlogz_sin_m1 = (k1 * z_1.imag) / z_0
    dlogz_cos_2m2 = (k2 * z_2.real) / z_0
    dlogz_sin_2m2 = (k2 * z_2.imag) / z_0


@cc.export('get_trig_mom', 'c16[:, :](i8, f8[:], f8[:, :], f8[:, :], f8[:, :], f8[:, :])')
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


@cc.export('get_grad_u_trig_m',
           'void(i8, f8[:], f8[:, :], f8[:, :], f8[:, :], f8[:, :], c16[:, :], c16[:, :], c16[:, :], c16[:, :])')
def get_grad_u_trig_m(ord, th, k1, k2, m1, m2, dfk1, dfk2, dfm1, dfm2):
    """
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


if __name__ == '__main__':
    cc.compile()
