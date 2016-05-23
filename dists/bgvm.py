import numpy as np
from scipy.integrate import dblquad


class BGvM(object):
    def __init__(self, k1, m1, W):
        """
        Class for the bivariate Generalised von Mises distribution
        :param k1: (linear) concentration parameter
        :param m1: (linear) mean parameter
        :param W: (overparametrised) covariance matrix
        :return:
        """
        self.k1 = k1
        self.m1 = m1
        self.W = W

    def __f_re(self, x1, x2, n, idx):
        """
        Unnormalised real part of the moment integrand function to be used when obtaining the moments numerically.
        :param x1:
        :param x2:
        :param n:
        :param idx:
        :return:
        """
        return ((1 - idx) * np.cos(n * x1) + idx * np.cos(n * x2)) * \
               np.exp(self.k1[0] * np.cos(x1 - self.m1[0]) + self.k1[1] * np.cos(x2 - self.m1[1]) -
                      0.5 * (x1 ** 2 * self.W[0, 0] + x2 ** 2 * self.W[1, 1] - 2. * x1 * x2 * self.W[0, 1]))

    def __f_im(self, x1, x2, n, idx):
        """
        Unnormalised imaginary part of the moment integrand function to be used when obtaining the moments numerically.
        :param x1:
        :param x2:
        :param n:
        :param idx:
        :return:
        """
        return ((1 - idx) * np.sin(n * x1) + idx * np.sin(n * x2)) * \
               np.exp(self.k1[0] * np.cos(x1 - self.m1[0]) + self.k1[1] * np.cos(x2 - self.m1[1]) -
                      0.5 * (x1 ** 2 * self.W[0, 0] + x2 ** 2 * self.W[1, 1] - 2. * x1 * x2 * self.W[0, 1]))

    def moment(self, idx, n=0, z=None):
        """
        Calculates bGvM moment of order n by numerical integration (quadrature)
        :param idx: variable index (either 0 or 1)
        :param n: moment order
        :return:
        """
        if z is None:
            z = dblquad(lambda x1, x2: self.__f_re(x1, x2, 0, idx), -np.pi, np.pi, lambda x: -np.pi, lambda x: np.pi)[0]

        if n == 0:
            return z
        else:
            m_re = dblquad(lambda x1, x2: self.__f_re(x1, x2, n, idx),
                           -np.pi, np.pi, lambda x: -np.pi, lambda x: np.pi)[0]

            m_im = dblquad(lambda x1, x2: self.__f_im(x1, x2, n, idx),
                           -np.pi, np.pi, lambda x: -np.pi, lambda x: np.pi)[0]

            return (m_re + 1.j * m_im) / z
