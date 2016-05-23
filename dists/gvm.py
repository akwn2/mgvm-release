from utils.circular import *

class GvM:
    def __init__(self, k1, k2, m1, m2, method='grid', n_pts=10000, terms=100):
        """
        generalised von mises model
        :param k1: 1st concentration parameter
        :param k2: 2nd concentration parameter
        :param m1: 1st location parameter
        :param m2: 2nd location parameter
        :param terms: number of terms to use in the series calculations
        :param method: method to be used in series / calculations term
        :return:
        """
        self.R = k1.shape[0]
        self.C = k1.shape[1]
        self.N = k1.shape[0] * k1.shape[1]
        self.S = 1E5

        self.k1 = k1.reshape(self.N, 1)
        self.k2 = k2.reshape(self.N, 1)
        self.m1 = mod2pi(m1.reshape(self.N, 1))
        self.m2 = np.mod(m2.reshape(self.N, 1), np.pi)

        self.dm = np.mod(self.m1 - self.m2, np.pi)

        self.terms = terms
        self.method = method
        self.n_pts = n_pts

        # Moment-related trigonometric integrals and their derivatives
        self.T0 = None
        self.T1 = None
        self.T2 = None

        # Derivatives of the moments, the order used here is k1, k2, m1, m2
        self.dT0 = [None, None, None, None]
        self.dT1 = [None, None, None, None]
        self.dT2 = [None, None, None, None]

    def _qmc_z(self):
        S = int(1E5)

        Z = np.zeros([self.N, 1])
        th = np.linspace(0., 2. * np.pi, S)

        for ii in xrange(0, self.N):
            f = np.exp(self.k1[ii] * np.cos(th - self.m1[ii]) + self.k2[ii] * np.cos(2. * (th - self.m2[ii])))
            Z[ii] = (2. * np.pi) / S * np.sum(f)

        return Z.reshape([self.R, self.C])

    def _find_modes(self):
        k1 = np.reshape(self.k1, [self.N, 1])
        k2 = np.reshape(self.k2, [self.N, 1])
        m1 = np.reshape(self.m1, [self.N, 1])
        dm = np.reshape(self.dm, [self.N, 1])

        rho = k1 / (4. * k2)
        b0 = (np.sin(dm) * np.cos(dm)) ** 2
        b1 = 2. * rho * (np.sin(dm) * np.cos(dm))
        b2 = rho ** 2 - 1.
        b3 = - 4. * rho * np.sin(dm) * np.cos(dm)

        modes = list()
        for ii in xrange(0, self.N):
            aux0 = np.roots([1., b3[ii, 0], b2[ii, 0], b1[ii, 0], b0[ii, 0]])
            blim = aux0 >= -1
            ulim = aux0 <= +1
            aux0 = aux0[np.logical_and(blim, ulim)]
            modes.append(mod2pi(mod2pi(np.arcsin(aux0)) + m1))

        return modes

    def _grid_fun(self, r):
        th = np.reshape(np.linspace(-np.pi, np.pi, self.n_pts), (1, self.n_pts))
        delta = th[0, 1] - th[0, 0]

        th = np.repeat(th, self.R * self.C, 0)

        k1 = np.repeat(np.reshape(self.k1, (self.R * self.C, 1)), self.n_pts, 1)
        k2 = np.repeat(np.reshape(self.k2, (self.R * self.C, 1)), self.n_pts, 1)
        m1 = np.repeat(np.reshape(self.m1, (self.R * self.C, 1)), self.n_pts, 1)
        m2 = np.repeat(np.reshape(self.m2, (self.R * self.C, 1)), self.n_pts, 1)

        T = np.sum(np.exp(k1 * np.cos(th - m1) + k2 * np.cos(2 * (th - m2)) + 1.j * r * th - k1 - k2), 1) * delta

        return np.reshape(T, (self.R, self.C))

    def _grad_grid_fun(self, r):

        f_rm2 = self._grid_fun(r - 2)
        f_rm1 = self._grid_fun(r - 1)
        f_r = self._grid_fun(r)
        f_rp1 = self._grid_fun(r + 1)
        f_rp2 = self._grid_fun(r + 2)

        df_k1 = 0.5 * (np.exp(-1.j * self.m1) * f_rp1 + np.exp(+1.j * self.m1) * f_rm1) - f_r
        df_k2 = 0.5 * (np.exp(-2.j * self.m2) * f_rp2 + np.exp(+2.j * self.m2) * f_rm2) - f_r
        df_m1 = -self.k1 * 0.5j * (np.exp(-1.j * self.m1) * f_rp1 - np.exp(+1.j * self.m1) * f_rm1)
        df_m2 = -self.k2 * 1.0j * (np.exp(-2.j * self.m2) * f_rp2 - np.exp(+2.j * self.m2) * f_rm2)

        if r == 0:
            return df_k1.real, df_k2.real, df_m1.real, df_m2.real
        else:
            return df_k1, df_k2, df_m1, df_m2

    def _series_fun(self, r, get_gradients=False):
        """
        calculate moments and functions
        :param r: moment order
        :return:
        """
        # Make gvm parameters be column arrays and then copy them for terms columns
        k1 = self.k1.reshape(self.N, 1).repeat(self.terms, 1)
        k2 = self.k2.reshape(self.N, 1).repeat(self.terms, 1)
        dm = self.dm.reshape(self.N, 1).repeat(self.terms, 1)

        # Create a row vector with indexes from 1 to terms and then copy it for N rows
        idx = np.arange(self.terms).reshape(1, self.terms).repeat(self.N, 0) + 1

        # vectors (N x 1)
        iv_0_k2 = ive(0, self.k2.reshape(self.N, 1))
        iv_r_k1 = ive(r, self.k1.reshape(self.N, 1))

        # matrices that come up in the vectorized calculation (N x terms)
        exp_p2jd = np.exp(+2.j * idx * dm)
        exp_m2jd = np.exp(-2.j * idx * dm)

        iv_2jpr_k1 = ive(2 * idx + r, k1)
        iv_2jmr_k1 = ive(abs(2 * idx - r), k1)

        ivj_k2 = ive(idx, k2)

        # Moment-related series, np.sum along the dimension of the number of terms (np.reshape to avoid issues)
        g = iv_0_k2 * iv_r_k1 + \
            np.sum(ivj_k2 * (exp_p2jd * iv_2jpr_k1 + exp_m2jd * iv_2jmr_k1), axis=1).reshape(self.N, 1)
        g = g.reshape(self.R, self.C) * np.exp(1.j * r * self.m1)

        # Derivatives
        if get_gradients:
            d_iv0_k2 = ive(1, np.reshape(self.k2, (self.N, 1)))
            d_ivr_k1 = 0.5 * (
            ive(r - 1, np.reshape(self.k1, (self.N, 1))) + ive(r + 1, np.reshape(self.k1, (self.N, 1))))

            d_iv_2j_p_r_k1 = 0.5 * (ive(2 * idx + r - 1, k1) + ive(2 * idx + r + 1, k1))
            d_iv_2j_m_r_k1 = 0.5 * (ive(2 * idx - r - 1, k1) + ive(2 * idx - r + 1, k1))
            d_iv_j_k2 = 0.5 * (ive(idx - 1, k2) + ive(idx + 1, k2))

            dg_k1 = iv_0_k2 * d_ivr_k1 + \
                    np.sum(ivj_k2 * (exp_p2jd * d_iv_2j_p_r_k1 + exp_m2jd * d_iv_2j_m_r_k1), axis=1).reshape(self.N, 1)
            dg_k1 = dg_k1.reshape(self.R, self.C) * np.exp(1.j * r * self.m1)

            dg_k2 = d_iv0_k2 * iv_r_k1 + \
                    np.sum(d_iv_j_k2 * (exp_p2jd * iv_2jpr_k1 + exp_m2jd * iv_2jmr_k1), axis=1).reshape(self.N, 1)
            dg_k2 = dg_k2.reshape(self.R, self.C) * np.exp(1.j * r * self.m1)

            # Compose dg_m1 unp.sing the chain rule
            dg_d = 2.j * np.sum(ivj_k2 * idx * (exp_p2jd * iv_2jpr_k1 - exp_m2jd * iv_2jmr_k1), axis=1).reshape(self.N,
                                                                                                                1)
            dg_m1 = dg_d.reshape(self.R, self.C) * np.exp(1.j * r * self.m1) + 1.j * r * g

            dg_m2 = - dg_d.reshape(self.R, self.C) * np.exp(1.j * r * self.m1)

            return 2. * np.pi * g, [2. * np.pi * dg_k1, 2. * np.pi * dg_k2, 2. * np.pi * dg_m1, 2. * np.pi * dg_m2]
        else:
            return 2. * np.pi * g

    def moments(self, get_gradients):
        """
        calculates the moments of the distributions.
        :param get_gradients: obtain the gradients of the moments w.r.t. phi parameters
        :return:
        """

        if self.method is 'grid':
            self.T0 = self._grid_fun(0)
            self.T1 = self._grid_fun(1)
            self.T2 = self._grid_fun(2)
            if get_gradients:
                self.dT0 = self._grad_grid_fun(0)
                self.dT1 = self._grad_grid_fun(1)
                self.dT2 = self._grad_grid_fun(2)
        elif self.method is 'series':
            if get_gradients:
                self.T0, self.dT0 = self._series_fun(0, get_gradients=True)
                self.T1, self.dT1 = self._series_fun(1, get_gradients=True)
                self.T2, self.dT2 = self._series_fun(2, get_gradients=True)
            else:
                self.T0 = self._series_fun(0, get_gradients=False)
                self.T1 = self._series_fun(1, get_gradients=False)
                self.T2 = self._series_fun(2, get_gradients=False)
        else:
            raise NotImplementedError

        self.T0 = self.T0.real

        mc = self.T1.real / self.T0
        ms = self.T1.imag / self.T0
        m2c = self.T2.real / self.T0
        m2s = self.T2.imag / self.T0

        if get_gradients:
            self.dT0 = [self.dT0[0].real, self.dT0[1].real, self.dT0[2].real, self.dT0[3].real]

            dmc = [None, None, None, None]
            dms = [None, None, None, None]
            dm2c = [None, None, None, None]
            dm2s = [None, None, None, None]
            for ii in xrange(0, 4):
                d_mom1 = (self.T0 * self.dT1[ii] - self.dT0[ii] * self.T1) / (self.T0 ** 2)
                d_mom2 = (self.T0 * self.dT2[ii] - self.dT0[ii] * self.T2) / (self.T0 ** 2)
                dmc[ii] = d_mom1.real
                dms[ii] = d_mom1.imag
                dm2c[ii] = d_mom2.real
                dm2s[ii] = d_mom2.imag

            return mc, ms, m2c, m2s, dmc, dms, dm2c, dm2s
        else:
            return mc, ms, m2c, m2s

    def entropy(self, get_gradients):
        """
        gets the entropy for the GvM distribution
        :param get_gradients: obtain the gradients of the moments w.r.t. phi parameters
        :return:
        """
        # Get moments if they are not precomputed
        if get_gradients:
            mc, ms, m2c, m2s, dmc, dms, dm2c, dm2s = self.moments(get_gradients=True)
        else:
            mc, ms, m2c, m2s = self.moments(get_gradients=False)

        # Calculate the entropy
        k1_term = self.k1 * (np.cos(self.m1) * mc + np.sin(self.m1) * ms)
        k2_term = self.k2 * (np.cos(2. * self.m2) * m2c + np.sin(2. * self.m2) * m2s)
        h = np.log(self.T0) + self.k1 + self.k2 - (k1_term + k2_term)

        if get_gradients:
            dh = [None, None, None, None]
            if self.method is 'grid':
                # dh_dk1
                dh[0] = self.dT0[0] / self.T0 + 1. - (np.cos(self.m1) * mc + np.sin(self.m1) * ms +
                                                      self.k1 * (np.cos(self.m1) * dmc[0] + np.sin(self.m1) * dms[0]) +
                                                      self.k2 * (np.cos(2. * self.m2) * dm2c[0] +
                                                                 np.sin(2. * self.m2) * dm2s[0]))
                # dh_dk2
                dh[1] = self.dT0[1] / self.T0 + 1. - (self.k1 * (np.cos(self.m1) * dmc[1] + np.sin(self.m1) * dms[1]) +
                                                      np.cos(2. * self.m2) * m2c + np.sin(2. * self.m2) * m2s +
                                                      self.k2 * (np.cos(2. * self.m2) * dm2c[1] +
                                                                 np.sin(2. * self.m2) * dm2s[1]))
                # dh_dm1
                dh[2] = self.dT0[2] / self.T0 - (self.k1 * (np.cos(self.m1) * ms - np.sin(self.m1) * mc +
                                                            np.cos(self.m1) * dmc[2] + np.sin(self.m1) * dms[2]) +
                                                 self.k2 * (
                                                 np.cos(2. * self.m2) * dm2c[2] + np.sin(2. * self.m2) * dm2s[2]))
                # dh_dm2
                dh[3] = self.dT0[3] / self.T0 - (self.k1 * (np.cos(self.m1) * dmc[3] + np.sin(self.m1) * dms[3]) +
                                                 self.k2 * (2. * np.cos(2. * self.m2) * m2s -
                                                            2. * np.sin(2. * self.m2) * m2c +
                                                            np.cos(2. * self.m2) * dm2c[3] + np.sin(2. * self.m2) *
                                                            dm2s[3]))

            elif self.method is 'series':
                # dh_dk1
                dh[0] = self.dT0[0] / self.T0 - (np.cos(self.m1) * mc + np.sin(self.m1) * ms +
                                                 self.k1 * (np.cos(self.m1) * dmc[0] + np.sin(self.m1) * dms[0]) +
                                                 self.k2 * (
                                                 np.cos(2. * self.m2) * dm2c[0] + np.sin(2. * self.m2) * dm2s[0]))
                # dh_dk2
                dh[1] = self.dT0[1] / self.T0 - (self.k1 * (np.cos(self.m1) * dmc[1] + np.sin(self.m1) * dms[1]) +
                                                 np.cos(2. * self.m2) * m2c + np.sin(2. * self.m2) * m2s +
                                                 self.k2 * (
                                                 np.cos(2. * self.m2) * dm2c[1] + np.sin(2. * self.m2) * dm2s[1]))
                # dh_dm1
                dh[2] = self.dT0[2] / self.T0 - (self.k1 * (np.cos(self.m1) * ms - np.sin(self.m1) * mc +
                                                            np.cos(self.m1) * dmc[2] + np.sin(self.m1) * dms[2]) +
                                                 self.k2 * (
                                                 np.cos(2. * self.m2) * dm2c[2] + np.sin(2. * self.m2) * dm2s[2]))
                # dh_dm2
                dh[3] = self.dT0[3] / self.T0 - (self.k1 * (np.cos(self.m1) * dmc[3] + np.sin(self.m1) * dms[3]) +
                                                 self.k2 * (2. * np.cos(2. * self.m2) * m2s -
                                                            2. * np.sin(2. * self.m2) * m2c +
                                                            np.cos(2. * self.m2) * dm2c[3] + np.sin(2. * self.m2) *
                                                            dm2s[3]))
            else:
                raise NotImplementedError

            return h, dh
        else:
            return h
