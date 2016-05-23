import numpy as np
from scipy.special import ive


class vM:
    def __init__(self, k1, m1):
        self.k1 = k1
        self.m1 = m1

    def moments(self, get_gradients):
        ive0 = ive(0, self.k1)
        ive1 = ive(1, self.k1)
        ive2 = ive(2, self.k1)

        T1 = ive1 / ive0 * np.exp(1.j * self.m1)
        T2 = ive2 / ive0 * np.exp(2.j * self.m1)

        mc = T1.real
        ms = T1.imag
        mc2 = 0.5 * (1 + T2.real)
        msc = 0.5 * T2.imag
        ms2 = 0.5 * (1 - T2.real)

        if get_gradients:
            ive3 = ive(3, self.k1)

            aux_1 = 0.50 * (ive0 + ive2) / ive0 - (ive1 / ive0) ** 2
            aux_2 = 0.25 * (ive1 + ive3) / ive0 - 0.50 * (ive1 / ive0) * (ive2 / ive0)
            aux_3 = ive1 / ive0
            aux_4 = ive2 / ive0

            c_m1 = np.cos(self.m1)
            s_m1 = np.sin(self.m1)
            s_2m1 = np.sin(2.0 * self.m1)
            c_2m1 = np.cos(2.0 * self.m1)

            ds_k1 = +s_m1 * aux_1
            dc_k1 = +c_m1 * aux_1
            ds2_k1 = -c_2m1 * aux_2
            dsc_k1 = +s_2m1 * aux_2
            dc2_k1 = +c_2m1 * aux_2

            ds_m1 = +c_m1 * aux_3
            dc_m1 = -s_m1 * aux_3
            ds2_m1 = +s_2m1 * aux_4
            dsc_m1 = +c_2m1 * aux_4
            dc2_m1 = -s_2m1 * aux_4

            dmc = [dc_k1, dc_m1]
            dms = [ds_k1, ds_m1]
            dmc2 = [dc2_k1, dc2_m1]
            dmsc = [dsc_k1, dsc_m1]
            dms2 = [ds2_k1, ds2_m1]

            return mc, ms, mc2, msc, ms2, dmc, dms, dmc2, dmsc, dms2
        else:
            return mc, ms, mc2, msc, ms2

    def entropy(self, get_gradients):
        """
        gets the entropy for the GvM distribution
        :param get_gradients: gets gradients with respect to the GvM parameters
        :return:
        """
        ive0 = ive(0, self.k1)
        ive1 = ive(1, self.k1)

        h = self.k1 + np.log(2 * np.pi * ive0) - self.k1 * ive1 / ive0

        if get_gradients:
            ive2 = ive(2, self.k1)
            dive1 = (ive2 + ive0) / 2.

            dh = [None, None]
            dh[0] = -self.k1 * (dive1 / ive0 - (ive1 / ive0) ** 2)
            dh[1] = np.zeros(self.m1.shape)

            return h, dh
        else:
            return h


class vM2:
    def __init__(self, k1cos, k1sin):
        self.k1cos = k1cos
        self.k1sin = k1sin

    def moments(self, get_gradients):
        k1 = np.abs(self.k1cos + 1.j * self.k1sin)
        m1 = np.angle(self.k1cos + 1.j * self.k1sin)

        ive0 = ive(0, k1)
        ive1 = ive(1, k1)
        ive2 = ive(2, k1)

        T1 = ive1 / ive0 * np.exp(1.j * m1)
        T2 = ive2 / ive0 * np.exp(2.j * m1)

        mc = T1.real
        ms = T1.imag
        mc2 = 0.5 * (1 + T2.real)
        msc = 0.5 * T2.imag
        ms2 = 0.5 * (1 - T2.real)

        if get_gradients:
            # Same as before
            ive3 = ive(3, k1)

            aux_1 = 0.50 * (ive0 + ive2) / ive0 - (ive1 / ive0) ** 2
            aux_2 = 0.25 * (ive1 + ive3) / ive0 - 0.50 * (ive1 / ive0) * (ive2 / ive0)
            aux_3 = ive1 / ive0
            aux_4 = ive2 / ive0

            c_m1 = np.cos(m1)
            s_m1 = np.sin(m1)
            s_2m1 = np.sin(2.0 * m1)
            c_2m1 = np.cos(2.0 * m1)

            ds_k1 = +s_m1 * aux_1
            dc_k1 = +c_m1 * aux_1
            ds2_k1 = -c_2m1 * aux_2
            dsc_k1 = +s_2m1 * aux_2
            dc2_k1 = +c_2m1 * aux_2

            ds_m1 = +c_m1 * aux_3
            dc_m1 = -s_m1 * aux_3
            ds2_m1 = +s_2m1 * aux_4
            dsc_m1 = +c_2m1 * aux_4
            dc2_m1 = -s_2m1 * aux_4

            # Now use the chain rule to get more stable numeric derivatives
            dk1_k1c = self.k1cos / k1
            dm1_k1c = - self.k1sin / ((1. + (self.k1sin / self.k1cos) ** 2) * self.k1cos ** 2)
            dk1_k1s = self.k1sin / k1
            dm1_k1s = 1 / ((1. + (self.k1sin / self.k1cos) ** 2) * self.k1cos)

            ds_k1c = ds_k1 * dk1_k1c + ds_m1 * dm1_k1c
            dc_k1c = dc_k1 * dk1_k1c + dc_m1 * dm1_k1c
            ds2_k1c = ds2_k1 * dk1_k1c + ds2_m1 * dm1_k1c
            dsc_k1c = dsc_k1 * dk1_k1c + dsc_m1 * dm1_k1c
            dc2_k1c = dc2_k1 * dk1_k1c + dc2_m1 * dm1_k1c

            ds_k1s = ds_k1 * dk1_k1s + ds_m1 * dm1_k1s
            dc_k1s = dc_k1 * dk1_k1s + dc_m1 * dm1_k1s
            ds2_k1s = ds2_k1 * dk1_k1s + ds2_m1 * dm1_k1s
            dsc_k1s = dsc_k1 * dk1_k1s + dsc_m1 * dm1_k1s
            dc2_k1s = dc2_k1 * dk1_k1s + dc2_m1 * dm1_k1s

            dmc = [dc_k1c, dc_k1s]
            dms = [ds_k1c, ds_k1s]
            dmc2 = [dc2_k1c, dc2_k1s]
            dmsc = [dsc_k1c, dsc_k1s]
            dms2 = [ds2_k1c, ds2_k1s]

            return mc, ms, mc2, msc, ms2, dmc, dms, dmc2, dmsc, dms2
        else:
            return mc, ms, mc2, msc, ms2

    def entropy(self, get_gradients):
        """
        gets the entropy for the GvM distribution
        :param get_gradients: gets gradients with respect to the GvM parameters
        :return:
        """
        k1 = np.abs(self.k1cos + 1.j * self.k1sin)
        ive0 = ive(0, k1)
        ive1 = ive(1, k1)

        h = k1 + np.log(2 * np.pi * ive0) - k1 * ive1 / ive0

        if get_gradients:
            ive2 = ive(2, k1)
            dive1 = (ive2 + ive0) / 2.

            dh = [None, None]
            dh_k1 = -k1 * (dive1 / ive0 - (ive1 / ive0) ** 2)

            # Now use the chain rule to get more stable numeric derivatives
            dk1_k1c = self.k1cos / k1
            dk1_k1s = self.k1sin / k1

            dh[0] = dh_k1 * dk1_k1c
            dh[1] = dh_k1 * dk1_k1s

            return h, dh
        else:
            return h
