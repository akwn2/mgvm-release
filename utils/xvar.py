import numpy as np
from numpy.random import rand


class XVar(object):
    """
    Special variable class that aggregates quantities important for numerical evaluation of a variable into a single
    object.
    """

    def __init__(self, N=1, M=1, et=False, name=''):

        self.name = name
        self.et = et

        self.xv = rand(N, M).astype(dtype=np.float64)
        if self.et:
            self.xv = np.abs(self.xv)

        self.dx = rand(N, M).astype(dtype=np.float64)
        self.lb = np.repeat(None, N * M)
        self.ub = np.repeat(None, N * M)

    def pack(self, pack_type):
        """
        pack values to a one-dimensional array
        :param pack_type:
        :return:
        """
        if pack_type == 'xv':
            if self.et:
                return np.log(self.xv.flatten())
            else:
                return self.xv.flatten()
        elif pack_type == 'dx':
            if self.et:
                return (self.dx * self.xv).flatten()
            else:
                return self.dx.flatten()
        else:
            print('!!! Error: Unknown pack_type supplied!')
            raise NotImplementedError

    def unpack(self, var_array, pack_type):
        """
        unpack values from a one-dimensional array
        :param var_array:
        :param pack_type:
        :return:
        """
        if pack_type == 'xv':
            if self.et:
                # This is to assure that the values do not overflow
                if np.any(var_array > 700.):
                    var_array[var_array > 700.] = 700.

                self.xv = np.reshape(np.exp(var_array), self.xv.shape)
            else:
                self.xv = np.reshape(var_array, self.xv.shape)
        elif pack_type == 'dx':
            if self.et:
                self.dx = np.reshape(var_array, self.dx.shape) / self.xv
            else:
                self.dx = np.reshape(var_array, self.dx.shape)
        else:
            print('!!! Error: Unknown pack_type supplied!')
            raise NotImplementedError
