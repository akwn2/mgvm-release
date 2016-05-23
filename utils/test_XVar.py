from unittest import TestCase

import numpy as np

from xvar import XVar


class TestXVar(TestCase):
    def test_pack(self):
        x = XVar(N=2, M=3, et=False, name='x')
        packed_x_xv = x.xv.flatten()
        np.testing.assert_array_almost_equal(packed_x_xv, x.pack('xv'))

        packed_x_dx = x.dx.flatten()
        np.testing.assert_array_almost_equal(packed_x_dx, x.pack('dx'))

        y = XVar(N=2, M=3, et=True, name='y')
        packed_y_xv = np.log(y.xv).flatten()
        np.testing.assert_array_almost_equal(packed_y_xv, y.pack('xv'))

        y = XVar(N=2, M=3, et=True, name='y')
        packed_y_dx = (y.dx / y.xv).flatten()
        np.testing.assert_array_almost_equal(packed_y_dx, y.pack('dx'))

    def test_unpack(self):
        x = XVar(N=2, M=3, et=False, name='x')
        x_xv_gt = x.xv
        x.unpack(x.pack('xv'), 'xv')
        np.testing.assert_array_almost_equal(x_xv_gt, x.xv)

        x = XVar(N=2, M=3, et=False, name='x')
        x_dx_gt = x.dx
        x.unpack(x.pack('dx'), 'dx')
        np.testing.assert_array_almost_equal(x_dx_gt, x.dx)

        y = XVar(N=2, M=3, et=True, name='x')
        y_xv_gt = y.xv
        y.unpack(y.pack('xv'), 'xv')
        np.testing.assert_array_almost_equal(y_xv_gt, y.xv)

        y = XVar(N=2, M=3, et=True, name='x')
        y_dx_gt = y.dx
        y.unpack(y.pack('dx'), 'dx')
        np.testing.assert_array_almost_equal(y_dx_gt, y.dx)
