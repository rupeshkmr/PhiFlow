from unittest import TestCase
from phi.math._extrapolation import *
from phi import math


class TestExtrapolation(TestCase):

    def test_pad(self):
        test_in_func_out = [
            (math.zeros(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(0).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             math.zeros(x=5, y=5, z=6, a=1)),
            (math.ones(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             math.ones(x=5, y=5, z=6, a=1)),
            (-math.ones(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(-1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             - math.ones(x=5, y=5, z=6, a=1)),
        ]
        for val_in, func, val_out in test_in_func_out:
            try:
                math.assert_close(val_out, func(val_in))
                # TypeError('__bool__ should return bool, returned NotImplementedType')
                # self.assertEqual(val_out, func(val_in))
            except Exception as e:
                raise BaseException(AssertionError(e, val_in, func, val_out))


class TestExtrapolationOperators(TestCase):
    """ensures that proper propagation of extrapolation occurs (for Field arithmetics)"""

    def test_constant(self):
        self.assertEqual(ConstantExtrapolation(2), ONE + ONE)
        self.assertEqual(ZERO, ONE - ONE)
        self.assertEqual(ONE, ONE * ONE)
        self.assertEqual(ONE, ONE / ONE)
        self.assertEqual(ZERO, ZERO / ONE)

    def test_constant_periodic_working(self):
        self.assertEqual(PERIODIC, PERIODIC + ZERO)
        self.assertEqual(PERIODIC, PERIODIC - ZERO)
        self.assertEqual(PERIODIC, ZERO + PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC / ONE)
        self.assertEqual(PERIODIC, PERIODIC * ONE)
        self.assertEqual(ZERO, PERIODIC * ZERO)

    def test_periodic_periodic(self):
        self.assertEqual(PERIODIC, PERIODIC + PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC - PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC * PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC / PERIODIC)

    def test_cross_errors(self):
        try:
            PERIODIC + BOUNDARY
            assert False
        except IncompatibleExtrapolations:
            pass

        try:
            PERIODIC + ONE
            assert False
        except IncompatibleExtrapolations:
            pass

    def test_pad_tensor(self):
        a = math.meshgrid([1, 2, 3, 4], [5, 6, 7])
        extrap = MixedExtrapolation({'x': PERIODIC, 'y': (ONE, REFLECT)})
        p = math.pad(a, {'x': (1, 2), 'y': (3, 4)}, extrap)
        self.assertEqual((7, 10, 2), p.shape.sizes)  # dimension check
        math.assert_close(p.x[1:-2].y[3:-4], a)  # contains original
        math.assert_close(p.y[:3], math.ones(x=7, y=3, vector=2))  # ONE Padding
        b = math.tensor(
            [[[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [4, 1, 2, 3, 4, 1, 2], [4, 1, 2, 3, 4, 1, 2], [4, 1, 2, 3, 4, 1, 2], [4, 1, 2, 3, 4, 1, 2], [4, 1, 2, 3, 4, 1, 2], [4, 1, 2, 3, 4, 1, 2], [4, 1, 2, 3, 4, 1, 2]],
             [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [5, 5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6, 6], [5, 5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7]]],
            batch_dims=0,
            names='vector, y, x'
        ).x.as_spatial().vector.as_channel()  # TODO: future this should be automatic
        math.assert_close(p, b)  # compare to reference

    def test_pad_collapsed(self):
        a = math.zeros(b=2, x=10, y=10, batch=10)
        p = math.pad(a, {'x': (1, 2)}, ZERO)
        self.assertIsInstance(p, CollapsedTensor)
        self.assertEqual((10, 2, 13, 10), p.shape.sizes)
        p = math.pad(a, {'x': (1, 2)}, PERIODIC)
        self.assertIsInstance(p, CollapsedTensor)
        self.assertEqual((10, 2, 13, 10), p.shape.sizes)
