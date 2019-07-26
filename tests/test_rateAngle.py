import math
from unittest import TestCase

from astropy import units
from astropy.units.quantity import Quantity

from mpsas.motion import RateAngle


class TestRateAngle(TestCase):

    def setUp(self):
        self.rateAngle = RateAngle(rate=13 * units.arcsec / units.hour, angle=45 * units.degree,
                                   ra0=180 * units.degree, dec0=0 * units.degree)

    def test_units(self):
        self.assertIsInstance(self.rateAngle.dec0, Quantity)

    def test_rate(self):
        self.rateAngle.predict('2000-01-01T00:00:00')
        self.assertAlmostEqual(self.rateAngle.coordinate.ra.value, self.rateAngle.ra0.value)
        self.assertTrue(self.rateAngle.coordinate.dec == self.rateAngle.dec0)
        self.rateAngle.predict('2000-01-01T01:00:00')
        self.assertAlmostEqual(self.rateAngle.coordinate.ra.value,
                               (self.rateAngle.ra0 - self.rateAngle.rate * math.cos(
                                   self.rateAngle.angle.to('radian').value) * units.hour).value)
        self.assertAlmostEqual(self.rateAngle.coordinate.dec.value,
                               (self.rateAngle.dec0 + self.rateAngle.rate * math.sin(
                                   self.rateAngle.angle.to('radian').value) * units.hour).value)
