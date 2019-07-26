from unittest import TestCase
from mpsas.utils import get_sky_background, percentile_stack
import math
from astropy.io import fits
from numpy.testing import assert_allclose
import numpy as np
from mpsas.image import Image


class TestGetSkyBackground(TestCase):

    def setUp(self):

        self.file_with_cat = Image('data/image.fits')
        self.file_witout_cat = Image('data/image_nocat.fits')
        self.bounding_box = [300, 500, 1300, 1500]

    def test_get_sky_background(self):
        """
        Confirm that the two ways of computing sky give the same answer to the 1% level.
        """

        values_with_cat = get_sky_background(self.file_with_cat, bounding_box=self.bounding_box)
        values_without_cat = get_sky_background(self.file_witout_cat, bounding_box=self.bounding_box)
        relative_difference = math.fabs((values_without_cat[0]-values_with_cat[0])/values_without_cat[0])
        self.assertLess(relative_difference, 0.01)


class TestPercentileStack(TestCase):

    def setUp(self):
        self.hdulist = Image('data/image.fits').hdulist

    def test_percentile_stack(self):
        stack_list = [self.hdulist[0], self.hdulist[0], self.hdulist[0]]
        data = percentile_stack(stack_list).data
        assert_allclose(data, self.hdulist[0].data)

    def test_masking(self):

        hdulist = fits.HDUList()
        data = np.ma.ones((3, 3))
        hdulist.append(fits.PrimaryHDU(data=data))
        hdulist.append(fits.PrimaryHDU(data=data+1))
        hdulist.append(fits.PrimaryHDU(data=data+2))
        data = np.ma.ones((3, 3)) + 3
        data.mask = data > 3
        hdulist.append(fits.PrimaryHDU(data=data))
        median = percentile_stack(hdulist).data[1, 1]
        self.assertAlmostEqual(median, 2)


class TestImage(TestCase):

    def setUp(self):
        self.image = Image('data/image.fits')

    def test_obs_date(self):

        self.assertTrue(self.image.obs_date.isot[0:20] == "2012-07-18T07:00:52.489"[0:20])
