import numpy as np
from astropy import units
from astropy.units.quantity import Quantity
from astropy.coordinates import SkyCoord
from astropy.time import Time
import logging


class RateAngle(object):
    """
    Compute the RA/DEC of the centre of cutout based on having been provided a rate/angle an RA/DEC starting location.
    """
    def __init__(self, rate, angle, ra0=None, dec0=None, t0=None):
        self.rate = rate
        self.angle = angle
        self.t0 = t0
        self.ra0 = Quantity(ra0)
        self.dec0 = Quantity(dec0)
        self.coordinate = None
        r = (self.rate * 1 * units.hour).to('arcsec').value
        a = self.angle.to('degree')
        self.name = "R{:0.1f}A{:05.1f}".format(r, a)
        logging.debug("Computing for rate: {} and angle: {} centred at {} {}".format(self.rate, self.angle,
                                                                                     self.ra0, self.dec0))

    def predict(self, observation_time, **kwargs):
        """

        :param observation_time: Date to predict location of object, as Time or str
        :param kwargs: These are extra arguments that mp_ephem.BKOrbit.predict might have, that this doesn't
        :return: None
        """
        if len(kwargs) > 0:
            logging.debug("RateAngle predict ignores: {}".format(kwargs))
        if self.t0 is None:
            self.t0 = Time(observation_time)
        dr = self.rate*(Time(observation_time)-self.t0)
        logging.debug("Amount of motion: {}".format(dr.to('arcsec')))
        dra = dr*np.cos(self.angle.to('radian'))/np.cos(self.dec0.to('radian'))
        ddec = dr*np.sin(self.angle.to('radian'))
        logging.debug("dRA:{}, dDE:{}".format(dra, ddec))
        self.coordinate = SkyCoord(self.ra0 - dra.to('degree'), self.dec0+ddec.to('degree'))

