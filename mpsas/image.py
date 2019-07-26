import os
import re

from astropy import wcs, units
from astropy.io import fits
from astropy.time import Time
from sip_tpv import pv_to_sip


class Image(object):
    """
    Class to hold information about image in a uniform way and also holds the data section.
    """

    def __init__(self, filename, header_extno=0, date_extno=0, datekw="DATE-OBS", timekw="UT",
                 exptimekw="EXPTIME"):
        self.filename = filename
        self._header_extno = header_extno
        self._data_extno = date_extno
        self._obs_date = None
        self._datekw = datekw
        self._timekw = timekw
        self._exptimekw = exptimekw
        self._exptime = None
        self._hdulist = None
        self._wcs = None
        self._bounding_box = None

    @property
    def hdulist(self):
        if self._hdulist is None:
            self._hdulist = fits.open(self.filename)
        return self._hdulist

    @property
    def header_extno(self):
        return self._header_extno

    @header_extno.setter
    def header_extno(self, extno):
        self._header_extno = extno

    @property
    def data_extno(self):
        return self._data_extno

    @data_extno.setter
    def data_extno(self, extno):
        self._data_extno = extno

    @property
    def obs_date(self):
        if self._obs_date is None:
            _date = self.date_header.get(self.datekw, None)
            _time = self.date_header.get(self.timekw, None)
            self._obs_date = Time("{}T{}".format(_date, _time))
        return self._obs_date

    @obs_date.setter
    def obs_date(self, obs_date):
        self._obs_date = obs_date

    @property
    def data(self):
        return self.hdulist[self.data_extno].data

    @property
    def datekw(self):
        return self._datekw

    @property
    def timekw(self):
        return self._timekw

    @property
    def date_header(self):
        return self.hdulist[self.header_extno].header

    @property
    def header(self):
        return self.hdulist[self.data_extno].header

    @property
    def wcs_filename(self):
        return os.path.splitext(self.filename)[0]+".wcs"

    @property
    def wcs(self):
        if self._wcs is None:
            if os.access(self.wcs_filename, os.F_OK):
                _wcs_header = fits.Header.fromfile(self.wcs_filename, padding=False, endcard=False, sep='\n')
            else:
                _wcs_header = self.header
            pv_to_sip(_wcs_header)
            self._wcs = wcs.WCS(_wcs_header)
        return self._wcs

    @property
    def exptimekw(self):
        return self._exptimekw

    @property
    def exptime(self):
        if self._exptime is None:
            self._exptime = self.date_header[self.exptimekw] * units.second
        return self._exptime

    @property
    def start_date(self):
        return self.obs_date

    @property
    def end_date(self):
        return self.obs_date + self.exptime

    @property
    def mid_exposure_time(self):
        return self.obs_date + (self.end_date - self.obs_date)/2.0

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            datasec = self.header.get('DATASEC', None)
            if datasec is not None:
                match = re.findall(r'\D*(\d*)\D*', datasec)
                self._bounding_box = [int(x) for x in match[0:4]]
            else:
                self._bounding_box = [1, self.header['NAXIS1'], 1, self.header['NAXIS2']]
        return self._bounding_box
