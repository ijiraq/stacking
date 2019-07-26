import argparse
import os
from mpsas.image import import Image
from astropy import units
from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from mp_ephem import Observation, BKOrbit
from ossos import daophot
from pyds9 import DS9

parser = argparse.ArgumentParser()

parser.add_argument('image_filenames', nargs='+', help="image to measure the source location on")
parser.add_argument('name', help="name of the object being measured")

args = parser.parse_args()

imdisplay = DS9()
imdisplay.set('scale zscale')
imdisplay.set('cmap invert')
imdisplay.set("frame delete all")
ast_filename = "{}.ast".format(args.name)
if os.access(ast_filename, os.R_OK):
    orbit = BKOrbit(None, ast_filename="{}".format(args.name))
else:
    orbit = None

frames = []
images = []
# Load images into DS9
for filenames in args.image_filenames:
    Image()
    images.append(image)
    hdulist = fits.open(image)
    date = Time(hdulist[1].header['MJD_MID'], format='mjd')
    # w = wcs.WCS(hdulist[1].header)
    # print w.sky2xy(orbit.coordinate.ra, orbit.coordinate.dec)
    # pv_to_sip(hdulist[1].header)
    frames.append(hdulist)
    imdisplay.set('frame new')
    imdisplay.set('scale zscale')
    imdisplay.set_pyfits(hdulist)
    if orbit is not None:
        orbit.predict(date.iso, minimum_delta=1 * units.second)
        print(date.iso, orbit.coordinate.to_string('hmsdms', sep=":"))
        imdisplay.set("regions", "fk5;  circle({},{},{}) # colour red".format(orbit.coordinate.ra.degree,
                                                                              orbit.coordinate.dec.degree,
                                                                              0.0003))

# Measure RA/DEC of images.
imdisplay.set('frame 1')
cnt = 0
number = 0
while os.access("{}_{:03d}.ast".format(args.name, number), os.R_OK):
    number += 1
ast_filename = "{}_{:03d}.ast".format(args.nam, number)
print("Writing measurements to {}".ast_filename)
with open(ast_filename, 'w') as ast:
    for frame in frames:
        # ra, dec = imdisplay.get('iexam coordinate wcs fk5 degrees').split()
        c, x, y = imdisplay.get('iexam key coordinate image').split()
        if c != 'h':
            phot_result = daophot.phot(images[cnt], float(x), float(y), aperture=5, sky=20, swidth=10,
                                       apcor=0.3, zmag=28.32, extno=1)
            print(phot_result)
            mag = phot_result['MAG'][0]
            x = phot_result['XCENTER'][0]
            y = phot_result['YCENTER'][0]
        else:
            mag = None
        w = wcs.WCS(frame[1].header)
        ra, dec = w.all_pix2world(float(x), float(y), 1)
        print(x, y, ra, dec)
        ast.write("{}\n".format(Observation(provisional_name=orbit.observations[0].provisional_name,
                                            ra=float(ra),
                                            dec=float(dec),
                                            mag=mag,
                                            date=Time(frame[1].header['MJD_MID'], format='mjd', precision=5).mpc)))
        imdisplay.set('frame next')
        cnt += 1
