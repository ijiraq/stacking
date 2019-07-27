import argparse
import os

from astropy import units
from astropy.time import Time
from mp_ephem import BKOrbit
from mp_ephem.ephem import Observation
from ossos import daophot
from pyds9 import DS9

from mpsas.image import Image
from mpsas.utils import get_filename


def date_key(obs):
    return obs.date.mpc[0:14]


ds9_init = {
    "bg": "black",
    "frame_lock": "wcs",
    "scale": "linear",
    "scale_mode": "zscale",
    "cmap_lock": "yes",
    "cmap_invert": "yes",
    "smooth": "no"
}


def app():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_filenames', nargs='+', help="image to measure the source location on")
    parser.add_argument('name', help="name of the object being measured")

    args = parser.parse_args()

    imdisplay = DS9()
    for key in ds9_init.keys():
        cmd = key.replace("_", " ")
        value = ds9_init[key]
        imdisplay.set('{} {}'.format(cmd, value))
    imdisplay.set('scale zscale')
    imdisplay.set('cmap invert')
    imdisplay.set("frame delete all")
    ast_filename = get_filename(args.name, exists=True)
    if os.access(ast_filename, os.R_OK):
        orbit = BKOrbit(None, ast_filename=ast_filename)
        print(orbit.summarize())
    else:
        orbit = None

    images = []
    # Load images into DS9
    for filename in args.image_filenames:
        images.append(Image(filename, timekw='UTC-OBS'))
        print(images[-1].filename)
        imdisplay.set('frame new')
        imdisplay.set('scale zscale')
        imdisplay.set_pyfits(images[-1].hdulist)
        if orbit is not None:
            date = images[-1].mid_exposure_time
            orbit.predict(date.iso, minimum_delta=1 * units.second)
            imdisplay.set("regions", "fk5;  circle({},{},{}) # colour red".format(orbit.coordinate.ra.degree,
                                                                                  orbit.coordinate.dec.degree,
                                                                                  0.0003))
    imdisplay.set('frame match wcs')
    # Measure RA/DEC of images.
    new_ast_filename = get_filename(args.name)
    print("Writing measurements to {}".format(new_ast_filename))
    observations = {}
    if orbit is not None:
        for obs in orbit.observations:
            observations[date_key(obs)] = obs
    for idx in range(len(images)):
        imdisplay.set("frame {}".format(idx+1))
        while True:
            result = imdisplay.get('iexam key coordinate wcs fk5 degrees')
            c, ra, dec = result.split()
            if c in ['h', 'x', 'r', 'q']:
                break
        if c == 'q':
            break
        if c == 'r':
            obs = Observation(provisional_name=args.name,
                              null_observation=True,
                              frame=os.path.splitext(images[idx].filename)[0],
                              ra=float(ra),
                              dec=float(dec),
                              mag=None,
                              date=Time(images[idx].mid_exposure_time, precision=5).mpc)
            observations[date_key(obs)] = obs
            continue
        x, y = images[idx].wcs.all_world2pix(float(ra), float(dec), 1)
        # c, x, y = imdisplay.get('iexam key coordinate image').split()
        if c != 'h':
            phot_result = daophot.phot(images[idx].filename,
                                       float(x), float(y), aperture=4, sky=20, swidth=4,
                                       exptime=images[idx].exptime.to('second').value,
                                       datamin=-5*images[idx].header['SKY_STD'],
                                       apcor=0.3, zmag=images[idx].header.get('PHOTZP', None),
                                       extno=images[idx].data_extno)
            mag = phot_result['MAG'][0]
            x = phot_result['XCENTER'][0]
            y = phot_result['YCENTER'][0]
        else:
            mag = None
        ra, dec = images[idx].wcs.all_pix2world(float(x), float(y), 1)
        obs = Observation(provisional_name=args.name,
                                            frame=os.path.splitext(images[idx].filename)[0],
                                            xpos=x+float(images[idx].header['XCEN']),
                                            ypos=y+float(images[idx].header['YCEN']),
                                            ra=float(ra),
                                            dec=float(dec),
                                            mag=mag,
                                            date=Time(images[idx].mid_exposure_time, precision=5))
        observations[date_key(obs)] = obs
    with open(new_ast_filename, 'w') as ast:
        keys = sorted(observations.keys())
        for key in keys:
            ast.write("{}\n".format(observations[key]))


if __name__ == '__main__':
    app()
