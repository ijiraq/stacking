from astropy import units
from astropy.utils.exceptions import AstropyWarning
import logging
import warnings
from mpsas import utils
from mpsas.image import Image
from mpsas.motion import RateAngle


def main():
    import argparse

    description = """
    A script to stack images based on the ephemeris of a TNO thought to be in the image.

    If target is an MPC ephemeris file then B&K fitting is used to determine an orbit and predict and ephemeris.
    If target is not a file, then assumed to name of an object and orbit is retrieved from JPL Horizons.

    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('images', nargs='*', help="list of images to stack.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--target', help="Traget name or MPC formatted ephemeris file.")
    group.add_argument('--ra-dec-rate-angle', nargs=4,
                       help="RA (deg) DEC (deg) Rate (arcsec/hr) and angle (deg) of motion to shift and stack at")
    group.add_argument('--x-y-rate-angle', nargs=4,
                       help="x (pix) y (pix), rate (pix/hr) and angle (deg) of motion for cutout and shift")
    parser.add_argument('--observer-location', default='500', help="Observer location based on JPL notation")
    parser.add_argument('--duration', default=20,
                        help="Write a combined image after duration (in minutes) has passed.  If timespan covered by "
                             "input exceeds duration, multiple output images will be writen.")
    parser.add_argument('--output', default=None, help="Base name for output image stacks, defaults to target name.")
    parser.add_argument('--dimension', help='size of cutout to extract for combining.', default=256)
    parser.add_argument('--header_extension_no', default=0, help="Which of the FITS extension has the Date keywords")
    parser.add_argument('--data_extension_no', default=0, help="Which of the FITS extension has the image DATA")
    parser.add_argument('--clip', action="store_true",
                        help="Clip all values that are more than 5-sigma from the sky flux.")
    parser.add_argument('--date-kw', help="Keyword with DATE keyword", default="DATE-OBS")
    parser.add_argument('--time-kw', help="Keyword with TIME keyword", default="UT")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--warnings', action="store_true")
    opts = parser.parse_args()

    if opts.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif opts.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    # Turn of astropy warnings
    if not opts.warnings:
        warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

    logging.info("Extracting {}x{} cutouts of {} images, "
                 "centred on location derived from {} and median combining".format(opts.dimension, opts.dimension,
                                                                                   len(opts.images), opts.target))

    images = []
    for image in opts.images:
        images.append(Image(image, opts.header_extension_no, opts.data_extension_no, opts.date_kw, opts.time_kw))

    time_sorted_images = sorted(images, key=lambda xx: xx.obs_date)

    if opts.target:
        import os
        ast_filename = utils.get_filename(opts.target, exists=True)
        print(ast_filename)
        if os.access(ast_filename, os.R_OK):
            from mp_ephem import BKOrbit
            orbit = BKOrbit(None, ast_filename)
            print(orbit.summarize())
        else:
            orbit = utils.load_orbit(opts.target, location=opts.observer_location)
    elif opts.x_y_rate_angle:
        x, y, rate, angle = [float(x) for x in opts.x_y_rate_angle]
        ra, dec = images[0].wcs.all_pix2world(x, y, 1)
        rate = rate*images[0].header['PIXSCAL1'] * units.arcsecond / units.hour
        angle *= units.degree
        ra *= units.degree
        dec *= units.degree
        orbit = RateAngle(rate=rate, angle=angle, ra0=ra, dec0=dec)
    else:
        # compute the ra/dec of the image centre.
        ra, dec, rate, angle = [float(x) for x in opts.ra_dec_rate_angle]
        logging.debug("Centre of input image: {} {}".format(ra, dec))
        ra *= units.degree
        dec *= units.degree
        rate *= rate*units.arcsec/units.hour
        angle *= units.degree
        orbit = RateAngle(rate=rate, angle=angle, ra0=ra, dec0=dec)

    utils.build_stack(time_sorted_images, orbit,
                      stack_duration=float(opts.duration)*units.minute,
                      output_basename=opts.output,
                      dim=opts.dimension)


if __name__ == '__main__':
    main()
