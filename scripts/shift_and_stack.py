from astropy.time import Time
from astropy.io import fits
from astropy import units
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
import numpy as np
import os
from scipy import ndimage, stats
from astropy.table import Table
from sip_tpv import pv_to_sip
import numpy
import logging
import warnings


def percentile_stack(stack_list, header, percentile=50):
    """
    Create a percentile stack of the images in the input HDUList, requires all HDUs to have images data sections of the
    same size.

    HDU data sections can be MaskedArrays and masked pixels will not be part of stack.

    :param stack_list: An HDUList to stack
    :param header: Master header to put into stacked image
    :param percentile: the percentile of the combined image data, 50 == median
    :return:
    """
    naxis1 = stack_list[0].header['NAXIS1']
    naxis2 = stack_list[0].header['NAXIS2']
    hdu_list = fits.HDUList()
    data_stack = []
    for image in stack_list:
        if image.data is None:
            # Not a data holding part of the HDUList sent for stacking.
            continue
        if naxis1 != image.header['NAXIS1'] or naxis2 != image.header['NAXIS2']:
            raise ValueError('Not all input HDUs have the same image size, no stacking possible.')
        data_stack.append(image.data)
    nimages = len(stack_list)
    vstack = np.ma.concatenate(data_stack)
    vstack.shape = [nimages, naxis2, naxis1]

    # Adding some arbitrary zeropoint to the image makes them display better in ds9.
    data = numpy.percentile(vstack.data, percentile, axis=0)
    del(header["NAXIS*"])
    del(header["DATA*"])

    hdu_list.append(fits.PrimaryHDU(data=data, header=header))
    # Append all the images that went into the stack onto the MEF.
    for image in stack_list:
        hdu_list.append(fits.ImageHDU(data=image.data, header=image.header))
    return hdu_list


class RateAngle(object):
    """
    Compute the RA/DEC of the centre of cutout based on having been provided a rate/angle an RA/DEC starting location.
    """
    def __init__(self, rate, angle, ra0=None, dec0=None, t0=None):
        self.rate = rate
        self.angle = angle
        self.t0 = t0
        self.ra0 = ra0
        self.dec0 = dec0
        self.coordinate = None
        r = (self.rate * 1 * units.hour).to('arcsec').value
        a = self.angle.to('degree')
        self.name = "R{:0.1f}A{:05.1f}".format(r, a)
        logging.debug("Computing for rate: {} and angle: {}".format(self.rate, self.angle))

    def predict(self, observation_time, **kwargs):
        if self.t0 is None:
            self.t0 = Time(observation_time)
        dr = self.rate*(Time(observation_time)-self.t0)
        logging.debug("Amount of motion: {}".format(dr.to('arcsec')))
        dra = dr*numpy.cos(self.angle.to('radian'))/numpy.cos(self.dec0)
        ddec = dr*numpy.sin(self.angle.to('radian'))
        logging.debug("dRA:{}, dDE:{}".format(dra.to('arcsec'), ddec.to('arcsec')))
        self.coordinate = SkyCoord(self.ra0+dra, self.dec0+ddec)


def load_orbit(target, location):
    """
    Build an mp_ephem BKOrbit object to use to compute x/y offsets.
    :param target: name of file with target ephemeris to fit.
    :return:
    """
    if not os.access(target, os.F_OK):
        # try and using 'mpc_filename' as target name for Horizons
        try:
            from mp_ephem import horizons
            return horizons.Body(target, center=location)
        except Exception as ex:
            logging.debug("Call to horizons returned: {}".format(str(ex)))
            return None
    from mp_ephem import BKOrbit
    return BKOrbit(None, target)


def get_dates(filenames, extno=0, dateobs="DATE-OBS", utcobs='UT', exptime="EXPTIME"):
    """
    Get the start and end dates for the exposure using the dateobs header keywords

    :param filenames: The FITS images to extract a dictionary of dates from.
    :param extno: Extension of FITS image that contains header with observation timing.
    :param dateobs: keyword with UT date of start of observation  (YYYY-MM-DD)
    :param utcobs: keyword with UT time of start of observation (HH:MM:SS.SS)
    :param exptime: Exposure Time keyword, expected to be in seconds
    :return: Dictionary of (start_date, end_date) tuples
    """
    _result = {}
    for filename in filenames:
        _header = fits.open(filename)[extno].header
        try:
           head_time_value = _header[dateobs]+"T"+_header[utcobs]
        except:
           head_time_value = _header[dateobs]
        start_date = Time(head_time_value, format='isot', scale='utc')
        end_date = start_date + _header[exptime] * units.second
        _result[filename] = (start_date, end_date)
    return _result


def get_sky_background(filename, bounding_box, extno=0):
    """
    Compute the mean sky background of the image using either a source_extractor catalog or mode of random pixel values.

    :param filename: name of image filename
    :param bounding_box: the x/y bounding box to determine the sky value inside of (x1,x2,y1,y2)
    :param extno: the extension number in the FITS image containing the data of interest.
    :return: sky value
    """
    sky = None
    catalog_filename = "{}.cat".format(filename.rstrip('.fits'))
    bounding_box = [ int(x) for x in bounding_box ]
    try:
        source_catalog = Table.read(catalog_filename, format='ascii')
        # determine the sky using sources that are in the cutout area.
        buffer = 0
        dim = bounding_box[1] - bounding_box[0]
        while sky is None and buffer < dim:
            try:
                cond = np.all((source_catalog['col1'] < bounding_box[1] + buffer,
                               source_catalog['col1'] > bounding_box[0] - buffer,
                               source_catalog['col2'] < bounding_box[3] + buffer,
                               source_catalog['col2'] > bounding_box[2] - buffer
                               ),
                              axis=0)
                sky = np.percentile(source_catalog['col5'][cond], 50)
            except Exception:
                # try expanding the buffer into the full image where we draw our
                # sky values from.
                buffer += dim / 4
        return sky, None
    except FileNotFoundError as err:
        logging.debug(str(err))
        logging.warning("Sky failed to open sextractor source catalog {}. "
                        "Using 1/10th random pixel sampling instead.".format(catalog_filename))
        sample = numpy.random.choice([True, True, True, False, False, False, False, False, False, False],
                                     (bounding_box[3] - bounding_box[2], bounding_box[1] - bounding_box[0]))
        with fits.open(filename) as hduList:
            try:
               image = hduList[extno]
               data = image.data[bounding_box[2]:bounding_box[3], bounding_box[0]:bounding_box[1]][sample]
               sky = float(numpy.percentile(data, 50))
               lclip = sky - 5 * numpy.sqrt(sky)
               uclip = sky + 5 * numpy.sqrt(sky)
               std = numpy.sqrt(numpy.mean(numpy.abs(data - data.mean()) ** 2))
               logging.info("Clipping out data values outside range ({:8.2f},{:8.2f})".format(uclip, lclip))
               cond = numpy.all((data < uclip, data > lclip), axis=0)
               sky = float(numpy.percentile(data[cond], 50))
               std = numpy.std(data[cond])
            except Exception as ex:
               logging.error(str(ex))
               logging.warning("Failed to get sky for {}, using 0 +/- 1E9".format(filename))
               sky = 0
               std = 1E9
        return sky, std


def get_wcs(filename, extno=0):
    """
    Construct a WCS object using content of filename.

    :param filename: name of the file that contains the wsc Header or whose extension changed to .wcs has the header.
    :param extno: which extension of a FITS file to get the HEADER from.
    :return: WCS
    """
    if 'wcs' == filename.split('.')[-1]:
        wcs_filename = filename
    else:
        wcs_filename = "{}.wcs".format(filename.split()[0])
    if os.access(wcs_filename, os.R_OK):
        wcs_header = fits.Header.fromfile(wcs_filename, padding=False, endcard=False, sep='\n')
    else:
        wcs_header = fits.open(filename)[extno].header

    pv_to_sip(wcs_header)
    # Determine the RA/DEC location of source in the current image and use as the reference point for cutouts.
    return wcs.WCS(wcs_header)


def get_xy_center(orbit, filename, observation_date, extno=0):
    """
    Using the target name or MPC file or list of x/y positions, determine the source location in this image.
    :param orbit: the orbit object used to compute the ra/dec location of source to centre on.
    :param filename: name of the file containing the image that will be used to convert RA/DEC to X/Y
    :param observation_date: Date position is needed for.
    :param extno:
    :return:
    """
    orbit.predict(observation_date.iso)
    ra = orbit.coordinate.ra.degree
    dec = orbit.coordinate.dec.degree
    xy = get_wcs(filename, extno=extno).all_world2pix([ra, ], [dec, ], 1)
    x = xy[0][0]
    y = xy[1][0]
    return x, y


def build_stack(filenames, orbit, stack_duration=60 * units.minute, dim=256, output_basename=None,
                header_extno=0, data_extno=0, clip=False, dateobs="DATE-OBS", timeobs="UT"):
    """
    Based on an input .mpc file (a file with MPC formated observations) compute an orbit and
    using that orbit stack fits images.  We also require a .cat file for each .fits image where
    the .cat file is a list of stellar sources, format:
    xcen ycen mag flux sky_flux.
    """

    write_cutout_separately = False

    if output_basename is None:
        output_basename = orbit.name

    observation_dates = get_dates(filenames, extno=header_extno, dateobs=dateobs, utcobs=timeobs)
    time_sorted_filenames = sorted(observation_dates.keys(), key=lambda xx: observation_dates[xx][0])

    # Initialize the stack list.
    stack_hdulist = []
    stack_number = 0
    stack_image_header = None
    stack_start_date = observation_dates[time_sorted_filenames[0]][0]
    stack_end_date = observation_dates[time_sorted_filenames[0]][1]

    for filename in time_sorted_filenames:
        logging.debug("Checking if {}[{}] should be in the stack.".format(filename, data_extno))
        basename = filename.split('.')[0]

        # Take the first extension of the FITS file as the image of interest.
        image = fits.open(filename)[data_extno]
        if stack_image_header is None:
            stack_image_header = image.header

        this_start_date, this_end_date = observation_dates[filename]
        (x, y) = get_xy_center(orbit, filename, this_start_date, extno=data_extno)
        x_limits = (5, image.header['NAXIS1'] - 5)
        y_limits = (5, image.header['NAXIS2'] - 5)
        if not (x_limits[0] < x < x_limits[1] and y_limits[0] < y < y_limits[1]):
            continue

        logging.debug('Cutout must be inside boundary of : ({},{})'.format(x_limits, y_limits))
        logging.info("filename:{:20} xcen:{:5.2f} ycen:{:5.2f}".format(filename, x, y))
        if (this_end_date - stack_start_date) > stack_duration:
            # we have accumulated images of length duration so we write the stack out to file.
            # and the start a new stack using the current image
            percentile_stack(stack_hdulist, stack_image_header).\
                writeto('{}_{:05d}.fits'.format(output_basename, stack_number), overwrite=True)
            # Increment the stack number, used to create a unique filename.
            stack_number += 1
            # Reset the contents of the stack to empty.
            stack_hdulist = []
            stack_start_date = this_start_date
            stack_end_date = this_start_date
            stack_image_header = image.header

        # determine the cutout boundary from the images, so that (x,y) is the centre of the cutout.
        dimx = image.header['NAXIS1'] if dim is None else int(dim)
        dimy = image.header['NAXIS2'] if dim is None else int(dim)

        x1 = max(x - dimx/2., x_limits[0])
        x2 = max(x1, min(x + dimx/2., x_limits[1]))
        y1 = max(y - dimy/2., y_limits[0])
        y2 = max(y1, min(y + dimy/2., y_limits[1]))
        logging.debug("Determined cutout boundaries to be : ({:8.1f},{:8.1f},{:8.1f},{:8.1f})".format(x1, x2, y1, y2))
        logging.debug("Data array has the shape: {}".format(image.data.shape))

        # make sure we have enough pixels in the stack such that adding this to the stack is useful signal
        # this boundary is set that at 1/8 the cutout size, rather arbitrary.
        if not (x1 + dimx/8 < x2 and y1 + dimy/8 < y2):
            logging.warning("SKIPPING: {} as too few pixels overlap".format(filename))
            continue

        # Add some information to the header so we can retrace our steps.
        stack_image_header.add_comment("{}[{},{}] - {}".format(filename, x, y, orbit.coordinate.to_string('hmsdms',
                                                                                                          sep=':')))
        # Cutout is done on integer pixel boundaries, the nd.shift task is used to move the real part over.
        # here 'cutout' provides the x/y location that the data cutout of image.data will go into in the stack.
        cutout_x1 = max(int(dimx/2) - int(x) + x_limits[0], 0)
        cutout_x2 = max(cutout_x1 + int(x2) - int(x1), 0)
        cutout_y1 = max(int(dimy/2) - int(y) + y_limits[0], 0)
        cutout_y2 = max(cutout_y1 + int(y2) - int(y1), 0)
        logging.debug("Bounds in cutout: [{}:{},{}:{}]".format(cutout_x1, cutout_x2, cutout_y1, cutout_y2))

        # shift the pixel data to remove the inter-pixel offsets.
        data = ndimage.shift(image.data, shift=(y1-int(y1), x1-int(x1)))
        logging.debug("After shifting data array has the shape: {}".format(data.shape))


        # Get the sky value to remove before combining.
        (sky, std) = get_sky_background(filename, (x1, x2, y1, y2), extno=0)

        # Use the Exposure Time value as the scale for combining. 
        scale = image.header['EXPTIME']

        # subtract the sky, scale the flux and extract from the input dataset.
        data = (data[int(y1):int(y2), int(x1):int(x2)] - sky) / scale
        logging.debug("After cutout extraction data has the shape:{}".format(data.shape))

        if std is not None and clip==True:
            # Mask if value is more then 5 std from sky
            mask = numpy.any((data < -5*std, data > 5*std), axis=0)
        else:
            mask = numpy.zeros(data.shape, dtype=bool)

        # Add the shifted data into a masked array where we mask out parts that don't have overlapping pixels.
        blank = np.ma.empty((dimy, dimx))
        blank.mask = True
        logging.debug("Bounds in cutout: [{}:{},{}:{}]".format(cutout_x1, cutout_x2, cutout_y1, cutout_y2))
        blank[cutout_y1:cutout_y2, cutout_x1:cutout_x2] = data
        blank[cutout_y1:cutout_y2, cutout_x1:cutout_x2].mask = mask[cutout_y1:cutout_y2, cutout_x1:cutout_x2]
        data = blank

        # set the pixels in the masked area to a value of '0'
        data[data.mask] = 0

        # Offset the WCS parameters to account for the cutout changing the DATASEC
        # when we write the stack we write out all the images and the stack, so good to keep these straight.
        image.header['DATASEC'] = "[{}:{},{}:{}]".format(cutout_x1+1, cutout_x2, cutout_y1+1, cutout_y2)
        image.header['CRPIX1'] -= x - dimx/2.0
        image.header['CRPIX2'] -= y - dimy/2.0
        image.header['MJD_MID'] = (stack_end_date.mjd + stack_start_date.mjd)/2.0
        image.header['XCEN'] = x
        image.header['YCEN'] = y
        image.header['TARGET'] = orbit.name
        image.header['T_RA'] = orbit.coordinate.ra.to('degree').value
        image.header['T_DE'] = orbit.coordinate.dec.to('degree').value
        image.data = data.data
        if write_cutout_separately:
            image.writeto('cutout/{}.fits'.format(basename.split('/')[-1]), overwrite=True)
        stack_hdulist.append(image)

    # we can exit the loop with a stack still to write.
    percentile_stack(stack_hdulist, stack_image_header).writeto('{}_{:05d}.fits'.format(output_basename, stack_number),
                                                                overwrite=True)


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
    group.add_argument('--rate-angle', nargs=2,
                       help="Rate (arcsec/hr) and angle (deg) of motion to shift and stack at")
    parser.add_argument('--observer-location', default='500', help="Observer location based on JPL notation")
    parser.add_argument('--duration', default=20,
                        help="Write a combined image after duration (in minutes) has passed.  If timespan covered by "
                             "input exceeds duration, multiple output images will be writen.")
    parser.add_argument('--output', default=None, help="Base name for output image stacks, defaults to target name.")
    parser.add_argument('--dim', help='size of cutout to extract for combining.', default=None)
    parser.add_argument('--header_extension_no', default=0, help="Which of the FITS extension has the Date keywords")
    parser.add_argument('--data_extension_no', default=0, help="Which of the FITS extension has the image DATA")
    parser.add_argument('--clip', action="store_true", help="Clip all values that are more than 5-sigma from the sky flux.")
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
                 "centred on location derived from {} and median combining".format(opts.dim, opts.dim,
                                                                                   len(opts.images), opts.target))
    if opts.target:
        orbit = load_orbit(opts.target, location=opts.observer_location)
    else:
        header = fits.open(opts.images[0])[opts.data_extension_no].header
        w = get_wcs(opts.images[0], extno=opts.data_extension_no)
        ra, dec = w.all_pix2world([header['NAXIS1'] / 2, ], [header['NAXIS2'] / 2, ], 0)
        logging.debug("Centre of input image: {} {}".format(ra,dec))
        rate, angle = opts.rate_angle
        rate *= units.arcsec/units.hour
        angle *= units.degree
        orbit = RateAngle(rate, angle, ra[0] * units.degree, dec[0] * units.degree)

    build_stack(opts.images, orbit, stack_duration=float(opts.duration) * units.minute,
                dim=opts.dim, output_basename=opts.output, clip=opts.clip, dateobs=opts.date_kw, timeobs=opts.time_kw)


if __name__ == '__main__':
    main()
