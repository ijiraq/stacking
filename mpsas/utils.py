import logging
import os

import numpy as np
from astropy import units
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from scipy import ndimage


def get_filename(basename, extension='ast', exists=False):
    cnt = 0
    answer = None
    while True:
        cnt += 1
        filename = "{}_{:03d}.{}".format(basename, cnt, extension)
        if answer is None:
            answer = filename
        if os.access(filename, os.R_OK) and exists:
            answer = filename
            continue
        if not os.access(filename, os.R_OK):
            if not exists:
                answer = filename
            break
    return answer


def get_sky_background(image, bounding_box, sky_col=5, x_col=1, y_col=2):
    """
    Compute the mean sky background of the image using either a source_extractor catalog or mode of random pixel values.

    :param image: image to get sky level of.
    :param bounding_box: the x/y bounding box to determine the sky value inside of (x1,x2,y1,y2)
    :param sky_col: column in source extractor file that holds the sky value (one-based numbering)
    :param x_col: column in source extractor file that holds the x column (one-based, used for bounding limits)
    :param y_col: column in source extractor file that holds the y column (one-based, used for bounding limits)
    :return: sky value
    """

    xcol = "col{}".format(x_col)
    ycol = "col{}".format(y_col)
    scol = "col{}".format(sky_col)

    sky = None
    bounding_box = [int(x) for x in bounding_box]
    try:
        filename = os.path.splitext(image.filename)[0]+".cat"
        source_catalog = Table.read(filename, format='ascii')
        # determine the sky using sources that are in the cutout area.
        buffer = 0
        dim = bounding_box[1] - bounding_box[0]
        while sky is None and buffer < dim:
            try:
                cond = np.all((source_catalog[xcol] < bounding_box[1] + buffer,
                               source_catalog[xcol] > bounding_box[0] - buffer,
                               source_catalog[ycol] < bounding_box[3] + buffer,
                               source_catalog[ycol] > bounding_box[2] - buffer
                               ),
                              axis=0)
                sky = np.percentile(source_catalog[scol][cond], 50)
            except Exception:
                # try expanding the buffer into the full image where we draw our
                # sky values from.
                buffer += dim / 4
        return sky, None
    except FileNotFoundError as err:
        logging.debug(str(err))
        # try opening filename as image instead.
        # Setup a sample vector to select 1/10th of the pixels.
        sample = np.random.choice([True, False, False, False, False, False, False, False, False, False],
                                  (bounding_box[3] - bounding_box[2], bounding_box[1] - bounding_box[0]))
        try:
            data = image.data[bounding_box[2]:bounding_box[3], bounding_box[0]:bounding_box[1]][sample]
            sky = float(np.percentile(data, 50))
            cond = np.all((data < sky + 5 * np.sqrt(sky), data > sky - 5 * np.sqrt(sky)), axis=0)
            sky = float(np.percentile(data[cond], 50))
            std = np.std(data[cond])
        except Exception as ex:
            logging.error(str(bounding_box))
            logging.error(str(image.filename))
            logging.error(str(ex))
            logging.debug("Failed to get sky for {}, using 0 +/- 1E9".format(image.filename))
            sky = None
            std = None
        return sky, std


def percentile_stack(hdulist, percentile=50):
    """
    Create a percentile stack from an input HDUList, requires all HDUs to have images data sections of the
    same size and the size of the image correctly expressed in header['NAXIS1'] and header['NAXIS2'].

    HDU data sections can be MaskedArrays, masked pixels will not be part of stack.

    :param hdulist: An HDUList to stack
    :param percentile: the percentile of the combined image data, 50 == median
    :return:
    """
    # Confirm list of arrays have the same input shapes.
    shape = hdulist[0].data.shape
    for image in hdulist[1:]:
        if shape != image.data.shape:
            raise ValueError('Not all input HDUs have the same image size, no stacking possible.')

    datalist = []
    for image in hdulist:
        if image.data is None:
            # Not a data holding part of the HDUList sent for stacking.
            continue
        datalist.append(image.data)
    # Set the masked values to 'nan' and then take percentile.
    vstack = np.ma.filled(np.ma.concatenate(datalist), np.nan)
    vstack.shape = [len(hdulist), shape[0], shape[1]]
    data = nan_percentile(vstack, percentile)[0]
    return data.data
    # return fits.ImageHDU(data=data.data, header=hdulist[0].header)


def load_orbit(target, location):
    """
    Build an mp_ephem BKOrbit object to use to compute x/y offsets.
    :param target: name of file with target ephemeris to fit.
    :param location: the JPL location of observer, ony used if horizons.Body being created. (e.g. @nh)
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


def get_dates(filenames, extno=0, dateobs="DATE-OBS", utcobs='UTC-OBS', exptime="EXPTIME"):
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
    utcobs = None if utcobs.lower() == 'none' else utcobs
    for filename in filenames:
        head_time_value = None
        _header = fits.open(filename)[extno].header
        if utcobs is not None:
            try:
                head_time_value = _header[dateobs] + "T" + _header[utcobs]
            except KeyError:
                pass
        else:
            head_time_value = _header[dateobs]
        try:
            start_date = Time(head_time_value, format='isot', scale='utc')
            end_date = start_date + _header[exptime] * units.second
            _result[filename] = (start_date, end_date)
        except Exception as ex:
            logging.error("Failed to get time from header of {} using {}/{}".format(filename, dateobs, utcobs))
            raise ex
    return _result


def get_wcs(filename, extno=0):
    """
    Construct a WCS object using content of filename.

    :param filename: name of the file that contains the wcs Header or whose extension changed to .wcs has the header.
    :param extno: which extension of a FITS file to get the HEADER from.
    :return: WCS
    """


def get_xy_center(orbit, image):
    """
    Using the target name or MPC file or list of x/y positions, determine the source location in this image.
    :param orbit: the orbit object used to compute the ra/dec location of source to centre on.
    :param image: an image object that contains the meta-data about the image.
    :return:
    """
    orbit.predict(image.mid_exposure_time)
    ra = orbit.coordinate.ra.degree
    dec = orbit.coordinate.dec.degree
    return image.wcs.all_world2pix(ra, dec, 1)


def build_stack(time_sorted_images, orbit,
                stack_duration=60 * units.minute, dim=256, output_basename=None, clip=True):
    """
    Based on an input .mpc file (a file with MPC formated observations) compute an orbit and
    using that orbit stack fits images.  We also require a .cat file for each .fits image where
    the .cat file is a list of stellar sources, format:
    xcen ycen mag flux sky_flux.
    """

    # During debugging we can write out all the individual images, not useful for production.
    write_cutout_separately = False

    if output_basename is None:
        output_basename = orbit.name

    # Initialize the stack list.
    stack_hdulist = []
    stack_number = 0
    stack_image_header = None
    stack_start_date = None
    sky = 1000
    for image in time_sorted_images:
        logging.debug("Checking if {} should be in the stack.".format(image.filename))

        (x, y) = get_xy_center(orbit, image)

        # check that source is on the data region of the image.
        if not image.bounding_box[0] < x < image.bounding_box[1] and image.bounding_box[2] < y < image.bounding_box[3]:
            continue
        # determine the cutout boundary from the images, so that (x,y) is the centre of the cutout.
        dimx = image.header['NAXIS1'] if dim is None else int(dim)
        dimy = image.header['NAXIS2'] if dim is None else int(dim)

        x1 = max(x - dimx / 2., image.bounding_box[0])
        x2 = max(x1, min(x + dimx / 2., image.bounding_box[1]))
        y1 = max(y - dimy / 2., image.bounding_box[2])
        y2 = max(y1, min(y + dimy / 2., image.bounding_box[3]))
        logging.debug("Determined cutout boundaries to be : ({:8.1f},{:8.1f},{:8.1f},{:8.1f})".format(x1, x2, y1, y2))
        logging.debug("Data array has the shape: {}".format(image.data.shape))

        # make sure we have enough pixels in the stack such that adding this to the stack is useful signal
        # this boundary is set that at 1/8 the cutout size, rather arbitrary.
        if not (x1 + dimx / 8 < x2 and y1 + dimy / 8 < y2):
            logging.warning("SKIPPING: {} as too few pixels overlap".format(image.filename))
            continue

        # set this as the beginning of the stack, if we haven't already set a starting point
        if stack_image_header is None:
            stack_image_header = image.header
            stack_start_date = image.start_date

        logging.debug('Cutout must be inside boundary of : {}'.format(image.bounding_box))
        logging.info("filename:{:20} xcen:{:5.2f} ycen:{:5.2f}".format(image.filename, x, y))
        if (image.end_date - stack_start_date) > stack_duration:
            # we have accumulated images of length duration so we write the stack out to file.
            # and the start a new stack using the current image

            median_data = percentile_stack(stack_hdulist)
            tmphdu = fits.PrimaryHDU(header=stack_image_header, data=median_data)
            tmphdu.writeto('{}_{:05d}.fits'.format(output_basename, stack_number), overwrite=True)

            # Increment the stack number, used to create a unique filename.
            stack_number += 1
            # Reset the contents of the stack to empty.
            stack_hdulist = []
            stack_start_date = image.start_date
            stack_image_header = image.header

        # Add some information to the header so we can retrace our steps.
        stack_image_header.add_comment(
            "{}[{},{}] - {}".format(image.filename, x, y, orbit.coordinate.to_string('hmsdms',
                                                                                     sep=':')))

        # Cutout is done on integer pixel boundaries, the nd.shift task is used to move the real part over.
        # here 'cutout' provides the x/y location that the data cutout of image.data will go into in the stack.
        cutout_x1 = max(int(dimx / 2) - int(x) + image.bounding_box[0], 0)
        cutout_x2 = max(cutout_x1 + int(x2) - int(x1), 0)
        cutout_y1 = max(int(dimy / 2) - int(y) + image.bounding_box[2], 0)
        cutout_y2 = max(cutout_y1 + int(y2) - int(y1), 0)
        logging.debug("Bounds in cutout: [{}:{},{}:{}]".format(cutout_x1, cutout_x2, cutout_y1, cutout_y2))

        # shift the pixel data to remove the inter-pixel offsets.
        data = ndimage.shift(image.data, shift=(y1 - int(y1), x1 - int(x1)))
        logging.debug("After shifting data array has the shape: {}".format(data.shape))

        # Get the sky value to remove before combining.
        (sky, std) = get_sky_background(image, (x1, x2, y1, y2))

        # subtract the sky, scale the flux and extract from the input dataset.
        # Reset the PHOTZP and EXPTIME keywords to reflect new values.
        data = (data[int(y1):int(y2), int(x1):int(x2)] - sky) / image.exptime.value
        image.header['PHOTZP'] = image.header['PHOTZP'] - 2.5*np.log10(float(image.header['EXPTIME']))
        image.header['EXPTIME'] = 1.0
        image.header['SKY_MEAN'] = sky
        image.header['SKY_STD'] = std
        logging.debug("After cutout extraction data has the shape: {}".format(data.shape))

        if std is not None and clip is True:
            # Mask if value is more then 5 std from sky
            logging.info("Image: {} has sky STD of {}".format(image.filename, std))
            mask = np.any((data < -5 * std, data > 5 * std), axis=0)
        else:
            mask = np.zeros(data.shape, dtype=bool)

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
        image.header['DATASEC'] = "[{}:{},{}:{}]".format(cutout_x1 + 1, cutout_x2, cutout_y1 + 1, cutout_y2)
        image.header['CRPIX1'] -= x - dimx / 2.0
        image.header['CRPIX2'] -= y - dimy / 2.0
        image.header['MJD_MID'] = (image.start_date.mjd + (image.end_date.mjd - image.start_date.mjd)/2.0)
        image.header['XCEN'] = float(x)
        image.header['YCEN'] = float(y)
        image.header['TARGET'] = orbit.name
        image.header['T_RA'] = orbit.coordinate.ra.to('degree').value
        image.header['T_DE'] = orbit.coordinate.dec.to('degree').value
        stack_input = fits.ImageHDU(header=image.header, data=data.data)
        if write_cutout_separately:
            output_filename = "{}_{}.fits".format(output_basename, os.path.splitext(image.filename)[0])
            try:
                os.mkdir('cutout')
            except Exception as ex:
                logging.debug(str(ex))
                pass
            stack_input.writeto('cutout/{}.fits'.format(output_filename, overwrite=True))
        stack_hdulist.append(stack_input)

    # we can exit the loop with a stack still to write.
    fits.PrimaryHDU(header=stack_image_header, data=percentile_stack(stack_hdulist)). \
        writeto('{}_{:05d}.fits'.format(output_basename, stack_number), overwrite=True)


def _zvalue_from_index(arr, ind):
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    _, n_col, n_row = arr.shape

    # get linear indices and extract elements with np.take()
    idx = n_col * n_row * ind + np.arange(n_col * n_row).reshape((n_col, n_row))
    return np.take(arr, idx)


def nan_percentile(arr, q):
    """
    This nan_percentile function was taken from KRSTN's blog post:

    https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/

    :param arr:
    :param q:
    :return:
    """
    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    if type(q) is list:
        qs = []
        qs.extend(q)
    else:
        qs = [q]

    result = []
    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
        ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=arr, ind=k_arr.astype(np.int32))[
            fc_equal_k_mask]  # if floor == ceiling take floor value

        result.append(quant_arr)

    return result
