"""


size tupes are always (height, width) so that image.shape == (height, width, :)
coordinates are always (row, column) so that `image[row, column]` where `0 < row < height`
"""

import warnings
import time
from skimage import img_as_float, io, transform
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import logging


def center_roi_around(center_rc, size_hw):
    """
    Return a rectangular region of interest (ROI) of size `size_hw` around the
    given center coordinates. The returned ROI is defined as

        (start_row, start_column, end_row, end_column)

    where the start_row/column are ment to be *included* and the
    end_row/column are excluded.

    -   `center_rc`: Tuple of `(row, column)`. The numbers will be rounded to
        the closest integer. The `row` corresponds to the height and the
        `column` to the width. In mathematical notation you could think of the
        center to be given as the tuple `(y, x)`. Be aware of this; It does
        fit well to numpy's `shape` method.
    -   `size_hw`: A tuple of `(height, width)` of the resulting ROI. If this
        numbers are even, a UserWarning is issued.
    """
    height, width = size_hw

    if height % 2 == 0 or width % 2 == 0:
        warnings.warn(f"ROI with even height and width cannot be exactly "
                      f"centered. (height, width)=({height}, {width})")

    row, col = int(round(center_rc[0])), int(round(center_rc[1]))
    return (row - height//2,
            col - width//2,
            row + height//2 + 1,
            col + width//2 + 1)

def rotation_around(degrees, around_rc):
    """
    Returns a `degrees` counter clock wise rotation around the point `around_rc`.

    - `degrees`: Number in degrees for ccw rotation.
    - `around_rc`: The center of the rotation in (row, column) coordinates.

    Returns a `skimage.transform.AffineTransform` object.

    Note: You can apply the transfomation with
    `skimage.transform.warp(image, rotation)
    center_rc...coordinates (row,col) of rotation in image coordinates
    """

    # Calculate transformation matrices (skimage uses xy notation, which is [col, row])
    center_xy = np.asarray(around_rc[::-1])  # reverse
    tform1 = transform.AffineTransform(translation=-center_xy)
    tform2 = transform.AffineTransform(rotation=sp.deg2rad(degrees))
    tform3 = transform.AffineTransform(translation=center_xy)
    return tform1 + tform2 + tform3


def find_pattern_rotated(PF, pattern, image, rescale=1.0, rotations=(0,),
                         roi_center_rc=None, roi_size_hw=(41,41), plot=False, progress=None,
                         log_level=logging.DEBUG):
    """

    - `rotations`: Iterable over all rotations that should be tried. In degree.
    """
    if progress is None:
        def progress(x):
            return x

    logger = logging.getLogger('find_pattern_rotated')
    logger.setLevel(log_level)
    # Get current time to determine runtime of search
    start_time = time.time()

    # Initialize values needed later on
    result = []
    vmax = 0.0
    vmin = sp.Inf
    if len(image.shape) > 2:
        multichannel = True
    else:
        multichannel = False
    assert len(image.shape) == len(pattern.shape)

    # Set region of interest
    if roi_center_rc is None:
        roi_center_rc = sp.array(image.shape[:2])/2.0 - 0.5
    else:
        roi_center_rc = sp.asarray(roi_center_rc)
    roi = center_roi_around(roi_center_rc*rescale, roi_size_hw)

    # Give user some feedback on what is happening
    logger.info(f"Rescaling image and target by scale={rescale}.\n"
          f"    image (row, columns): {image.shape[0:2]} px --> {sp.asarray(image.shape[:2])*rescale} px.")
    logger.info(f"ROI center_rc={roi_center_rc}, in unscaled image.\n"
                f"     (height, width) = {roi_size_hw} in scaled image.")

    if len(rotations) > 1:
        logger.info(f"Trying rotations: {rotations}.")


    # Create rescaled copies of image and pattern, determine center coordinates
    pattern_scaled = transform.rescale(pattern, rescale, anti_aliasing=False, multichannel=multichannel, mode='constant')
    image_scaled = transform.rescale(image, rescale, anti_aliasing=False, multichannel=multichannel, mode='constant')
    PF.set_image(image_scaled)
    pattern_scaled_center = sp.array(pattern_scaled.shape[:2])/2. - 0.5
    pattern_center = sp.array(pattern.shape[:2])/2. - 0.5

    # Launch PatternFinder for all rotations defined in function input
    for r in progress(rotations):
        # Calculate transformation matrix for rotation around center of scaled pattern
        rotation_matrix = rotation_around(r, around_rc=pattern_scaled_center)
        # Launch Patternfinder
        pattern_scaled_rotated = transform.warp(pattern_scaled, rotation_matrix, mode='constant')
        # Make sure that the pixel at the image border are transparent, so that
        # pixel that are outside of the pattern are also transparent. This is because
        # we use the closest (border) pixel for getting the value of the pattern.
        pattern_scaled_rotated[0,:,3] = 0
        pattern_scaled_rotated[-1,:,3] = 0
        pattern_scaled_rotated[:,0,3] = 0
        pattern_scaled_rotated[:,-1,3] = 0

        out, min_coords, value = PF.find(pattern_scaled_rotated, roi=roi)
        opaque_pixel = pattern_scaled_rotated[...,-1].sum()  # the last number in RGBA
        out /= opaque_pixel
        value /= opaque_pixel
        # logger.info(f"r={r} opaque_pixel={opaque_pixel}")
        # min_ccords are (row, col)

        # Collect Min and Max values for plotting later on
        outmax = out.max()
        outmin = out.min()
        if outmax > vmax:
            vmax = outmax
        if outmin < vmin:
            vmin = outmin
        # undo the rescale for the coordinates
        min_coords = min_coords.astype(sp.float64) / rescale
        # create a list of results for all rotations
        result.append([r, min_coords, value, out])
    logger.info(f"took {time.time()-start_time} seconds.")

    # Select the best result from the result list and extract its parameters
    # The rotation angle is the 0-th element in result
    # The coordinates are in the 2-nd element
    # The actual value is the 3-rd element
    best_angle, best_coord, best_value, _ = result[sp.argmin([r[2] for r in result])]
    logger.info(f"best_angle: {best_angle} deg, best_coord (row,column): {best_coord} in input image")
    # Calculate transformation to transform image onto pattern
    # (note, PF.find did transform the pattern and NOT the image)
    translation = transform.AffineTransform(translation=(best_coord-pattern_center)[::-1])
    rotation = rotation_around(-best_angle, best_coord)
    T = translation + rotation
    #Create a plot showing error over angle
    if plot and len(rotations) > 1:
        fig, ax = plt.subplots(1)
        ax.plot([a[0] for a in result], [a[2] for a in result])
        ax.set_xlabel('Angle (rotation)')
        ax.set_ylabel('difference image-target')
        plt.show()
        plt.close()

    #Create heat plot of where target is in image
    if plot == 'all':
        fig, ax = plt.subplots()
        ax.imshow(image_scaled)
        ax.plot(sp.array([roi[1], roi[3], roi[3], roi[1], roi[1]]),
                sp.array([roi[2], roi[2], roi[0], roi[0], roi[2]]), "yellow")
        n_rows = int(sp.sqrt(len(result)))
        n_cols = int(sp.ceil(len(result)/n_rows))
        fig, ax = plt.subplots(n_rows, n_cols, squeeze=False, figsize = (2 * n_cols, 2 * n_rows))
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.suptitle("Correlation map of where target is in image\n", size=16)
        n = 0
        for i in range(n_rows):
            for j in range(n_cols):
                ax[i,j].axis("off")
                if n < len(result):
                    ax[i,j].imshow(result[n][3], interpolation="nearest", cmap='cubehelix', vmin=vmin, vmax=vmax)
                    ax[i,j].annotate('Angle:{0:.1f}\nValue:{1:.3f}'
                                     .format(result[n][0],result[n][2]),[0,0])
                    ax[i,j].plot(*(result[n][1]*rescale-sp.array(roi[:2]))[::-1], "rx")
                n += 1
        plt.show()
        plt.close()

    return T, best_value


