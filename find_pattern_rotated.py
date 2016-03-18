"""

"""

import warnings
import time
from skimage import img_as_float, io, transform

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp


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
        warnings.warn("ROI with even height and width cannot be exactly "
                      "centered. The (height, width)=({height}, {width})"
                      .format(height=height, width=width))

    row, col = int(round(center_rc[0])), int(round(center_rc[1]))
    return (row - height//2,
            col - width//2,
            row + height//2 + 1,
            col + width//2 + 1)

def rotation_transform_center(image, angle, center_xy=None):
    """
    This function returns the transformation matrix for a rotation of a given image for a given angle 
    around a given center
    The operation is implemented by the following steps to avoid unwanted translational side effects:
    1.) Translate the image to the rotation center
    2.) Rotating the image for the given angle
    3.) Translate the image back to its original translatory position

    image...ndarray
    angle...angle in degree
    center_xy...coordinates of rotation in image coordinates
    """

    #If no rotation center is defined, set the center of the image as center
    if center_xy is None:
        cols, rows = image.shape[:2]
        center_xy = sp.array((rows, cols)) / 2. - 0.5
    #Calculate transformation matrices 
    tform1 = transform.AffineTransform(translation=-center_xy)
    tform2 = transform.AffineTransform(rotation=sp.deg2rad(angle))
    tform3 = transform.AffineTransform(translation=center_xy)
    #Return transformation matrix
    return tform1 + tform2 + tform3

def find_pattern_rotated(PF, pattern, image, rescale=0.2, rotate=(-60,61,120),
                         ellipsecorr=(1,1,1), ellipseres=1,
                         roi_center=None,
                         roi_size=(41,41),
                         plot=False):
    #Get current time to determine runtime of search
    start_time = time.time()

    #Initialize values needed later on
    result = []
    vmax = 0.0
    vmin = sp.Inf
    
    #Set region of interest
    if roi_center is None:
        roi_center = sp.array(im.shape[:2])/2.0 - 0.5
    roi = center_roi_around(roi_center*rescale, roi_size)
    
    #Give user some feedback on what is happening
    print("Rescaling image and target by scale={rescale}.\n"
          "   image {0}x{1} px to {2:.2f}x{3:.02f} px."
          .format(image.shape[0], image.shape[1],
                  image.shape[0]*rescale, image.shape[1]*rescale, rescale=rescale), flush=True)
    
    print("ROI: center={0}, {1}, in unscaled image.\n"
          "     height={2}, width={3} in scaled image"
          .format(roi_center[0], roi_center[1], roi_size[0], roi_size[1]))

    if rotate[2]>1:
        print("Now correlating rotations from {0}º to {1}º in {2} steps:"
              .format(*rotate))
    else:
        print("Rotation is kept constant at {0}°".format(rotate[0]))
    
    # Create rescaled copies of image and pattern, determine center coordinates of both
    pattern_scaled = transform.rescale(pattern, rescale)
    image_scaled = transform.rescale(image, rescale)
    PF.set_image(image_scaled)
    cols_scaled, rows_scaled = pattern_scaled.shape[:2]
    pattern_scaled_center = sp.array((rows_scaled, cols_scaled))/2. - 0.5
    cols, rows = pattern.shape[:2]
    pattern_center = sp.array((rows, cols))/2. - 0.5 
  
    # Launch PatternFinder for all rotations defined in function input
    rotations = sp.linspace(*rotate)
    for r in rotations:
        # Calculate transformation matrix for rotation around center of scaled pattern
        rotation_matrix = rotation_transform_center(pattern_scaled,r,center_xy=pattern_scaled_center)
        # Launch Patternfinder
        out, min_coords, value = PF.find(transform.warp(pattern_scaled,rotation_matrix), image=None, roi=roi)
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
        # Progress bar... kind of :)
        print(".",end="", flush=True)
    print("")
    print("took {0} seconds.".format(time.time()-start_time))
    
    #Select the best result from the result list and extract its parameters
    best_param_set = result[sp.argmin([r[2] for r in result])]
    best_angle = best_param_set[0]  # The rotation angle is the 0-th element in result
    best_coord = best_param_set[1]  # The coordinates are in the 2-nd element
    best_value = best_param_set[2]  # The actual value is the 3-rd element
    
    # Calculate transformation to transform image onto pattern
    move_to_center = transform.AffineTransform(translation=-(best_coord)[::-1])
    move_back = transform.AffineTransform(translation=(best_coord[::-1]))
    rotation = transform.AffineTransform(rotation=-sp.deg2rad(best_angle))
    translation = transform.AffineTransform(translation=sp.asmatrix((best_coord-pattern_center)[::-1]))
    T = translation + move_to_center + rotation + move_back
      
    #Create a plot showing error over angle
    if plot and rotate[2] > 1:
        fig, ax = plt.subplots(1)
        ax.plot([a[0] for a in result], [a[2] for a in result])
        ax.set_xlabel('Angle (rotation)')
        ax.set_ylabel('difference image-target')
        plt.show()
    
    #Create heat plot of where target is in image
    if plot == 'all':
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
                    ax[i,j].annotate('Angle:{0:.2f}; Value:{1:.2f}'
                                     .format(result[n][0],result[n][2]),[0,0])
                n += 1
        plt.show()
        
    return T, best_value
