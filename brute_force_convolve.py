
import sys
import gc
import pyopencl as cl
import numpy as np
import warnings
import os
import logging
import time


# Filename containing the src of the kernel
CONVOLVE_WITH_WEIGHTING_CL_KERNEL_FILENAME = os.path.dirname(__file__) + "/convolve_with_weighting.cl"


class PatternAtROIBorderWarning(UserWarning):

    """
    A warning that is issued when the PatternFinder retuns the best match
    coordinates at the border of the ROI. This indicates that the true global
    minimal value might be outside of the ROI.
    """


def idx_array_split(length, n_parts):
    """
    Helper that splits the number `length` into `n_parts` and returns a list
    of indices that can be used to index an array dimension of `length`.

    Similar (adapted from) numpy.split_array.
    """
    parts = []
    Neach_section, extras = divmod(length, n_parts)
    section_sizes = ([0] +
                     extras * [Neach_section + 1] +
                     (n_parts - extras) * [Neach_section])
    div_points = np.array(section_sizes).cumsum()
    for i in range(n_parts):
        st = div_points[i]
        end = div_points[i + 1]
        parts.append(end - st)
    return parts


class PatternFinder():

    """Find a given pattern in an image (OpenCL accelerated implementation)"""

    def __init__(self, opencl_queue=None, partitions=1):
        """
        Create a new PatternFinder object.

        Optionally providing an OpenCL command queue and partitions to limit
        the size one signle kernel has to compute.

        -   `opencl_queue`: The command queue will default to the last platform
             and the last device. Usually that is the GPU.
             You can assign this your self

                 import pyopencl as cl
                 platform = cl.get_platforms()[i]
                 device = platform.get_devices[j]
                 queue = cl.CommandQueue(cl.Context([device]))  # , properties=cl.command_queue_properties.PROFILING_ENABLE)

        -   `partitions`: Split the image in this many parts (default=1) in
            order to keep the kernel runtime shorter for buggy nVidia-Windows
            driver. Usually, there is no speed-up when using more partitions
            for large ROIs as the GPU is alreay fully busy because each pixel
            in the ROI is computed in parallel. However, if you have fewer
            pixels in your ROI than the GPU can handle parallel executions,
            theoretically you might gain a small speed-up with higher
            `partitions` number
            If you experience GPU driver crashes, try to set this number
            higher, e.g. 10 is a goog guess.
        """
        self.logger = logging.getLogger("PatternFinder")
        self.logger.level = logging.DEBUG

        self.partitions = partitions
        assert partitions >= 1, "partitions must be >= 1"

        # Create an OpenCL context and queue, if None was given
        if opencl_queue is None:
            # In my observation the last device in the last platform
            # (which is mostly just one) is the powerful GPU
            self.ctx = cl.Context([cl.get_platforms()[-1].get_devices()[-1]])  # , properties=cl.command_queue_properties.PROFILING_ENABLE)
            self.queue = cl.CommandQueue(self.ctx)
            self.logger.info(self.ctx)
        else:
            self.ctx = opencl_queue.context
            self.queue = opencl_queue

        if not self.queue.device.get_info(cl.device_info.IMAGE_SUPPORT):
            raise Exception("OpenCL device {} does not support image".format(self.ctx.device))

        # The Sampler for how OpenCL accesses the images (pixel coordinates, no interpolation)
        self.sampler_gpu = cl.Sampler(self.ctx,
                                      False,
                                      cl.addressing_mode.CLAMP_TO_EDGE,
                                      cl.filter_mode.NEAREST)

        # Read-in and compile the OpenCL Kernel-Program
        with open(CONVOLVE_WITH_WEIGHTING_CL_KERNEL_FILENAME, 'r') as f:
            self._opencl_prg = cl.Program(self.ctx, f.read()).build()

    def transfrom_target(self, transform_matrix):
        raise NotImplemented

    def transform_image(self, transform_matrix):
        raise NotImplemented

    def _upload_image(self, image):
        assert image.max() <= 1.0

        # Check the number of channels in the image
        if image.ndim == 2:
            num_channels = 1
        else:
            if sys.platform.startswith('win') and 'geforce' in self.ctx.devices[0].name.lower() and image.shape[2] == 3:
                # This is a hack for Windows/nVidia, as we believe and found so
                # far for various GeFoce cards that the nvidia OpenCL
                # implementation sucks. Reporting an out-of-resources error when
                # trying to upload an RGB three channel image to the GPU
                # Quite counterintuitively adding an unneeded fourth channel
                # makes the out-of-resources error go away. FIXME if you can.
                tmp = image
                image = np.ones((tmp.shape[0], tmp.shape[1], 4))
                num_channels = 4
                image[:, :, :3] = tmp[:]
            else:
                num_channels = image.shape[2]

        # Tell OpenCL to copy the image into device memory
        image_gpu = cl.image_from_array(self.ctx, image.astype(np.float32),
                                        num_channels=num_channels, mode="r")
        return image_gpu

    def set_pattern(self, pattern):
        """
        Assign and upload a pattern image to the GPU for the find method.
        """
        # Check that pattern is an RGBA image
        assert pattern.ndim == 3 and pattern.shape[2] == 4, "pattern has to be a 4-channel RGBA image."

        if pattern.shape[0] % 2 == 0 or pattern.shape[1] % 2 == 0:
            warnings.warn("For best results, pattern should have an odd "
                          "number of columns/rows but its shape is: {}".format(pattern.shape),
                          stacklevel=2)

        self._target_gpu = self._upload_image(pattern)

    property(fset=set_pattern, doc=set_pattern.__doc__)

    def set_image(self, image):
        """
        Assign and upload an image to the GPU for the find method.
        """
        self._image_gpu = self._upload_image(image)

    property(fset=set_image, doc=set_image.__doc__)

    def find(self, pattern=None, image=None, roi=None):
        """
        Find the position where `pattern` is most likely in `image` in the ROI.

        This is a GPU implemented (OpenCL) brute force (global) search.

        You can vastly improve the runtime by providing a region-of-interest
        (ROI) in image coordinates where to look for the pattern.

        The position of the *center* of the pattern is returned and *NOT* the
        top-left corner. This is helpful is your pattern is centered.

        If you need to call this method multiple times, it might be advied to
        set the `pattern` and/or the `image` before

        -   `pattern`: A numpy 2d image with a shape (columns, rows, 4) where
            the last color band is the alpha channel (0.0 is transparent).
            Optional, if the property `.pattern` has been set.
        -   `image`: A numpy 2d image with a shape (columns, rows, 3).
            Optional, if the property `.image` has been set.
        -   `roi`: Optional region-of-interest in zero-based image coordinates.
             This is a fourh-tuple `(start_row, start_col, row_end, col_end)`
             where the start row/cols are included but row/col_end are
             excluded.

        Warns, if the minimum is at the border of the ROI as that hints to the
        true minimum being outside of the ROI.

        Returns a three-tuple of the convoluted ROI, the coordinates in the
        image coordinate system and the minimal value at the minimum.

        """
        time_start = time.time()

        if image is not None:
            self.set_image(image)

        if pattern is not None:
            self.set_pattern(pattern)

        try:
            image_gpu = self._image_gpu
            target_gpu = self._target_gpu
        except AttributeError:
            raise Exception("No image or pattern available. "
                            "Forgot to provide these or call set_image/set_pattern before?")

        # Compute the Region Of Interest if not given
        # (row_start, col_start, row_end, col_end)
        # Including start row/cols but without row/col_end
        if roi is None:
            roi = (0, 0, image_gpu.shape[1], image_gpu.shape[0])

        partitions = min(image_gpu.shape[1], self.partitions)
        # split the image into self.partitions parts but not more than rows in the image
        parts = idx_array_split(length=image_gpu.shape[1], n_parts=partitions)

        output_final = np.zeros((roi[2]-roi[0], roi[3]-roi[1]), dtype=np.float32)

        # For each partition create one output array (and one that is copied to the GPU)
        outputs = [np.zeros_like(output_final) for i in range(partitions)]
        outputs_gpu = [cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)
                       for out in outputs]

        # partitions split the image in parts. We compute init the start row:
        image_start_row = 0
        for part, out_gpu, out in zip(parts, outputs_gpu, outputs):
            cl_op = self._opencl_prg.convolve_image(self.queue,
                                                    output_final.shape,  # --> height, width -> {get_global_id(0), get_global_id(1)} in kernel
                                                    None,  # no local workgroups
                                                    image_gpu, target_gpu, out_gpu,
                                                    self.sampler_gpu,
                                                    np.array([roi[0], roi[1]], dtype=np.int32),  # start_row, start_col in image
                                                    np.int32(image_start_row),
                                                    np.int32(image_start_row + part))
            # For the next round, we have to adapt the start column
            # by the height of the current part
            image_start_row += part
            # Start to copy back already now (non-blocking)
            cl.enqueue_copy(self.queue, out, out_gpu,
                            wait_for=[cl_op], is_blocking=False)
        assert image_start_row == image_gpu.shape[1], ("${}".format(image_gpu.height, image_gpu.height, image_start_row))
        self.queue.finish()
        for i in range(partitions):
            output_final += outputs[i]
        del outputs
        del outputs_gpu
        self.queue.flush()
        gc.collect()
        idx = np.array(np.unravel_index(output_final.argmin(), output_final.shape))
        if (idx[0] in (0, output_final.shape[0] - 1) or idx[1] in (0, output_final.shape[1] - 1)):
            warnings.warn("PatternFinder: Minimal value at border of ROI! "
                          "This hints at a too small ROI. Actual minimum "
                          "might be outside of the ROI.",
                          PatternAtROIBorderWarning)

        self.logger.debug("Execution PatternFinder.find took %f ms",
                          (time.time()-time_start) / 1000)
        return output_final, idx + roi[0:2], output_final[idx[0], idx[1]]
