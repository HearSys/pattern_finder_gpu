
# Below is the Kernel which is uploaded and to does all the computations on
# the GPU. It is written in openCL language. Also see:


# https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf

import sys
import gc
import pyopencl as cl
import numpy as np
import warnings
import os

# Filename containing the src of the kernel
CONVOLVE_WITH_WEIGHTING_CL_KERNEL_FILENAME = os.path.dirname(__file__) + "/convolve_with_weighting.cl"


class PatternAtROIBorderWarning(UserWarning):

    """
    A warning that is issued when the PatternFinder retuns the best match
    coordinates at the border of the ROI. This indicates that the true global
    minimal value might be outside of the ROI.
    """


def idx_array_split(length, n_parts):
    parts = []
    Neach_section, extras = divmod(length, n_parts)
    section_sizes = ([0] +
                     extras * [Neach_section+1] +
                     (n_parts-extras) * [Neach_section])
    div_points = np.array(section_sizes).cumsum()
    for i in range(n_parts):
        st = div_points[i]
        end = div_points[i + 1]
        parts.append(end-st)
    return parts


class PatternFinder():

    """Find a given pattern in an image (OpenCL accelerated implementation)"""

    def __init__(self, opencl_queue=None, partitions=1):
        self.partitions = partitions
        assert partitions >= 1, "partitions must be >= 1"
        # Create an OpenCL context and queue, if None was given
        if opencl_queue is None:
            # In my observation the last device in the last platform (which is mostly just one)
            # is the powerful GPU
            self.ctx = cl.Context([cl.get_platforms()[-1].get_devices()[-1]])
            self.queue = cl.CommandQueue(self.ctx)#, properties=cl.command_queue_properties.PROFILING_ENABLE)
            print(self.ctx)
        else:
            self.ctx = opencl_queue.context
            self.queue = opencl_queue

        if not self.queue.device.get_info(cl.device_info.IMAGE_SUPPORT):
            raise Exception("OpenCL device {} does not support image".format(ctx.device))


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

    def set_image(self, image):
        """
        Assign and upload an image to the GPU for the find method.
        """
        self._image_gpu = self._upload_image(image)

    def find(self, pattern=None, image=None, roi=None):

        if image is not None:
            self.set_image(image)

        if pattern is not None:
            self.set_pattern(pattern)

        try:
            image_gpu = self._image_gpu
            target_gpu = self._target_gpu
        except AttributeError:
            raise Exception("No image or pattern available. Forgot to provide these or call set_image/set_pattern before?")

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

        image_start_row = 0
        for part, out_gpu in zip(parts, outputs_gpu):
            self._opencl_prg.convolve_image(self.queue,
                                            output_final.shape,  # --> height, width -> {get_global_id(0), get_global_id(1)} in kernel
                                            None,  # no local workgroups
                                            image_gpu, target_gpu, out_gpu,
                                            self.sampler_gpu,
                                            np.array([roi[0], roi[1]], dtype=np.int32),  # start_row, start_col in image
                                            np.int32(image_start_row),
                                            np.int32(image_start_row+part))
            # For the next round, we have to adapt the start column
            # by the height of the current part
            image_start_row += part
        assert image_start_row == image_gpu.shape[1], "${}".format(image_gpu.height, image_gpu.height, image_start_row)
        for i in range(partitions):
            cl.enqueue_copy(self.queue, outputs[i], outputs_gpu[i])
        self.queue.finish()
        for i in range(partitions):
            output_final += outputs[i]
        del outputs
        del outputs_gpu
        self.queue.flush()
        gc.collect()
        idx = np.array(np.unravel_index(output_final.argmin(), output_final.shape))
        if (idx[0] in (0, output_final.shape[0]-1) or
                idx[1] in (0, output_final.shape[1]-1)):
            warnings.warn("PatternFinder: Minimal value at border of ROI! "
                          "This hints at a too small ROI. Actual minimum "
                          "might be outside of the ROI.",
                          PatternAtROIBorderWarning)
        return output_final, idx+roi[0:2], output_final[idx[0], idx[1]]
