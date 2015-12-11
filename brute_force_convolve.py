
# Below is the Kernel which is uploaded and to does all the computations on
# the GPU. It is written in openCL language. Also see:


# https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf

import sys
import gc
import pyopencl as cl
import numpy as np


class PatternAtROIBorderWarning(UserWarning):
    ...


class PatternFinder():

    """Find a given pattern in an image (OpenCL accelerated implementation)"""

    opencl_kernel = """
    kernel void convolve_image(__read_only image2d_t image,
                               __read_only image2d_t target,
                               __global __write_only float* output,
                               sampler_t sampler,
                               const int2 top_left_rc,      // in image coords (row, column)
                               const int image_start_r,  // where in the image row to start writing
                               const int image_end_r)    // end row writing (for partitions)
    {
        // position (row, column) of the pixel in the output image
        // note, that opencl has `.x` as the first member in the int2 struct!
        const int2 pos = {get_global_id(0), get_global_id(1)};

        // The width and height of the target image
        const int image_height = get_image_height(image);

        // target should be at least 3x3 and have an odd width and height
        const int target_half_w = (get_image_width(target)-1)/2;
        const int target_half_h = (get_image_height(target)-1)/2;


        // 4-tuple of the RGBA values from the target image
        float4 t_value = (float4)(0.0, 0.0, 0.0, 0.0);

        int2 image_pos = (int2)(0,0);
        image_pos.s1 = top_left_rc.s0 + pos.s0 - target_half_h; // row
        image_pos.s0 = top_left_rc.s1 + pos.s1 - target_half_w; // column

        // value for the output pixel we are going to compute in this kernel
        float value = 0.0f;

        // Go though all pixel of the target image
        for (int c = image_start_r; c < image_end_r; c++) {

            for (int r = 0; r < image_height; r++) {

                t_value = read_imagef(target, sampler, (int2)(c, r));

                if(t_value.w > 0.0) {
                    // Use the alpha value w from an RGBA image (target) as weight
                    // and compute the `distance` which is the absolute difference
                    value += (1-t_value.w) * 3 + t_value.w * distance(t_value.xyz,
                                                                      read_imagef(image,
                                                                                  sampler,
                                                                                  image_pos + (int2)(c, r)).xyz);
                } else {
                    value += 3;
                }
            }
        }

        output[pos.s0 * get_global_size(1) + pos.s1] = value;
    }
    """

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
        # compile the OpenCL Program
        self.opencl_prg = cl.Program(self.ctx, PatternFinder.opencl_kernel).build()

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
        # Check that pattern is an RGBA image
        assert pattern.ndim == 3 and pattern.shape[2] == 4, "pattern has to be a 4-channel RGBA image."
        if pattern.shape[0] % 2 == 0:
            raise UserWarning("pattern should have an odd number of rows")
        if pattern.shape[1] % 2 == 0:
            raise UserWarning("pattern has to have an odd number of columns")

        self._target_gpu = self._upload_image(pattern)

    def set_image(self, image):
        self._image_gpu = self._upload_image(image)

    def find(self, pattern=None, image=None, roi=None):

        # Compute the Region Of Interest if not given (row_start, col_start, row_end, col_end)
        # Including start row/cols but without row/col_end
        if roi is None:
            roi = (0, 0, image.shape[0], image.shape[1])

        if image is not None:
            self.set_image(image)

        if pattern is not None:
            self.set_pattern(pattern)

        partitions = min(image.shape[0], self.partitions)

        # split the image into self.partitions parts but not more than rows in the image
        splitted_image = np.array_split(image, partitions, axis=0)

        image_gpu = self._image_gpu
        target_gpu = self._target_gpu

        output_final = np.zeros((roi[2]-roi[0], roi[3]-roi[1]), dtype=np.float32)

        # For each partition create one output array (and one that is copied to the GPU)
        outputs = [np.zeros_like(output_final) for i in range(partitions)]
        outputs_gpu = [cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)
                       for out in outputs]

        image_start_row = 0
        for part, out_gpu in zip(splitted_image, outputs_gpu):
            self.opencl_prg.convolve_image(self.queue,
                                           output_final.shape,  # --> height, width -> {get_global_id(0), get_global_id(1)} in kernel
                                           None,  # no local workgroups
                                           image_gpu, target_gpu, out_gpu,
                                           self.sampler_gpu,
                                           np.array([roi[0], roi[1]], dtype=np.int32),  # start_row, start_col in image
                                           np.int32(image_start_row),
                                           np.int32(image_start_row+part.shape[0]))
            # For the next round, we have to adapt the start column
            # by the height of the current part
            image_start_row += part.shape[0]
        assert image_start_row == image.shape[0], "${}".format(image.shape, image_start_row)
        for i in range(partitions):
            cl.enqueue_copy(self.queue, outputs[i], outputs_gpu[i])
        self.queue.finish()
        for i in range(partitions):
            output_final += outputs[i]
        del outputs
        del outputs_gpu
        del target_gpu
        del image_gpu
        self.queue.flush()
        gc.collect()
        idx = np.array(np.unravel_index(output_final.argmin(), output_final.shape))
        return output_final, idx+roi[0:2], output_final[idx[0], idx[1]]
