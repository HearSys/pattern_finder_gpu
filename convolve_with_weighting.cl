
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
