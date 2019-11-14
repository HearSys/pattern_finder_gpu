
// OpenCL Kernel
// https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf


kernel void convolve_image(__read_only image2d_t image,
                           __read_only image2d_t pattern,
                           __global __write_only float* output,
                           sampler_t sampler,
                           const int2 top_left_rc,   // image (row, column)
                           const int image_start_r,  // where in the image row to start writing (for partitions)
                           const int image_end_r)    // end row writing (for partitions)
{
   // From a birds-eye-view what this kernel does is to compute a similarity
   // measure almost like a convolve operation, but with a squared distance measure
   // that is multipied with the alpha channel of the pattern as a mask (to ignore or
   // down-weight parts of the pattern). Therefore, I think, one cannot express the same
   // operation with a FFT because I don't know how to implement weighting and/or maskoing
   // with the FFT.

   // We can doe this embarrassingly parallel. And this OpenCL function call just computes
   // the value of a single pixel.

   // The `output` image (`output` is our region-of-interest (ROI)) is set up outside of
   // this function and we just get the location of the single pixel that we should
   // compute with the `get_global_id()` method.

   // The output just contains float values (NOT in scale 0..1) for how similar
   // pattern and image are at that pixel. Lower value mean "more similar".
   // `top_left_rc` defines to which pixel coordinate the (0, 0) pixel in the
   // output corresponds, so by chaning this variable, we can "move" the pattern over
   // the image.

    // Note, that OpenCL uses column-major ("fortran"-style) storage, so
    // in the end we have to access the pixel data (through a sampler) in
    // (column, row) order. But we stick to naming like "C"-style row-major
    // ordering and just do the flip in the two nested for-loops in the loop
    // variables c and r, which are then used according to OpenCL's convention.
    const int2 pos = {get_global_id(0), get_global_id(1)};

    // The pattern should be at least 3x3 and have an odd width and height
    const int pattern_height = get_image_height(pattern);
    const int pattern_width = get_image_width(pattern);
    const int pattern_half_w = (pattern_width-1) / 2;
    const int pattern_half_h = (pattern_height-1) / 2;

    // 4-tuple of the RGBA values from the pattern image and RRB of the image.
    float4 pattern_color = (float4)(0.0, 0.0, 0.0, 0.0);
    float4 image_color = (float4)(0.0, 0.0, 0.0, 0.0);

    // The (column, row) of the center pixel in the pattern is computed
    // Note, that later one we use this as (column, row), therefore we switch
    // already here the .s0 and .s1 assignment
    // (s0 is the first entry, s1 second)
    int2 pattern_pos_cr = (int2)(0, 0);
    pattern_pos_cr.s1 = -top_left_rc.s0 - pos.s0 + pattern_half_h; // row
    pattern_pos_cr.s0 = -top_left_rc.s1 - pos.s1 + pattern_half_w; // column

    int2 image_pos_cr = (int2)(0, 0);
    image_pos_cr.s1 = -pattern_half_h + pos.s0 + top_left_rc.s0; // row
    image_pos_cr.s0 = -pattern_half_w + pos.s1 + top_left_rc.s1; // column

    // value for the output pixel we are going to compute in this kernel
    double value = 0.0;
    // the distance in the color-space between the image and the pattern
    double dist = 0.0;

    // Go through all pixel of the pattern
    for (int r = 0; r < pattern_height; r++) {
        for (int c = 0; c < pattern_width; c++) {
            pattern_color = read_imagef(pattern, sampler, (int2)(c, r));
            if (pattern_color.w > 0.0f) {
                image_color = read_imagef(image, sampler, image_pos_cr + (int2)(c, r) );
                dist = fast_distance(pattern_color.xyz, image_color.xyz);
                value += pattern_color.w * dist * dist;
            }
        }
    }
    // write the value for the single pixel that we just computed:
    output[pos.s0 * get_global_size(1) + pos.s1] = value;
}
