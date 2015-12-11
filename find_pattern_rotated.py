"""

"""


def center_roi_around(center_cr, size_hw):
    """
    Return a rectangular region of interest (ROI) of size `size_hw` around the
    given center coordinates. The returned ROI is defined as

        (start_column, start_row, end_column, end_row)

    where the start_column/row are ment to be *included* and the
    end_row/column are excluded.

    -   `center_cr`: Tuple of (column, row). The numbers will be rounded to the
        closest integer.
    -   `size_hw`: A tuple of (height, width) of the resulting ROI. If this
        numbers are even
    """
    height, width = size_hw
    if height % 2 == 0:
        raise UserWarning("ROI with even height is not centered.")
    if width % 2 == 0:
        raise UserWarning("ROI with even width is not centered.")
    col, row = int(round(center_cr[0])), int(round(center_cr[1]))
    return (col-height//2,
            row-width//2,
            1+col+height//2,
            1+row+width//2)


def test_center_roi_around():
    import numpy as np
    test_roi = [0, 0, 10, 10]
    assert np.allclose(center_roi_around((5, 5), (10, 10)), test_roi)
    test_roi = [0, 0, 1, 1]  # this is a 1x1 ROI
    assert np.allclose(center_roi_around((0, 0), (1, 1)), test_roi)
