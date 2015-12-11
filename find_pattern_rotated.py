"""

"""



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
    if height % 2 == 0:
        raise UserWarning("ROI with even height is not centered.")
    if width % 2 == 0:
        raise UserWarning("ROI with even width is not centered.")

    row, col = int(round(center_rc[0])), int(round(center_rc[1]))
    return (row - height//2,
            col - width//2,
            row + height//2 + 1,
            col + width//2 + 1)
