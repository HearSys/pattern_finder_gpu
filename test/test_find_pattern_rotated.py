from pathlib import Path
import numpy as np
import warnings


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
print(sys.path[0])
from pattern_finder_gpu import center_roi_around, find_pattern_rotated, rotation_around



def test_center_roi_around():
    test_roi = [0, 0, 3, 5]
    assert np.allclose(center_roi_around((1, 2), (3, 5)), test_roi)
    test_roi = [0, 0, 1, 1]  # this is a 1x1 ROI
    assert np.allclose(center_roi_around((0, 0), (1, 1)), test_roi)


def test_center_roi_around_row_vs_column():
    row = 2
    col = 3
    h = 5
    w = 7
    roi = center_roi_around([row, col], [h, w])
    print(roi)
    assert roi[2] - roi[0] == h
    assert roi[3] - roi[1] == w


def test_warnings_for_even_roi_height_width():
    with warnings.catch_warnings(record=True) as ws:
        # Cause all warnings to always be triggered.
        #warnings.simplefilter("always")
        # Trigger a warning.
        h, w = (3, 4)
        center_roi_around((10, 10), (h, w))
        center_roi_around((10, 10), (h, w))
        # Verify some things
        assert len(ws) == 1, "This warning should only be raised once"
        assert issubclass(ws[-1].category, UserWarning)
        assert "(height, width)=({height}, {width})".format(height=h, width=w) in str(ws[-1].message)


def test_rotation_around():
    """The rotation matrix should contain the angle and it should be independet"""
    degs = [10, 0, -33, np.pi, np.pi/2, 15]
    centers = [(0, 0), (20, 3), (3, 30), (-10, 10), (-1, 1), (-5, -5)]
    for deg in degs:
        for center in centers:
            m = rotation_around(deg, center)
            assert np.allclose(deg, np.rad2deg(-np.arcsin(m.params[0,1])))
            #assert np.allclose(deg, np.rad2deg(np.arccos(m.params[0,0])))
            assert np.allclose(m.params[2, 2], 1.)  # this elem in the rot matrix should alwyas be 1
            assert np.allclose(m.params[2, :], [0., 0., 1.])  # row should be 0, 0, 1
