import numpy as np

from scripts.make_organ_extent_figure import extent_slices


def test_extent_slices_pick_percentiles_of_the_organs_own_z_range():
    coverage = np.zeros(100)
    coverage[20:61] = 1.0                    # organ spans slices 20..60
    assert extent_slices(coverage, (10, 50, 90)) == [24, 40, 56]


def test_extent_slices_only_return_slices_containing_the_organ():
    coverage = np.zeros(50)
    coverage[[5, 6, 30, 31]] = 1.0           # two separate pieces
    out = extent_slices(coverage, (10, 50, 90))
    assert all(coverage[z] > 0 for z in out)
