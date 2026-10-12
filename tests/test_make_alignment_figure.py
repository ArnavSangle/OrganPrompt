import numpy as np

from scripts.make_alignment_figure import pick_views


def test_pick_views_takes_the_plane_with_most_mask_in_each_direction():
    m = np.zeros((5, 10, 12), bool)
    m[1, 2:4, 3:5] = True              # 4 voxels on slice 1
    m[3, 6:9, 7:10] = True             # 9 voxels on slice 3
    m[2, 9, 0:12] = True               # a full row 9: most voxels, but off-centroid
    z, row, col = pick_views(m)
    assert z == 2                                   # slice 2 now holds 12 voxels
    assert row == 9                                 # centroid would give 7
    assert col == int(np.argmax(m.sum(axis=(0, 1))))
