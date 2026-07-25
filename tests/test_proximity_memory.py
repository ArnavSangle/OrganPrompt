import numpy as np
from scipy.ndimage import distance_transform_edt

from src.tumor.index import min_distance_mm, proximity


def _edt_reference(tumor, organ, spacing):
    """The old full-volume EDT implementation, kept only as a correctness oracle."""
    d = distance_transform_edt(~organ, sampling=spacing)
    return float(d[tumor].min())


def test_kdtree_matches_edt_on_anisotropic_spacing():
    rng = np.random.RandomState(0)
    shape = (12, 20, 20)
    for _ in range(5):
        tumor = np.zeros(shape, bool)
        organ = np.zeros(shape, bool)
        for _ in range(6):
            tumor[tuple(rng.randint(0, s) for s in shape)] = True
            organ[tuple(rng.randint(0, s) for s in shape)] = True
        spacing = (2.5, 0.7, 0.7)   # realistic anisotropic CT voxel
        got = min_distance_mm(np.argwhere(tumor), np.argwhere(organ), spacing)
        assert np.isclose(got, _edt_reference(tumor, organ, spacing))


def test_proximity_scales_with_spacing():
    shape = (4, 4, 4)
    tumor = np.zeros(shape, bool); tumor[0, 0, 0] = True
    organ = np.zeros(shape, bool); organ[0, 0, 2] = True
    # 2 voxels apart in y; at 3mm spacing that is 6mm -> 1/(1+6)
    assert np.isclose(proximity(tumor, organ, (1.0, 1.0, 3.0)), 1.0 / 7.0)
    # same voxels, finer spacing -> nearer
    assert np.isclose(proximity(tumor, organ, (1.0, 1.0, 0.5)), 1.0 / 2.0)


def test_proximity_empty_organ_is_zero():
    shape = (4, 4, 4)
    tumor = np.zeros(shape, bool); tumor[0, 0, 0] = True
    assert proximity(tumor, np.zeros(shape, bool), (1, 1, 1)) == 0.0


def test_proximity_does_not_allocate_over_the_volume():
    """A big volume with tiny masks must stay cheap -- the EDT version allocated
    a float64 array the size of the whole volume here (~1.6GB)."""
    shape = (200, 512, 512)
    tumor = np.zeros(shape, bool); tumor[10, 100, 100] = True
    organ = np.zeros(shape, bool); organ[10, 100, 110] = True
    assert np.isclose(proximity(tumor, organ, (1.0, 1.0, 1.0)), 1.0 / 11.0)
