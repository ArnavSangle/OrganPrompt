import numpy as np
from src.tumor.index import overlap, proximity, blended_score


def _cube(shape, sl):
    m = np.zeros(shape, bool); m[sl] = True; return m


def test_overlap_full_and_none():
    T = _cube((10,10,10), np.s_[2:5,2:5,2:5])
    assert overlap(T, T) == 1.0
    O = _cube((10,10,10), np.s_[7:9,7:9,7:9])
    assert overlap(T, O) == 0.0


def test_overlap_partial():
    T = _cube((10,10,10), np.s_[0:4,0:1,0:1])   # 4 voxels
    O = _cube((10,10,10), np.s_[0:2,0:1,0:1])   # overlaps 2
    assert abs(overlap(T, O) - 0.5) < 1e-6


def test_proximity_touching_is_high():
    T = _cube((10,10,10), np.s_[0:1,0:1,0:1])
    O = _cube((10,10,10), np.s_[1:2,0:1,0:1])   # 1 voxel away
    p = proximity(T, O, spacing=(1.0,1.0,1.0))
    assert 0.4 < p < 0.6                          # d=1mm -> 1/(1+1)=0.5


def test_proximity_uses_spacing_mm():
    T = _cube((10,10,10), np.s_[0:1,0:1,0:1])
    O = _cube((10,10,10), np.s_[1:2,0:1,0:1])
    near = proximity(T, O, spacing=(0.5,0.5,0.5)) # 0.5mm -> 1/1.5
    far  = proximity(T, O, spacing=(4.0,4.0,4.0)) # 4mm  -> 1/5
    assert near > far


def test_proximity_empty_is_zero():
    Z = np.zeros((5,5,5), bool)
    assert proximity(Z, Z, (1,1,1)) == 0.0


def test_blended_score_weights():
    T = _cube((10,10,10), np.s_[0:2,0:1,0:1])
    O = _cube((10,10,10), np.s_[1:3,0:1,0:1])   # overlaps 1 of 2
    s = blended_score(T, O, (1,1,1), alpha=1.0, beta=0.0)
    assert abs(s - 0.5) < 1e-6                    # overlap only
