from pathlib import Path
import numpy as np
import pytest
from src.data.spine_mets import load_patient, stride_coords
from src.data.spine_mets_labels import load_lesion_labels

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024")
CASES = ROOT / "spine_mets_ct_seg"


@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
def test_load_patient_builds_tumor_mask():
    labels = load_lesion_labels(str(ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    # pick a case that (a) exists on disk and (b) has >=1 labeled level
    case = next(c.name for c in sorted(CASES.iterdir())
                if c.is_dir() and labels.get(c.name))
    p = load_patient(str(ROOT), case, labels)
    assert p.ct.shape == p.tumor_mask.shape == p.lesion_type.shape
    assert p.tumor_mask.sum() > 0
    assert set(np.unique(p.lesion_type)).issubset({0, 1, 2, 3})
    assert p.lesion_type[p.tumor_mask].min() >= 1     # labeled where tumor is


@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
def test_load_patient_exposes_each_lesion_separately():
    """Per-lesion evaluation needs the levels kept apart, not merged.

    tumor_mask is a union, which is all a per-patient index needs. Scoring each
    lesion against its own expected organ set needs them individually, and the
    union must remain exactly their sum so per-patient results stay comparable.

    Coordinates rather than masks: a patient can carry 17 labelled levels, and 17
    volume-sized boolean masks is over a gigabyte, while a vertebra's coordinates
    are a couple of megabytes.
    """
    labels = load_lesion_labels(str(ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    case = next(c.name for c in sorted(CASES.iterdir())
                if c.is_dir() and labels.get(c.name))
    p = load_patient(str(ROOT), case, labels)
    assert p.lesion_coords, "no per-lesion coordinates exposed"
    assert set(p.lesion_coords) <= set(labels[case]), "only labelled levels"
    rebuilt = np.zeros(p.ct.shape, dtype=bool)
    for c in p.lesion_coords.values():
        rebuilt[c[:, 0], c[:, 1], c[:, 2]] = True
    assert np.array_equal(rebuilt, p.tumor_mask), "union must equal tumor_mask"


def test_stride_coords_keeps_only_sampled_slices_and_reindexes():
    """Striding the volume must stride the lesions identically.

    compute_index subsamples z for speed; lesion coordinates are in full-volume
    index space, so they have to be filtered to the retained slices and then
    renumbered. Getting the renumber wrong shifts every lesion along z and
    silently corrupts every distance.
    """
    coords = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 3], [4, 4, 4], [5, 5, 5]])
    got = stride_coords(coords, 2)
    assert got.tolist() == [[0, 1, 1], [1, 3, 3], [2, 4, 4]]


def test_stride_coords_is_identity_at_stride_one():
    coords = np.array([[0, 1, 1], [3, 2, 2]])
    assert np.array_equal(stride_coords(coords, 1), coords)


def test_stride_coords_can_empty_a_lesion():
    """A thin lesion sitting only on dropped slices vanishes. The caller has to
    notice and skip it rather than score an empty lesion as distance zero."""
    coords = np.array([[1, 0, 0], [3, 0, 0]])
    assert len(stride_coords(coords, 2)) == 0
