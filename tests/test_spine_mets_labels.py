from pathlib import Path
import pytest
from src.data.spine_mets_labels import parse_levels, load_lesion_labels, VERTEBRA_LEVELS

XLSX = Path("data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG_Clinical.xlsx")


def test_parse_levels_splits_commas_and_spaces():
    assert parse_levels("T1, T2 T3, T4") == ["T1", "T2", "T3", "T4"]


def test_parse_levels_handles_empty():
    assert parse_levels(None) == []
    assert parse_levels("") == []
    assert parse_levels("No lesions observed") == []  # no valid level tokens


def test_parse_levels_dedupes_and_uppercases():
    assert parse_levels("t1, T1, l5") == ["T1", "L5"]


def test_vertebra_levels_cover_thoracolumbar():
    for lvl in ["T1", "T12", "L1", "L5"]:
        assert lvl in VERTEBRA_LEVELS


@pytest.mark.skipif(not XLSX.exists(), reason="clinical xlsx not present")
def test_load_lesion_labels_real_file():
    labels = load_lesion_labels(str(XLSX))
    assert "10355" in labels                       # known Mixed-lesion patient
    assert any(t == "mixed" for t in labels["10355"].values())
    # patient 10352 recorded "No lesions observed"
    assert labels.get("10352", {}) == {}
