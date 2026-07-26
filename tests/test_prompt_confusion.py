import numpy as np
import pytest

import scripts.prompt_confusion as pc


def test_distractors_cover_both_phantom_bands():
    """The diagnostic is only meaningful if it names the right suspects.

    Femur phantoms sit at T1-T5 and bladder phantoms at T11-L3, so the
    distractor set has to contain structures that actually occupy those levels.
    A distractor list that misses ribs or the gallbladder cannot explain the two
    bands no matter what the similarities come out to.
    """
    names = set(pc.DISTRACTOR_TEMPLATES)
    assert {"rib", "humerus"} <= names, "no upper-thoracic bone distractors"
    assert {"gallbladder", "stomach"} <= names, "no thoracolumbar fluid distractors"
    assert all(len(t) > 80 for t in pc.DISTRACTOR_TEMPLATES.values()), \
        "distractor prompts must match the FEW_SHOT style, not be bare labels"


def test_cosine_matrix_is_rows_by_cols():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.7071068, 0.7071068], [-1.0, 0.0]], dtype=np.float32)
    sim = pc.cosine_matrix(a, b)
    assert sim.shape == (2, 3)
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert sim[0, 1] == pytest.approx(0.7071068, abs=1e-5)
    assert sim[0, 2] == pytest.approx(-1.0, abs=1e-5)
    assert sim[1, 0] == pytest.approx(0.0, abs=1e-5)


def test_confusion_z_scores_against_the_organs_own_baseline():
    """A raw similarity means nothing on its own.

    Every prompt in this vocabulary starts 'Axial CT scan of...', so any two of
    them are similar in absolute terms. What matters is whether a distractor is
    closer to an organ than that organ's OWN classmates are -- otherwise
    'femur is 0.8 similar to rib' is indistinguishable from 'femur is 0.8
    similar to liver', and explains nothing.
    """
    # organ 0 sits 0.5 from its classmates with no spread, and 0.9 from a distractor
    organ_sim = np.array([[np.nan, 0.5, 0.5, 0.5]])
    dist_sim = np.array([[0.9, 0.5]])
    z = pc.confusion_z(organ_sim, dist_sim)
    assert z.shape == (1, 2)
    assert z[0, 0] > 3.0, "a far-outlying distractor must score high"
    assert z[0, 1] == pytest.approx(0.0, abs=1e-6), "a typical distractor scores 0"


def test_confusion_z_ignores_the_self_similarity_diagonal():
    """An organ is 1.0 similar to itself; letting that into the baseline inflates
    the mean and hides exactly the confusions we are looking for."""
    organ_sim = np.array([[1.0, 0.5, 0.5, 0.5]])       # diagonal NOT masked
    masked = np.array([[np.nan, 0.5, 0.5, 0.5]])
    dist_sim = np.array([[0.9]])
    assert pc.confusion_z(masked, dist_sim)[0, 0] > pc.confusion_z(organ_sim, dist_sim)[0, 0]


def test_top_confusions_ranks_distractors_per_organ():
    sim = np.array([[0.1, 0.9, 0.5],
                    [0.8, 0.2, 0.3]])
    got = pc.top_confusions(sim, ["femur-left", "bladder"], ["rib", "gallbladder", "aorta"], k=2)
    assert [n for n, _ in got["femur-left"]] == ["gallbladder", "aorta"]
    assert [n for n, _ in got["bladder"]] == ["rib", "aorta"]
    assert got["femur-left"][0][1] == pytest.approx(0.9)
