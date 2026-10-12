import gc
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.fit_organ_stats as fos
from src.labels.templates import CLASS_NAMES

_IDX = {v: k for k, v in CLASS_NAMES.items()}


# --------------------------------------------------------------------------
# which patients the constants are fitted on
# --------------------------------------------------------------------------

def test_single_level_patients_are_excluded_by_default():
    """The evaluation set must never enter the fit.

    The 20 single-level patients are what scripts/validate_anatomy.py scores the
    method on. TYPICAL_PEAK was fitted on those same patients, so the calibration
    the evaluation depends on had already seen the evaluation data. Excluding them
    here is the entire reason this script exists.
    """
    labels = {"a": {"T2": "lytic"},                       # single level -> held out
              "b": {"T2": "lytic", "L4": "blastic"},      # multi level  -> fitting
              "c": {}}                                    # no lesions   -> fitting
    assert fos.fitting_cases(labels, {"a", "b", "c"}) == ["b", "c"]


def test_single_level_patients_can_be_opted_back_in():
    labels = {"a": {"T2": "lytic"}, "b": {"T2": "lytic", "L4": "blastic"}, "c": {}}
    assert fos.fitting_cases(labels, {"a", "b", "c"},
                             include_single_level=True) == ["a", "b", "c"]


def test_lesion_free_patients_are_kept_in_the_fitting_set():
    """9 of the 55 patients have no lesions listed at all.

    They are perfectly good scans for measuring an organ's response level, and
    they carry no evaluation signal, so dropping them would throw away a quarter
    of the fitting set for no reason. They are easy to lose because they do not
    appear in any lesion-level index -- membership has to be driven by what is on
    disk, not by the label table.
    """
    labels = {"a": {"T2": "lytic"}, "b": {}, "c": {}}
    assert fos.fitting_cases(labels, {"a", "b", "c"}) == ["b", "c"]


def test_cases_only_on_disk_are_used():
    labels = {"a": {"T2": "lytic", "L4": "lytic"}, "b": {}, "gone": {}}
    assert fos.fitting_cases(labels, {"a", "b"}) == ["a", "b"]


def test_a_case_on_disk_with_no_label_row_is_still_fitted():
    labels = {"a": {"T2": "lytic", "L4": "lytic"}}
    assert fos.fitting_cases(labels, {"a", "unlisted"}) == ["a", "unlisted"]


# --------------------------------------------------------------------------
# streaming reduction of one volume
# --------------------------------------------------------------------------

def test_volume_stats_returns_peak_sum_sumsq_and_count():
    v = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    peak, total, total_sq, count = fos.volume_stats(v)
    assert peak == pytest.approx(4.0)
    assert total == pytest.approx(10.0)
    assert total_sq == pytest.approx(30.0)
    assert count == 4


def test_volume_stats_is_independent_of_the_chunk_size():
    """Reduction is done in slabs to bound memory; the slab size must not show up
    in the answer, including when the last slab is short."""
    rng = np.random.RandomState(1)
    v = rng.uniform(-2, 3, size=(35, 6, 7)).astype(np.float32)
    one = fos.volume_stats(v, chunk=1)
    eight = fos.volume_stats(v, chunk=8)
    huge = fos.volume_stats(v, chunk=1000)
    for a, b in zip(one, eight):
        assert a == pytest.approx(b, rel=1e-12)
    for a, b in zip(one, huge):
        assert a == pytest.approx(b, rel=1e-12)


def test_volume_stats_handles_all_negative_similarity():
    """Similarity is not bounded below at 0 -- a peak initialised to 0 would be
    silently reported for an organ whose every voxel is negative."""
    v = np.full((2, 3, 3), -0.5, dtype=np.float32)
    v[0, 0, 0] = -0.2
    peak, _, _, _ = fos.volume_stats(v)
    assert peak == pytest.approx(-0.2)


# --------------------------------------------------------------------------
# cross-patient aggregation
# --------------------------------------------------------------------------

def test_streaming_aggregation_matches_stats_over_the_concatenated_data():
    """The test that actually matters.

    A naive streaming sd is easy to get subtly wrong: averaging the per-patient
    sds, or weighting patients equally instead of by voxel count, both look
    plausible and both give the wrong number. The pooled mean/sd must equal what
    numpy gives over every voxel of every patient concatenated into one array.
    Patients here differ in both shape and scale so the wrong answers separate.
    """
    rng = np.random.RandomState(0)
    vols = [rng.uniform(lo, hi, size=shape).astype(np.float32)
            for lo, hi, shape in [(-3.0, 5.0, (3, 4, 5)),
                                  (0.0, 0.5, (7, 2, 2)),
                                  (10.0, 11.0, (1, 9, 9))]]
    acc = fos.OrganStatsAccumulator()
    for i, v in enumerate(vols):
        acc.add_patient(f"case{i}", {"liver": fos.volume_stats(v)})
    got = acc.result()["liver"]

    pooled = np.concatenate([v.ravel() for v in vols]).astype(np.float64)
    assert got["n_voxels"] == pooled.size
    assert got["n_patients"] == 3
    assert got["mean"] == pytest.approx(pooled.mean(), rel=1e-9)
    assert got["sd"] == pytest.approx(pooled.std(), rel=1e-9)
    # the two plausible wrong answers must be excluded, not merely unasserted
    assert got["sd"] != pytest.approx(float(np.mean([v.std() for v in vols])), rel=1e-3)
    assert got["mean"] != pytest.approx(float(np.mean([v.mean() for v in vols])), rel=1e-3)


def test_peak_mean_is_the_mean_of_per_patient_peaks():
    """Must match TYPICAL_PEAK's definition exactly, or it is not a drop-in.

    peak_mean replaces a constant that was defined as the mean of each patient's
    maximum. The max of the maxima and a pooled max are both single-patient
    numbers and would silently shift the whole calibration.
    """
    peaks = [0.10, 0.20, 0.60]
    acc = fos.OrganStatsAccumulator()
    for i, p in enumerate(peaks):
        acc.add_patient(f"case{i}", {"liver": (p, 0.0, 0.0, 1)})
    got = acc.result()["liver"]
    assert got["peak_mean"] == pytest.approx(0.30)
    assert got["peak_mean"] != pytest.approx(0.60)          # not max-of-maxima
    assert got["peak_sd"] == pytest.approx(float(np.std(peaks)))


def test_organs_are_aggregated_independently():
    acc = fos.OrganStatsAccumulator()
    acc.add_patient("a", {"liver": (0.4, 4.0, 8.0, 4), "spleen": (0.1, 0.0, 0.0, 4)})
    acc.add_patient("b", {"liver": (0.6, 4.0, 8.0, 4), "spleen": (0.3, 0.0, 0.0, 4)})
    res = acc.result()
    assert res["liver"]["peak_mean"] == pytest.approx(0.5)
    assert res["spleen"]["peak_mean"] == pytest.approx(0.2)
    assert res["liver"]["n_voxels"] == 8


def test_accumulator_records_the_per_patient_peaks_for_provenance():
    acc = fos.OrganStatsAccumulator()
    acc.add_patient("a", {"liver": (0.4, 0.0, 0.0, 1)})
    acc.add_patient("b", {"liver": (0.6, 0.0, 0.0, 1)})
    assert acc.per_case == {"a": {"liver": pytest.approx(0.4)},
                            "b": {"liver": pytest.approx(0.6)}}


def test_accumulator_holds_no_volume_sized_state():
    """Memory is the binding constraint: 35 patients x 11 organs of retained
    volume would be hundreds of gigabytes. Only scalars may survive add_patient."""
    acc = fos.OrganStatsAccumulator()
    for i in range(5):
        v = np.zeros((4, 8, 8), dtype=np.float32)
        acc.add_patient(f"case{i}", {"liver": fos.volume_stats(v)})
    for obj in vars(acc).values():
        assert not isinstance(obj, np.ndarray)
        if isinstance(obj, dict):
            assert not any(isinstance(x, np.ndarray) for x in obj.values())


# --------------------------------------------------------------------------
# per-patient streaming
# --------------------------------------------------------------------------

def test_patient_stats_reduces_each_organ_without_stacking(monkeypatch):
    """One organ resident at a time -- never an [11,D,H,W] array.

    Each yielded volume must be unreachable by the time the next one is
    generated, which is what keeps peak memory at one organ instead of eleven on
    an uncropped scan.
    """
    def fake_iter(ct, model, tf, device, organs=None):
        organs = list(range(len(CLASS_NAMES))) if organs is None else organs
        prev = None
        for oc in organs:
            if prev is not None:
                gc.collect()
                assert prev() is None, "caller still holds the previous organ"
            v = np.full(ct.shape, 0.1 * (oc + 1), dtype=np.float32)
            prev = weakref.ref(v)
            yield oc, v
            del v               # drop the generator's own reference

    monkeypatch.setattr(fos, "iter_organ_similarity", fake_iter)
    stats = fos.patient_stats(np.zeros((2, 4, 4), np.float32), object(), None, "cpu")
    assert set(stats) == set(CLASS_NAMES.values())
    assert stats["bladder"][0] == pytest.approx(0.1)
    assert stats[CLASS_NAMES[10]][0] == pytest.approx(1.1)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def _fake_root(tmp_path, cases):
    root = tmp_path / "root"
    (root / "spine_mets_ct_seg").mkdir(parents=True)
    for c in cases:
        (root / "spine_mets_ct_seg" / c).mkdir()
    return root


def _stub_pipeline(monkeypatch, labels, loader=None, shapes=None):
    monkeypatch.setattr(fos, "load_lesion_labels", lambda p: labels)
    monkeypatch.setattr(fos, "load_organcam", lambda ckpt, device: (object(), None))

    def default_loader(root, case, lbl):
        return SimpleNamespace(case=case, ct=np.full((6, 4, 4), 0.25, np.float32),
                               spacing=(1.0, 1.0, 1.0))

    monkeypatch.setattr(fos, "load_patient", loader or default_loader)

    def fake_iter(ct, model, tf, device, organs=None):
        if shapes is not None:
            shapes.append(ct.shape)
        organs = list(range(len(CLASS_NAMES))) if organs is None else organs
        for oc in organs:
            yield oc, np.full(ct.shape, float(ct.mean()), dtype=np.float32)

    monkeypatch.setattr(fos, "iter_organ_similarity", fake_iter)


def test_run_fits_on_the_held_out_patients_only(tmp_path, monkeypatch):
    labels = {"a": {"T2": "lytic"},                    # evaluation patient
              "b": {"T2": "lytic", "L4": "lytic"},
              "c": {}}
    root = _fake_root(tmp_path, ["a", "b", "c"])
    _stub_pipeline(monkeypatch, labels)
    out = tmp_path / "stats.json"
    fos.run(str(root), "ckpt", "cpu", str(out), stride=1)
    data = json.loads(out.read_text())
    assert data["cases"] == ["b", "c"]
    assert data["n_patients"] == 2
    assert data["include_single_level"] is False
    assert set(data["organs"]) == set(CLASS_NAMES.values())
    assert data["organs"]["liver"]["n_patients"] == 2


def test_run_records_stride_and_applies_it_to_the_volume(tmp_path, monkeypatch):
    """Stride is provenance, not just speed: the constants describe the volume
    that was actually measured, so the file has to say which one that was."""
    labels = {"b": {"T2": "lytic", "L4": "lytic"}}
    root = _fake_root(tmp_path, ["b"])
    shapes = []
    _stub_pipeline(monkeypatch, labels, shapes=shapes)
    out = tmp_path / "stats.json"
    fos.run(str(root), "ckpt", "cpu", str(out), stride=2)
    assert shapes == [(3, 4, 4)]                    # 6 slices, every 2nd
    assert json.loads(out.read_text())["stride"] == 2


def test_run_skips_a_patient_that_fails_to_load(tmp_path, monkeypatch, capsys):
    """A multi-hour run must not be lost to one unreadable DICOM series."""
    labels = {c: {"T2": "lytic", "L4": "lytic"} for c in ("a", "b", "c")}
    root = _fake_root(tmp_path, ["a", "b", "c"])

    def loader(root_, case, lbl):
        if case == "b":
            raise RuntimeError("CT or SEG series missing")
        return SimpleNamespace(case=case, ct=np.zeros((4, 4, 4), np.float32),
                               spacing=(1.0, 1.0, 1.0))

    _stub_pipeline(monkeypatch, labels, loader=loader)
    out = tmp_path / "stats.json"
    fos.run(str(root), "ckpt", "cpu", str(out), stride=1)
    data = json.loads(out.read_text())
    assert data["cases"] == ["a", "c"]
    assert data["skipped"] == ["b"]
    # the failed patient contributes to no organ at all -- not a partial record
    assert all(v["n_patients"] == 2 for v in data["organs"].values())
    assert "SKIP" in capsys.readouterr().out


def test_run_does_not_half_count_a_patient_that_fails_mid_stream(tmp_path, monkeypatch):
    """A failure after 6 of 11 organs must discard all 6.

    Folding organs into the accumulator as they arrive would leave that patient
    counted for some organs and not others, so peak_mean would be averaged over
    different patient sets per organ without anything reporting it.
    """
    labels = {c: {"T2": "lytic", "L4": "lytic"} for c in ("a", "b")}
    root = _fake_root(tmp_path, ["a", "b"])
    monkeypatch.setattr(fos, "load_lesion_labels", lambda p: labels)
    monkeypatch.setattr(fos, "load_organcam", lambda ckpt, device: (object(), None))
    state = {"case": None}

    def loader(root_, case, lbl):
        state["case"] = case
        return SimpleNamespace(case=case, ct=np.zeros((4, 4, 4), np.float32),
                               spacing=(1.0, 1.0, 1.0))

    monkeypatch.setattr(fos, "load_patient", loader)

    def fake_iter(ct, model, tf, device, organs=None):
        organs = list(range(len(CLASS_NAMES))) if organs is None else organs
        for oc in organs:
            if state["case"] == "b" and oc == 6:
                raise RuntimeError("CUDA out of memory")
            yield oc, np.full(ct.shape, 0.2, np.float32)

    monkeypatch.setattr(fos, "iter_organ_similarity", fake_iter)
    out = tmp_path / "stats.json"
    fos.run(str(root), "ckpt", "cpu", str(out), stride=1)
    data = json.loads(out.read_text())
    assert data["cases"] == ["a"]
    assert data["skipped"] == ["b"]
    assert all(v["n_patients"] == 1 for v in data["organs"].values())


def test_run_honours_max_cases_and_the_single_level_optin(tmp_path, monkeypatch):
    labels = {"a": {"T2": "lytic"}, "b": {"T2": "lytic", "L4": "lytic"}, "c": {}}
    root = _fake_root(tmp_path, ["a", "b", "c"])
    _stub_pipeline(monkeypatch, labels)
    out = tmp_path / "stats.json"
    fos.run(str(root), "ckpt", "cpu", str(out), stride=1, max_cases=1,
            include_single_level=True)
    data = json.loads(out.read_text())
    assert data["cases"] == ["a"]
    assert data["include_single_level"] is True


def test_run_output_carries_every_field_the_calibration_needs(tmp_path, monkeypatch):
    labels = {"b": {"T2": "lytic", "L4": "lytic"}}
    root = _fake_root(tmp_path, ["b"])
    _stub_pipeline(monkeypatch, labels)
    out = tmp_path / "stats.json"
    fos.run(str(root), "ckpt/v5.pt", "cpu", str(out), stride=1)
    data = json.loads(out.read_text())
    for key in ("root", "checkpoint", "stride", "cases", "skipped", "n_patients",
                "include_single_level", "organs", "per_case"):
        assert key in data, key
    assert data["checkpoint"] == "ckpt/v5.pt"
    for name, v in data["organs"].items():
        for key in ("peak_mean", "peak_sd", "mean", "sd", "n_patients", "n_voxels"):
            assert key in v, (name, key)
    assert set(data["per_case"]) == {"b"}


def test_accumulator_records_per_patient_moments_for_crossfitting():
    """src/tumor/crossfit.fold_stats re-pools mean/sd per fold, which needs each
    patient's raw (sum, sumsq, count), not just the pooled totals."""
    acc = fos.OrganStatsAccumulator()
    acc.add_patient("a", {"liver": (0.4, 2.0, 3.0, 4)})
    acc.add_patient("b", {"liver": (0.6, 1.0, 5.0, 2)})
    assert acc.per_case_moments == {"a": {"liver": [2.0, 3.0, 4]},
                                    "b": {"liver": [1.0, 5.0, 2]}}


def test_run_output_carries_per_case_moments(tmp_path, monkeypatch):
    labels = {"b": {"T2": "lytic", "L4": "lytic"}}
    root = _fake_root(tmp_path, ["b"])
    _stub_pipeline(monkeypatch, labels)
    out = tmp_path / "stats.json"
    fos.run(str(root), "ckpt/v5.pt", "cpu", str(out), stride=1)
    data = json.loads(out.read_text())
    assert set(data["per_case_moments"]) == {"b"}
    for name, m in data["per_case_moments"]["b"].items():
        assert len(m) == 3 and m[2] > 0, name
