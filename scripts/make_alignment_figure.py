"""Multiplanar check of CT / involved-vertebra alignment (BSPC revision, R5 point 3).

For the two cases shown in the qualitative figure, draws the involved vertebra
over the CT four ways: axial as the submitted version placed it (SEG frames
pasted index-for-index, which ignores the SEG's reversed column direction), and
axial, sagittal and coronal with the geometry-based placement now used for every
distance (src/data/spine_mets.py). No model is involved.

    python scripts/make_alignment_figure.py
"""
import sys
from pathlib import Path

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._figstyle import apply_paper_style
from src.data.spine_mets import find_series, load_ct_volume, load_vertebra_masks, \
    _level_from_label

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024/spine_mets_ct_seg")
CASES = [("10355", "T3"), ("13683", "T4")]
OUT = Path("paper/figures_for_upload/alignment_multiplanar.png")
BRICK = "#B4543A"
WINDOW = (-250.0, 750.0)


def pick_views(mask):
    """(axial z, coronal row, sagittal col): in each direction the plane that cuts
    the most mask voxels. The centroid of a vertebra sits in the spinal canal, so
    a centroid plane shows little of the bone."""
    return (int(np.argmax(mask.sum(axis=(1, 2)))),
            int(np.argmax(mask.sum(axis=(0, 2)))),
            int(np.argmax(mask.sum(axis=(0, 1)))))


def legacy_mask(seg_path, ct_paths, level):
    """The submitted placement: SEG frame pasted onto the CT slice matched by z,
    index-for-index, with no in-plane geometry. Reproduced only for the figure."""
    seg = pydicom.dcmread(seg_path)
    frames = seg.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    ct_z = np.array([float(pydicom.dcmread(p, stop_before_pixels=True)
                           .ImagePositionPatient[2]) for p in ct_paths])
    seg_level = {int(s.SegmentNumber): _level_from_label(s.SegmentLabel)
                 for s in seg.SegmentSequence}
    out = np.zeros((len(ct_paths), int(seg.Rows), int(seg.Columns)), bool)
    for i, fg in enumerate(seg.PerFrameFunctionalGroupsSequence):
        num = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        if seg_level.get(num) != level:
            continue
        zpos = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
        z = int(np.argmin(np.abs(ct_z - zpos)))
        if abs(ct_z[z] - zpos) <= 2.0:
            out[z] |= frames[i].astype(bool)
    return out


def _show(ax, img, mask, aspect=1.0):
    ax.imshow(np.clip(img, *WINDOW), cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1],
              aspect=aspect, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([BRICK]),
              alpha=0.75, aspect=aspect, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    apply_paper_style(font_size=10)
    fig, axes = plt.subplots(len(CASES), 4, figsize=(6.5, 3.9),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    titles = ["Axial, submitted\n(index-for-index)", "Axial, corrected",
              "Sagittal, corrected", "Coronal, corrected"]
    for r, (case, level) in enumerate(CASES):
        ct_paths, seg_path = find_series(str(ROOT / case))
        vol, (dz, dy, dx) = load_ct_volume(ct_paths)
        good = load_vertebra_masks(seg_path, ct_paths)[level]
        old = legacy_mask(seg_path, ct_paths, level)
        z, row, col = pick_views(good)
        mean_old = float(vol[old].mean())
        mean_new = float(vol[good].mean())
        print(f"{case}:{level} mean HU in mask: submitted {mean_old:.0f}, "
              f"corrected {mean_new:.0f}")
        _show(axes[r, 0], vol[z], old[z])
        _show(axes[r, 1], vol[z], good[z])
        # sagittal: rows = z (flip so head is up), cols = image rows (anterior left)
        _show(axes[r, 2], vol[::-1, :, col], good[::-1, :, col], aspect=dz / dy)
        _show(axes[r, 3], vol[::-1, row, :], good[::-1, row, :], aspect=dz / dx)
        axes[r, 0].set_ylabel(f"{case} {level}")
        axes[r, 0].text(0.03, 0.04, f"{mean_old:.0f} HU", color="white",
                        transform=axes[r, 0].transAxes, fontsize=9)
        axes[r, 1].text(0.03, 0.04, f"{mean_new:.0f} HU", color="white",
                        transform=axes[r, 1].transAxes, fontsize=9)
        if r == 0:
            for ax, t in zip(axes[0], titles):
                ax.set_title(t, fontsize=10)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
