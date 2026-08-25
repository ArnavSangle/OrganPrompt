"""Panel A input node: the TotalSegmentator CT slice with the 14x14 lattice.

Reuses node1_ct_slice.png rather than re-scanning the training split for a
liver slice -- it is already the 224x224 windowed slice panel A shows, so
overlaying the lattice on it is the same image with the patch grid made
visible.

Usage:
    python scripts/make_lattice_panelA.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.make_pipeline_node_images import OUT  # noqa: E402

src = OUT / "node1_ct_slice.png"
if not src.exists():
    raise SystemExit(f"missing {src} -- run make_pipeline_node_images.py first")

img = mpimg.imread(src)
fig, ax = plt.subplots(figsize=(2.0, 2.0))
ax.imshow(img, extent=[0, 224, 224, 0])
for k in range(1, 14):                    # 13 internal lines each way
    ax.axhline(k * 16, color="#888888", linewidth=0.3)
    ax.axvline(k * 16, color="#888888", linewidth=0.3)
ax.set_xlim(0, 224); ax.set_ylim(224, 0)
ax.axis("off")
out = OUT / "node0a_input_lattice_totalseg.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
print(f"[lattice] wrote {out}  (source {img.shape[1]}x{img.shape[0]})")
