"""Panel A encoder-output node: a schematic of the 196 patch tokens.

Deliberately NOT data. What leaves the image encoder is a (196, 512) tensor --
there is no image in it. Any rendering (token L2 norm, a PCA component,
channel 0) would be a derived scalar that appears nowhere in the method, so
this draws uniform neutral cells instead: the 14x14 geometry survives the
encoder, but the values are not yet spatial. Label it "196 x 512-d tokens" in
draw.io rather than baking text into the PNG.

Same 0.3pt #888888 lattice as node5 / node0a / node0b so all the 14x14 nodes
read as one family.

Usage:
    python scripts/make_token_schematic_node.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.make_pipeline_node_images import OUT  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(np.full((224, 224), 0.88), cmap="gray", vmin=0, vmax=1)
    for k in range(1, 14):                    # 13 internal lines each way
        ax.axhline(k * 16, color="#888888", linewidth=0.3)
        ax.axvline(k * 16, color="#888888", linewidth=0.3)
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.axis("off")

    out = OUT / "node6_patch_tokens.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"[tokens] wrote {out}  (schematic, 14x14 = 196 cells, no data)")


if __name__ == "__main__":
    main()
