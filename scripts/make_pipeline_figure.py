"""Method schematic: training objective (A) and inference + index (B).

The paper describes the pipeline in prose across two subsections, and a reader
has to assemble six steps from text: hook block 11, drop CLS, project 768->512,
L2-normalize, downsample the mask to the 14x14 grid, BCE per patch. At inference
the path changes (blocks 8-11 fused, no gradients, no mask). This figure shows
both so the difference between them is visible rather than inferred.

Palette matches scripts/make_lesion_figures.py: muted, print-safe,
validated colorblind-safe.

Usage:
    python scripts/make_pipeline_figure.py   # -> figures/paper/fig0_pipeline.{svg,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("figures/paper")

# Three muted hues carry the three paths; the loss/index endpoint and all
# connective annotation wear ink instead of a fourth hue. Four MUTED hues do not
# survive validation -- desaturating a 4-slot set drops adjacent pairs below the
# normal-vision separation floor, so the fourth slot is deliberately achromatic.
BLUE = "#3B6EA5"    # image path
BRICK = "#B4543A"   # text path
GREEN = "#1F8A66"   # similarity / mask
INK = "#1A1A1A"     # loss/index endpoint, arrows, annotation
LIGHT = "#F2F2F2"

EN = "–"  # matplotlib text is not LaTeX: "--" would render literally

def _register_latin_modern():
    """Match the paper body font (main.tex loads lmodern).

    TeX Live ships Latin Modern as OTF but does not install it as a system font,
    so matplotlib cannot see it until it is registered. No-op without TeX Live.
    """
    import matplotlib.font_manager as fm
    for root in (Path(r"C:/texlive"), Path("/usr/share/texmf-dist")):
        if not root.exists():
            continue
        for otf in root.glob("**/fonts/opentype/public/lm/lmroman10-*.otf"):
            try:
                fm.fontManager.addfont(str(otf))
            except Exception:
                pass


_register_latin_modern()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "svg.fonttype": "none",
})


def box(ax, x, y, w, h, label, color, sub=None, fs=7.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.0, edgecolor=color, facecolor=LIGHT, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fs, zorder=4)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=fs - 1.3, color=INK, style="italic", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, zorder=4)
    return (x, y, w, h)


def arrow(ax, p, q, color=INK, lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=8, linewidth=lw, color=color,
        zorder=2, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3))


def polyline(ax, pts, color=INK, lw=1.0):
    """Explicit right-angle routing, then a single arrowhead on the last leg."""
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=2,
                solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color, lw)


R = lambda b: (b[0] + b[2], b[1] + b[3] / 2)
L = lambda b: (b[0], b[1] + b[3] / 2)
T = lambda b: (b[0] + b[2] / 2, b[1] + b[3])
B_ = lambda b: (b[0] + b[2] / 2, b[1])


def panel_training(ax):
    ax.text(0.0, 1.05, "A", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")
    ax.text(0.05, 1.06, "Training: patch-level contrastive objective",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    h = 0.185
    y1, y2, y3 = 0.72, 0.40, 0.08          # vision, text, label rows

    ct = box(ax, 0.00, y1, 0.115, h, "CT slice", BLUE, "224$\\times$224")
    vit = box(ax, 0.145, y1, 0.145, h, "ViT-B/16", BLUE, "hook block 11")
    tok = box(ax, 0.320, y1, 0.150, h, "196 patch tokens", BLUE, "CLS dropped")
    prj = box(ax, 0.500, y1, 0.130, h, "visual.head", BLUE, "768 $\\to$ 512")

    pr = box(ax, 0.00, y2, 0.115, h, "Organ prompt", BRICK, "few-shot text")
    bert = box(ax, 0.145, y2, 0.145, h, "PubMedBERT", BRICK, f"blocks 8{EN}11")
    tem = box(ax, 0.320, y2, 0.150, h, "text embedding", BRICK, "512-d, $L_2$")

    gt = box(ax, 0.320, y3, 0.150, h, "GT organ mask", GREEN,
             "area-pool $\\to$ 14$\\times$14")

    sim = box(ax, 0.665, y2, 0.135, h, "per-patch\ncosine sim.", GREEN)
    loss = box(ax, 0.840, y2, 0.150, h, "BCE", INK,
               "$\\mathcal{L}_{patch}$,  $\\lambda = 0.5$")

    for a, b in [(ct, vit), (vit, tok), (tok, prj)]:
        arrow(ax, R(a), L(b), BLUE)
    for a, b in [(pr, bert), (bert, tem)]:
        arrow(ax, R(a), L(b), BRICK)

    # three paths converge on the similarity box
    polyline(ax, [R(prj), (0.6325, y1 + h / 2), (0.6325, y2 + h * 0.72),
                  (0.665, y2 + h * 0.72)], BLUE)
    arrow(ax, R(tem), (0.665, y2 + h * 0.42), BRICK)
    polyline(ax, [R(gt), (0.6325, y3 + h / 2), (0.6325, y2 + h * 0.16),
                  (0.665, y2 + h * 0.16)], GREEN)

    arrow(ax, R(sim), L(loss), INK)
    ax.text(0.915, y2 - 0.055, "supervises every patch,\nnot only the CLS token",
            ha="center", va="top", fontsize=6.6, color=INK)


def panel_inference(ax):
    ax.text(0.0, 1.05, "B", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")
    ax.text(0.05, 1.06,
            "Inference: no gradients, no classifier, no mask refiner",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    h = 0.30
    yt, yb = 0.62, 0.05                     # top row, bottom row
    lane = 0.50                             # clear horizontal band between rows

    fuse = box(ax, 0.00, yt, 0.150, h, f"ViT blocks 8{EN}11", BLUE, "averaged")
    head = box(ax, 0.180, yt, 0.125, h, "visual.head", BLUE, "$L_2$-norm")
    cos = box(ax, 0.335, yt, 0.135, h, "cosine sim.", GREEN, "raw, in $[-1,1]$")
    up = box(ax, 0.500, yt, 0.150, h, "14$\\times$14 $\\to$ 224", GREEN,
             "bilinear")
    thr = box(ax, 0.680, yt, 0.135, h, "per-class\nthreshold", GREEN)
    m2d = box(ax, 0.845, yt, 0.145, h, "2D organ mask", GREEN)

    for a, b in [(fuse, head), (head, cos), (cos, up), (up, thr), (thr, m2d)]:
        arrow(ax, R(a), L(b), BLUE if b is head else GREEN)

    stack = box(ax, 0.00, yb, 0.150, h, "stack slices", BLUE, "3D volume")
    comp = box(ax, 0.185, yb, 0.180, h, "competitive\nassignment", INK)
    dist = box(ax, 0.395, yb, 0.175, h, "overlap +\nKD-tree distance", INK)
    idx = box(ax, 0.600, yb, 0.180, h, "metastasis index", INK,
              "$\\alpha\\,$ov $+\\ \\beta\\,$prox")

    # wrap from the end of the top row back to the start of the bottom row,
    # routed through the empty lane so it crosses nothing
    polyline(ax, [B_(m2d), (0.9175, lane), (0.075, lane), T(stack)], INK)
    ax.text(0.50, lane + 0.018, "repeat per axial slice, all 11 organs",
            ha="center", va="bottom", fontsize=6.6, color=INK)

    for a, b in [(stack, comp), (comp, dist), (dist, idx)]:
        arrow(ax, R(a), L(b), INK)

    ax.text(0.275, yb - 0.035, "organs that lose their own peak\nare reported absent",
            ha="center", va="top", fontsize=6.6, color=INK)
    ax.text(0.855, yb + h * 0.5, "per lesion", ha="left", va="center",
            fontsize=6.6, color=INK)
    arrow(ax, R(idx), (0.845, yb + h * 0.5), INK)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2))
    for ax in axes:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.06, 1.0)
        ax.axis("off")
    panel_training(axes[0])
    panel_inference(axes[1])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03, hspace=0.20)
    fig.savefig(OUT / "fig0_pipeline.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig0_pipeline.png", dpi=400, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"[pipeline_fig] wrote {OUT / 'fig0_pipeline.png'}")


if __name__ == "__main__":
    main()
