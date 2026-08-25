"""Shared matplotlib style for every figure that goes into the paper.

All figure scripts should call `apply_paper_style()` before creating a figure, so
the paper's figures agree with each other and with the body text. Previously each
script carried its own rcParams block, which is how half of them ended up in
Helvetica/Arial while the rest were in the paper's serif.

The body font is Latin Modern (main.tex loads lmodern). TeX Live ships it as OTF
but does not install it as a system font, so matplotlib cannot see it until it is
registered here. Registration is a silent no-op without TeX Live, in which case
font.serif falls through to DejaVu Serif.
"""
from pathlib import Path

import matplotlib.pyplot as plt

_TEXLIVE_ROOTS = (Path("C:/texlive"), Path("/usr/share/texmf-dist"),
                  Path("/usr/local/texlive"))


def register_latin_modern() -> bool:
    """Register Latin Modern Roman with matplotlib. True if it was found."""
    import matplotlib.font_manager as fm
    found = False
    for root in _TEXLIVE_ROOTS:
        if not root.exists():
            continue
        for otf in root.glob("**/fonts/opentype/public/lm/lmroman10-*.otf"):
            try:
                fm.fontManager.addfont(str(otf))
                found = True
            except Exception:
                pass
    return found


def apply_paper_style(font_size: float = 9) -> None:
    """Set the shared rcParams. Call before creating any figure."""
    register_latin_modern()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Keep SVG text as text so figures stay editable in Illustrator/Inkscape
        # rather than needing a Python round-trip for a typo.
        "svg.fonttype": "none",
    })
