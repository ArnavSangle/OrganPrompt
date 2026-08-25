"""Package the full paper as a self-contained Overleaf upload.

Overleaf has no notion of the repo layout, so this flattens the tree the paper
actually depends on into one directory and zips it:

    main.tex                 graphicspath rewritten to the archive root
    sections/*.tex           unchanged
    references.bib           unchanged
    <figure>.png             every figure main.tex actually references
    totalseg/<figure>.png    kept in its subdirectory, since the \\includegraphics
                             call names that path explicitly

Only figures reachable from the .tex sources are copied, so an unused diagnostic
never inflates the archive. The script fails loudly if a referenced figure is
missing rather than shipping a package that will not compile.

Usage:
    python scripts/build_overleaf.py
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

# Two documents live in paper/. "full" is main.tex plus sections/; "short" is the
# condensed single-file main_short.tex. They are packaged separately because they
# are genuinely different papers, not two builds of one.
VARIANTS = {
    "full": {
        "main": "main.tex",
        "sections": True,
        "stage": PAPER / "overleaf_full",
        "zip": PAPER / "organprompt-overleaf-full.zip",
    },
    "short": {
        "main": "main_short.tex",
        "sections": False,
        "stage": PAPER / "overleaf_short",
        "zip": PAPER / "organprompt-overleaf-short.zip",
    },
}

# Where \graphicspath points in the repo, relative to paper/.
SEARCH_DIRS = [ROOT / "figures", ROOT / "figures" / "paper", ROOT / "diagnostics",
               ROOT / "figures" / "totalseg"]


def referenced_figures(cfg: dict) -> list[str]:
    """Every \\includegraphics target in this variant's sources."""
    pat = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}")
    texs = [PAPER / cfg["main"]]
    if cfg["sections"]:
        texs += sorted((PAPER / "sections").glob("*.tex"))
    found: list[str] = []
    for tex in texs:
        found += pat.findall(tex.read_text(encoding="utf-8"))
    return sorted(set(found))


def locate(name: str) -> Path:
    """Resolve a figure reference against the repo's graphicspath."""
    for d in SEARCH_DIRS:
        candidate = d / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def build(variant: str) -> int:
    cfg = VARIANTS[variant]
    stage, zip_path = cfg["stage"], cfg["zip"]

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    if cfg["sections"]:
        (stage / "sections").mkdir()

    # The main file, with graphicspath collapsed to the archive root. Everything
    # else about it is left alone so the repo stays the source of truth.
    main_tex = (PAPER / cfg["main"]).read_text(encoding="utf-8")
    main_tex, n = re.subn(r"\\graphicspath\{[^\n]*\}", r"\\graphicspath{{./}}", main_tex)
    if n != 1:
        print(f"error: expected exactly 1 graphicspath in {cfg['main']}, found {n}")
        return 1
    (stage / cfg["main"]).write_text(main_tex, encoding="utf-8")

    if cfg["sections"]:
        for tex in sorted((PAPER / "sections").glob("*.tex")):
            shutil.copy2(tex, stage / "sections" / tex.name)
    shutil.copy2(PAPER / "references.bib", stage / "references.bib")

    missing: list[str] = []
    for ref in referenced_figures(cfg):
        try:
            src = locate(ref)
        except FileNotFoundError:
            missing.append(ref)
            continue
        # A reference like totalseg/foo.png keeps its subdirectory.
        dst = stage / ref
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if missing:
        print(f"error: figures referenced by {cfg['main']} not found:")
        for m in missing:
            print(f"  {m}")
        return 1

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(stage))

    files = sum(1 for p in stage.rglob("*") if p.is_file())
    print(f"[overleaf:{variant}] staged {files} files in {stage}")
    print(f"[overleaf:{variant}] wrote {zip_path} "
          f"({zip_path.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    wanted = sys.argv[1:] or ["full"]
    if wanted == ["all"]:
        wanted = list(VARIANTS)
    for v in wanted:
        if v not in VARIANTS:
            print(f"error: unknown variant {v!r}; choose from {list(VARIANTS)} or 'all'")
            return 1
    return max(build(v) for v in wanted)


if __name__ == "__main__":
    sys.exit(main())
