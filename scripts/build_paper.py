"""Build the paper PDF and fail loudly on anything a reader would notice.

Plain `pdflatex` exits 0 even when every citation is undefined and tables run off
the page, so a green exit code means very little. This wrapper runs the full
pdflatex -> bibtex -> pdflatex -> pdflatex sequence and then greps the log for the
three failure classes that silently ship:

  * undefined references / citations  -> "??" and "[?]" in the PDF
  * overfull hboxes                   -> text running past the margin
  * missing graphics                  -> a figure that is simply absent

Usage:
    python scripts/build_paper.py            # build, report, exit nonzero on problems
    python scripts/build_paper.py --quick    # single pass (refs will be stale)
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).parent.parent / "paper"
# TeX Live on this machine is not on PATH by default.
TEXBIN = Path("C:/texlive/2025/bin/windows")


def _tool(name):
    exe = TEXBIN / f"{name}.exe"
    if exe.exists():
        return str(exe)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"[build_paper] cannot find {name} -- is TeX Live installed?")


def _run(tool, *args):
    r = subprocess.run([_tool(tool), *args], cwd=PAPER,
                       capture_output=True, text=True, errors="replace")
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="single pdflatex pass; cross-references will be stale")
    a = ap.parse_args()

    passes = [("pdflatex", ("-interaction=nonstopmode", "main.tex"))]
    if not a.quick:
        passes = [
            ("pdflatex", ("-interaction=nonstopmode", "main.tex")),
            ("bibtex", ("main",)),
            ("pdflatex", ("-interaction=nonstopmode", "main.tex")),
            ("pdflatex", ("-interaction=nonstopmode", "main.tex")),
        ]

    log = ""
    for tool, args in passes:
        code, out = _run(tool, *args)
        print(f"[build_paper] {tool} {' '.join(args)} -> exit {code}", flush=True)
        log = out  # keep the LAST pass's log; earlier passes always warn

    problems = []

    # Hard TeX errors.
    errs = [l for l in log.splitlines() if l.startswith("!")]
    problems += [f"TeX error: {e}" for e in errs]

    # pdflatex hard-wraps its log at ~79 characters, so a single warning is
    # routinely split mid-sentence across two lines. Matching per-line silently
    # misses any warning unlucky enough to straddle the wrap -- which is how an
    # undefined \cite slipped past an earlier version of this check while the
    # undefined \ref beside it was caught. Scan a de-wrapped copy instead.
    flat = re.sub(r"\s+", " ", log)

    # Undefined refs/citations -- these render as "??" or "[?]" in the PDF.
    # The prefix differs by who emits it: plain LaTeX says "LaTeX Warning:" for
    # \ref, but natbib says "Package natbib Warning:" for \cite. Requiring the
    # LaTeX prefix catches the former and silently ignores the latter.
    for m in re.finditer(
            r"(?:LaTeX|Package natbib) Warning: (Reference|Citation) "
            r"`([^']+)' on page (\S+) undefined",
            flat):
        kind, key, page = m.groups()
        problems.append(f"undefined {kind.lower()}: `{key}' on page {page}")

    # A \cite key with no entry in references.bib. bibtex reports this in its own
    # .blg, which is not part of the pdflatex log the loop above captured, so it
    # has to be read separately or the failure is invisible here.
    blg = PAPER / "main.blg"
    if blg.exists():
        for m in re.finditer(r"didn't find a database entry for \"([^\"]+)\"",
                             blg.read_text(errors="replace")):
            problems.append(f"no references.bib entry for `{m.group(1)}'")

    # Overfull boxes -- text past the margin. Only flag visually obvious ones;
    # sub-point overflows are invisible in print.
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)", flat):
        if float(m.group(1)) > 2.0:
            problems.append(f"Overfull hbox: {m.group(1)}pt too wide")

    # Missing graphics.
    for m in re.finditer(r"File `([^']+)' not found", flat):
        problems.append(f"missing graphic: {m.group(1)}")

    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        problems.append("main.pdf was not produced")

    print()
    if problems:
        print(f"[build_paper] {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    kb = pdf.stat().st_size // 1024
    print(f"[build_paper] clean build: {pdf} ({kb} KB)")
    print("[build_paper] no undefined references, no overfull boxes, "
          "no missing graphics")


if __name__ == "__main__":
    main()
