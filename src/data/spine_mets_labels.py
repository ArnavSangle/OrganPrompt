"""Parse the Spine-Mets-CT-SEG clinical spreadsheet into per-vertebra lesion labels.

The clinical xlsx (downloaded from TCIA, CC BY 4.0) has a sheet
``Spine-Mets-CT-SEG`` with one row per patient. Columns ``Blastic``, ``Lytic``,
and ``Mixed`` each list the affected vertebral levels as free text
(e.g. ``"T1, T2 T3"``). This module turns that into
``{case_id: {level: lesion_type}}``.
"""
import re
import openpyxl

VERTEBRA_LEVELS = (
    [f"C{i}" for i in range(1, 8)]
    + [f"T{i}" for i in range(1, 13)]
    + [f"L{i}" for i in range(1, 6)]
    + ["S1"]
)
_LEVEL_RE = re.compile(r"\b([CTLS])\s*0*([0-9]{1,2})\b", re.IGNORECASE)
_VALID = set(VERTEBRA_LEVELS)


def parse_levels(cell):
    """Split a spreadsheet cell into a deduped list of valid vertebral levels."""
    if not cell:
        return []
    out = []
    for prefix, num in _LEVEL_RE.findall(str(cell)):
        lvl = f"{prefix.upper()}{int(num)}"
        if lvl in _VALID and lvl not in out:
            out.append(lvl)
    return out


def load_lesion_labels(xlsx_path):
    """Return {case_id: {level: lesion_type}} with lesion_type in lytic/blastic/mixed."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Spine-Mets-CT-SEG"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    idx = {name: header.index(name) for name in ("Case", "Blastic", "Lytic", "Mixed")}
    labels = {}
    for r in rows[1:]:
        case = r[idx["Case"]]
        if case is None:
            continue
        case = str(case).strip()
        per_level = {}
        for kind in ("blastic", "lytic", "mixed"):
            for lvl in parse_levels(r[idx[kind.capitalize()]]):
                per_level[lvl] = kind      # mixed listed last -> wins on overlap
        labels[case] = per_level
    return labels
