# Phase 2 figure captions

**Figure 1 — Per-vertebral-level localization accuracy.** Mean rank of the
correct organ(s) at each labelled vertebral level (lower is better), against
the chance mean rank for that level's expected-organ set size (grey dashed
line; chance = (n_organs+1)/(k+1) where k is the number of organs expected at
that level). Bars are patient-clustered across n=180 lesions / 46 patients;
per-level n is annotated inside each bar. Thoracic levels T1–T10 beat chance
clearly, including two perfect levels (T8, T9). Vermillion bars mark the
thoracolumbar-junction failure band (T11, T12, L1, L2, L5), which sits at or
below chance; L3 and L4 in between recover to beat chance clearly. This
per-level structure, not a single aggregate number, is the main localization
result.

**Figure 2 — Bone-metastasis index score distribution.** Per-patient index
score (max over organs of `0.5×overlap + 0.5×proximity`; proximity =
1/(1+distance_mm)) across n=46 labelled Spine-Mets-CT-SEG patients, computed
with the current (unleaked) calibration. Mean = 0.139 (vermillion), median =
0.083 (dashed grey) — the right-skewed shape (21/46 patients near zero, a
smaller cluster of clearly-involved cases at 0.25–0.5) reflects the expected
mix of indolent and locally aggressive metastases in this cohort, not a
calibration artifact.

**Figure 3 — Correct localization vs. a phantom, side by side.** Left column:
CT slice at the lesion's peak axial position with the ground-truth lesion
mask overlaid (red). Right column: raw patch-text cosine similarity for the
top-ranked organ at that same slice, thresholded to the top 2% of that
slice's own similarity distribution (organs routinely respond with uniformly
negative raw similarity across a slice — e.g. femur-right ranges -0.21 to
-0.03 here — and win only in relative, not absolute, terms, so an
absolute-zero cutoff would hide the real signal). **Top (blue):** patient
10355, level T3 — lung-left correctly ranked #1 at 19mm; the highlighted
region sits over both lung fields, strongest near the left lung adjacent to
the lesion. **Bottom (vermillion):** patient 13683, level T4 — femur-right
incorrectly ranked #1 at 8mm, ahead of the true candidates (heart,
lung-left, lung-right). The highlighted patches sit on rib cross-sections at
three separate points around the chest wall, not on any bone resembling a
femur shaft — visual confirmation of the diagnosed mechanism
(`scripts/prompt_confusion.py`): the femur prompt's appearance clause
("bright hyperdense cortical ring... central medullary canal") matches a rib
or humeral head in cross-section as well as it matches a real femur, and
nothing in the 11-class vocabulary was defined to outbid that confusion.
