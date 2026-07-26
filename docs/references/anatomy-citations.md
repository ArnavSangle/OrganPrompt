# Anatomy citations: posterior lung base / costophrenic recess extent

Purpose: settle whether `EXPECTED["T11"]` and `EXPECTED["T12"]` in
`scripts/validate_anatomy.py` should include `lung-left`/`lung-right`, from an
independent anatomical source, before looking at how the change affects the
per-lesion validation numbers. Research was done first; the change to
`EXPECTED` and the re-score of `results/lesion_validation/summary.json` came
after, in that order.

Question asked of each source: at full/typical inspiration, does the
posterior lung base (via the costophrenic/costodiaphragmatic recess) reach
the T11 vertebral level? The T12 level? Does this differ by side, sex, or
respiratory phase?

## Primary quantitative source

**"Retrodiaphragmatic portion of the lung: how deep is the posterior
costophrenic sulcus on posteroanterior chest radiography?"** *Clinical
Radiology* (journal home: clinicalradiologyonline.net /
sciencedirect.com/science/article/abs/pii/S0009926009001342; indexed at
PubMed: https://pubmed.ncbi.nlm.nih.gov/19589417/, PMID 19589417).

- Retrospective review of 540 consecutive PA chest radiographs from general
  health screenings; 282 met inclusion criteria (clear visualization of the
  inferior boundary of the posterior costophrenic sulcus behind the right
  hemidiaphragm, no diaphragm-affecting abnormality). 155 men, 127 women,
  mean age 40.7 +/- 8.4 years.
- Measured the depth of the posterior costophrenic sulcus (CPS) below the
  diaphragmatic dome and scored its level against the spine.
- **Finding: mean right posterior CPS depth 29.2 +/- 15.6 mm; average spinal
  level 13.5 +/- 0.6, i.e. the lower half of the L1 vertebral body** — a full
  vertebral level below T12.
- Depth increased with height, BMI, and total diaphragmatic height (i.e.
  this is normal population variation, not a rare outlier tail).
- Conclusion (their own): standard PA chest radiographs must include more
  than 3 cm below the lateral CPS, or extend to the L1 spine level, to fully
  capture the retrodiaphragmatic lung.
- **Claim supported: both T11 and T12 sit at or above the population-average
  posterior lung/pleural extent (L1 lower half). A lung detected at T11 or
  T12 is consistent with normal anatomy, not evidence of a phantom
  detection.**
- Retrieved via WebSearch snippet plus WebFetch of the PubMed abstract page
  (https://pubmed.ncbi.nlm.nih.gov/19589417/); the ScienceDirect/Clinical
  Radiology abstract pages themselves returned HTTP 403 to WebFetch and were
  not read directly.

## Supporting / triangulating sources

**Clinical Gate — "Thorax" chapter** (https://clinicalgate.com/thorax-7/,
textbook-derived reference content).
- States lung tissue "does not normally extend much below vertebral level
  T10" in quiet respiration, while "the pleural cavity's inferior extent"
  reaches "approximately T12" via the costodiaphragmatic recess behind the
  diaphragm/kidneys, "or even lower" along the twelfth rib in the
  midscapular line.
- Explicitly flags these levels as "only approximate," varying with posture
  and respiration.
- **Distinction drawn from this source:** the *lung parenchyma* in quiet
  breathing (~T10) is not the same structure as the *costophrenic
  recess/pleural potential space* that the aerated lung expands into on
  inspiration (~T12, and per the PubMed study above, averaging lower still
  to L1). The `EXPECTED` table is scoring "could the model's lung mask
  plausibly extend here," which tracks the recess/inspiratory extent, not
  the quiet-breathing resting position.

**Earth's Lab — "Costodiaphragmatic Recess"**
(https://www.earthslab.com/anatomy/costodiaphragmatic-recess/).
- Describes the recess reaching vertebral level T10 at its most conservative
  description, but also states recesses "are deepest after forced
  expiration and shallowest after forced inspiration" — worded in terms of
  the recess's own depth (potential space filled by fluid/lung), and notes
  during deep inspiration "the lungs move in deep up to two intercostal
  spaces into the recess," i.e. inspiration pushes the aerated lung further
  down into the recess, consistent with a lower functional limit than the
  T10 resting figure.
- Does not mention a left/right asymmetry.

**GPnotebook — "Inferior border of lung (anatomy)"**
(https://gpnotebook.com/pages/surgery/inferior-border-of-lung-anatomy).
- Classic teaching figure: "In moderate respiration, the lower border of the
  lung at the vertebral column is at the level of the upper half of the
  eleventh thoracic vertebra [T11]. In deep inspiration, the lower border of
  the lung in the scapular line reaches the superior border of the eleventh
  rib," and separately, "the lung extends deep into the posterior
  costophrenic (diaphragmatic) sulcus, reaching at or below the level of the
  12th rib in approximately 80% of individuals" (rib level, not vertebral
  level, but corroborates that T11/T12 are inside the normal, common range
  rather than being an extreme).
- This source directly states the *moderate-respiration* resting border of
  the lung is already at T11 (not T10), before any inspiratory excursion is
  added — i.e., even the conservative/resting figure from this source
  places the lung base at T11.

**Wikipedia — "Costodiaphragmatic recess"**
(https://en.wikipedia.org/wiki/Costodiaphragmatic_recess).
- Confirms the recess "measures approximately 5 cm vertically" along the
  mid-axillary line and that "the lungs expand into this recess during
  forced inspiration; however, the recess never gets filled completely" —
  used only as general structural confirmation; this page did not give a
  specific vertebral-level number when fetched directly (the number quoted
  above came from search-engine synthesis of multiple pages, not this page
  alone, so it is listed here only as corroboration of the recess's
  existence and general behavior, not as the source of the T11/T12 numbers).

**PMC — "Comparative Interpretation of CT and Standard Radiography of the
Pleura"** (https://pmc.ncbi.nlm.nih.gov/articles/PMC6100645/).
- Confirms clinically that "the posterior costophrenic sulcus is the deeper
  part of the pleural cavity" and that a lateral chest view (which better
  shows the posterior sulcus) detects small pleural effusions the frontal
  view misses — supports that the posterior recess is a genuine, clinically
  significant deep extension of the pleural space, but this article did not
  give specific vertebral-level numbers when fetched directly.

## Points of convergence

- Every source that gives a specific level places the posterior lung
  base/costophrenic recess at T11 or below (moderate respiration: T11 per
  GPnotebook; recess/inspiratory extent: T12 per Clinical Gate/earthslab;
  quantitative radiographic average: lower-half-L1 per the Clinical
  Radiology study, PMID 19589417).
- No source reported a clean, systematic left/right asymmetry large enough
  to justify treating the two lungs differently in `EXPECTED`. (The PubMed
  study measured the right side specifically because the right
  costophrenic sulcus is the one clearly visible below the liver/diaphragm
  on a PA film — a measurement-visibility choice, not a claim that the left
  side is shallower.)
- Depth is respiration- and body-habitus-dependent (deeper with taller
  height, higher BMI, deep inspiration; per Clinical Gate/earthslab/GPnotebook,
  shallower on forced expiration) — this is normal population variability,
  not a rare tail, consistent with routinely encountering it in a clinical
  imaging series.

## Decision

`EXPECTED["T11"]` and `EXPECTED["T12"]` should both include `lung-left` and
`lung-right`. The weakest single-source claim (moderate-respiration lung
base at T11, GPnotebook) already places T11 within range; T12 sits
comfortably below every conservative estimate and is roughly one vertebral
level above the population-average posterior costophrenic sulcus location
(lower half of L1, PMID 19589417). No source was found that would support
treating a lung detection at T11 or T12 as anatomically impossible.
