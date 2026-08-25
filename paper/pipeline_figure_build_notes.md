# Pipeline figure — build reference

Hand-built replacement for `figures/paper/fig0_pipeline.png`, styled after
the Nature Sci Rep infographic (`paper/41598_2025_93718_Fig1_HTML.webp`).
Two rows: **Panel A = training** (left→right), **Panel B = inference**
(loops back on itself). 11 boxes total.

## Reusable sub-component: the encoder icon

Used 3 times — nodes 2, 4, 7 below. Build it once, duplicate.

```
grid (14x14) → [ Self-Attention ⊕loop ] → [ MLP/FFN ⊕loop ] → grid (14x14)
```

- **Grid stays the same size in and out.** No funnel, no shrinking to a
  point — a transformer never collapses the 196-token grid to a single
  latent vector. This is a real architectural fact, not a style choice:
  the paper's whole argument is that patch identity is preserved, never
  pooled. (Also: the very next box after this one bilinearly *upsamples*
  the grid — that only makes sense if nothing was collapsed first.)
- **Self-Attention processes before MLP/FFN, not after.** Order matters.
- **Two residual loops, one per sublayer**, each fully contained inside
  its own box — never one big loop spanning the whole encoder. Each loop:
  forks right where the arrow enters the sublayer, curves around the
  outside, rejoins at a small `⊕` circle right after the sublayer, before
  the next arrow continues.
- **No decoder, anywhere.** This is a headline claim of the paper (no
  gradients, no classifier, no second-stage mask refiner at inference) —
  don't draw anything that looks like one.
- Current build style: blue-filled boxes, white text, black main-flow
  arrows, green residual loops + green `⊕`, white background.
- Explanatory note (add once, next to the *first* `⊕` only — don't repeat
  on the second): **"output = input + sublayer(input)"**
- Font: Georgia or Times New Roman (closest Google-Drawings substitute for
  the paper's Latin Modern serif).

**Three copies needed:**
| Copy | Outer label | Extra |
|---|---|---|
| Node 2 | "Image encoder" — ViT-B/16, block 11 → 196 patch tokens | plain |
| Node 4 | "Text encoder" — PubMedBERT → 512-d embedding | **input/output differ — see below** |
| Node 7 | "Frozen encoder" — blocks 8–11, fused | **wrap whole icon in a light-blue box** to mark it frozen |

### Node 4 specifically — how it differs from node 2

The Self-Attention→MLP block itself is identical for both (same mechanism
in ViT and BERT) — copy it unchanged. Only the input/output shape changes,
because text has no 2D spatial structure and PubMedBERT's output pools to
one vector here, unlike the image path which never pools:

| | Node 2 (image) | Node 4 (text) |
|---|---|---|
| Input | 14×14 grid, 196 squares | **Not a grid** — a short row of ~4–6 squares (a sequence), or skip it and take one plain arrow in from node 3 directly |
| Block | Self-Attention (loop+⊕) → MLP/FFN (loop+⊕) | Identical — copy exactly |
| Output | 14×14 grid, same size as input | **One single box, no grid** — PubMedBERT pools the whole prompt to one 512-d vector, labeled "512-d embedding" |

**Real output image available:** `figures/paper/pipeline_nodes/node4_text_embedding.png`
— the actual liver embedding, pulled straight from `load_organcam`'s output
(row `liver_idx` of the real `(11, 512)` text-feature tensor, L2-norm
verified exactly 1.000). Rendered as a 1D strip on purpose, not a grid —
reshaping it into rows/columns would reintroduce the same "text has spatial
structure" mistake this node's redesign corrected. Use this in place of a
plain label box if you want the real thing instead of a placeholder.
Regenerate with `python scripts/make_embedding_viz.py --device cuda`.

## Wiring diagram

Layout reference only — boxes and arrows won't align pixel-perfect once
rebuilt in Drawings, treat this as the wiring, not a template to trace. What
actually matters to preserve: three lanes converging into one box in Panel A,
the grid staying the same size through both encoder icons (no funnel), the
light-blue frozen box on node 7 only, and the loop-back in Panel B routed
through empty space so it doesn't cross any other line.

```
════════════════════════════════════════════════════════════════════════
PANEL A — TRAINING
════════════════════════════════════════════════════════════════════════

 ┌──────────────┐     ┌────────────────────┐
 │  CT slice    │────▶│  Image encoder     │──┐
 │  224×224     │     │  ViT-B/16, blk 11  │  │
 │  [IMG]       │     │  [□□□□□□] 14×14    │  │   grid stays 14×14 the
 └──────────────┘     │  grid, NOT funneled│  │   whole way through —
                       │  [icon]            │  │   no bottleneck, no
                       └────────────────────┘  │   latent vector
                                                ▼
 ┌──────────────┐     ┌────────────────────┐  ┌─────────────────────┐   ┌──────────┐
 │ Organ prompt │────▶│  Text encoder      │─▶│ Patch similarity     │──▶│  Loss    │
 │ few-shot text│     │  PubMedBERT, 512-d │  │  → loss              │   │ BCE      │
 │ [text]       │     │  [icon]            │  │  [text]              │   │ λ = 0.5  │
 └──────────────┘     └────────────────────┘  └──────────▲───────────┘   │ [text]   │
                                                            │              └──────────┘
 ┌────────────────┐                                        │
 │ GT organ mask  │────────────────────────────────────────┘
 │ pooled to 14×14│
 │ [IMG — blocky] │        note: "supervises every patch,
 └────────────────┘              not only the CLS token"


                    ┌───────────────────────────────┐
                    │   checkpoint, frozen           │
              ══════▶  (connector between panels)   ══════▶
                    └───────────────────────────────┘


════════════════════════════════════════════════════════════════════════
PANEL B — INFERENCE   (whole model frozen, single forward pass, no grads)
════════════════════════════════════════════════════════════════════════

 ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
 │  Frozen encoder    │───▶│  Similarity map     │───▶│  2D organ mask      │
 │  blocks 8–11, fused│    │  cosine, raw        │    │  thresholded        │
 │  [□□□□□□] 14×14    │    │  (still 14×14)      │    │  (upsampled→224×224)│
 │  grid — same shape │    │  [IMG — case 10355, │    │  [IMG — same case/  │
 │  as node above,     │    │   T3, lung-left]    │    │   slice]            │
 │  light-blue box     │    └────────────────────┘    └──────────┬──────────┘
 │  = frozen           │                                          │
 │  [icon]             │                                          │
 └────────────────────┘                                           │
                                                                    │ loop back:
                          ┌─────────────────────────────────────────┘  "repeat per
                          │                                            axial slice,
                          ▼                                            all 11 organs;
 ┌────────────────────┐    ┌─────────────────────┐                    stack into 3D"
 │  Assign + score     │───▶│  Metastasis index   │
 │  competitive        │    │  α·overlap + β·prox │
 │  assignment,         │    │  [IMG — case 10355, │
 │  overlap + KD-tree   │    │   patient index]    │
 │  distance             │    └─────────────────────┘
 │  [text]              │
 └────────────────────┘
   note: "organs that lose their own peak are reported absent"
```

**Legend:** `[IMG]` = real generated/extracted image, not drawable by hand
(5 total — CT slice, GT mask, similarity map, 2D mask, metastasis index).
`[icon]` = generic hand-drawn shape, no data content (image encoder, text
encoder, frozen encoder — same icon reused for all three). `[text]` =
label/equation box only, no image (organ prompt, patch similarity→loss,
assign+score).

## Node 3 — Organ prompt, exact text

**Not a list.** `FEW_SHOT_TEMPLATES` in `src/labels/templates.py` is one
continuous multi-sentence string per organ, not several short items — a
bulleted "Item 1 / Item 2 / Item 3" box misrepresents the input shape.

Real liver prompt, verbatim (447 characters — matches the organ already
used in nodes 1 and 5, so the prompt describes the same organ the CT slice
and mask actually show):

> "Axial CT scan of the upper abdomen showing the liver. Dominates the LEFT
> portion of the image from the left edge toward and past the center, in the
> top half. The right lobe is largest and sits furthest left in the image.
> The left lobe tapers toward image center. Intrahepatic vessels (portal and
> hepatic veins) branch throughout as tubular structures. The gallbladder
> may appear as a small oval hypodense structure at the bottom of the liver
> mass."

Too long to fit the box legibly — truncate with a visible ellipsis so it
reads as "snippet of a longer real prompt," not as the complete input:

> "Axial CT scan of the upper abdomen showing the liver. Dominates the LEFT
> portion of the image... [truncated]"

## Full 11-node table

### Panel A — Training

| # | Label | Subtitle | Content |
|---|---|---|---|
| 1 | CT slice | 224×224 | **Real image** — see below |
| 2 | Image encoder | ViT-B/16, block 11 → 196 patch tokens | Encoder icon (no frozen box) |
| 3 | Organ prompt | few-shot text | Text only |
| 4 | Text encoder | PubMedBERT → 512-d embedding | Encoder icon (no frozen box) |
| 5 | GT organ mask | pooled to 14×14 | **Real image** — see below |
| 6 | Patch similarity → loss | cosine sim. → BCE, λ=0.5 | Text/equation box |

Connector 6 → 7: **"checkpoint, frozen"**

### Panel B — Inference (whole model frozen, single forward pass)

| # | Label | Subtitle | Content |
|---|---|---|---|
| 7 | Frozen encoder | blocks 8–11, fused | Encoder icon **+ light-blue frozen box** |
| 8 | Similarity map | cosine, raw | **Real image** — see below |
| 9 | 2D organ mask | thresholded | **Real image** — see below |
| 10 | Assign + score | competitive assignment, overlap + KD-tree distance | Text box |
| 11 | Metastasis index | α·overlap + β·proximity | **Real image** — see below |

Loop-back annotation (9 → 10): **"repeat per axial slice, all 11 organs"**
Note near node 10: **"organs that lose their own peak are reported absent"**

## The 5 real images — already generated, audited

`figures/paper/pipeline_nodes/`

| File | Node | Caption |
|---|---|---|
| `node1_ct_slice.png` | 1 | TotalSegmentator training slice, liver-windowed (soft tissue, −150 to 250 HU). |
| `node5_gt_mask_pooled.png` | 5 | Same slice's ground-truth liver mask, pooled to 14×14 by area interpolation + 0.5 threshold — the literal training-time operation, not a stylized redraw. |
| `node8_similarity_map.png` | 8 | Case 10355, T3 lesion slice — raw cosine similarity for the `lung-left` prompt, top 2% of the slice's own distribution highlighted. |
| `node9_organ_mask.png` | 9 | Same slice, thresholded to a binary mask. |
| `node11_metastasis_index.png` | 11 | Case 10355, whole-volume peak-tumor slice — **note: this is level L2/L3, a different slice than nodes 8/9's T3 lesion.** Same patient, deliberately different location, because this is the patient-level index (matching how `fig5` is built), not the T3-specific per-lesion rank. Index = 0.202, read from `results/bone_index/10355.json`. |

Case 10355 used throughout Panel B for continuity — it's the same "clean
detection" case already shown in `fig3_qualitative.png`. Node 11 breaks
continuity of *anatomy* on purpose (see caption above) while keeping
continuity of *patient*.

**Audit note on node 1/5:** the liver slice was picked as the best of only
421 candidates (an early-exit bug capped the search short of the full 3,698
available in the training split), not a true dataset-wide maximum. The image
itself is still correctly computed — just not exhaustively optimal. Rerun
`make_pipeline_node_images.py` without the cap if a better example matters.

Regenerate with: `python scripts/make_pipeline_node_images.py --device cuda`

## Progress

- [x] Encoder icon design finalized (grid→block→grid, correct order, both
      residual loops, explanatory note)
- [ ] 3 copies placed (node 2 plain / node 4 plain / node 7 + frozen box)
- [ ] All 5 real images dropped into their boxes
- [ ] Text-only boxes (3, 6, 10) built
- [ ] Two connectors + two annotations added
- [ ] Font pass (Georgia/Times New Roman throughout)
