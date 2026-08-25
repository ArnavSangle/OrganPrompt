"""Do the femur and bladder prompts describe structures we never labeled?

results/lesion_validation/scan_extent.json killed the field-of-view explanation
for the 24 bladder/femur phantoms: the patients that produce them have MORE
caudal coverage than the clean ones (median 122mm vs 88.5mm below the last
segmented vertebra), and 9 of 12 reach L5. The pelvis is in view precisely where
we claimed a bladder could not be.

What the phantoms do have is a sharp anatomical signature:

    femur   -> T1-T5, 10 of 11    (one stray at L4)
    bladder -> T11-L3, 13 of 13   (none anywhere else)

Two disjoint bands, and neither is an artifact of the impossible-top-1 filter --
a bladder would be flagged just as loudly at T1-T10 and never appears there.

Read the prompts against those bands and the suspects name themselves. The femur
template asks for "a bright hyperdense cortical ring surrounding lower-density
cancellous bone with a central medullary canal", which is a RIB in cross-section
at every thoracic level, and a humeral head at T1-T5. The bladder template asks
for "a smooth-walled fluid-filled hypodense oval", which is the gallbladder at
L1 and the stomach at T11-L1. Both templates also carry a location clause -- "of
the thigh", "of the pelvis" -- that the bands say is being ignored.

So the hypothesis is a MISSING CLASS, not a broken model: the appearance clause
matches a structure that is genuinely there, and nothing in our 11-class
vocabulary can outbid the wrong answer because the right answer was never
defined. That is a different failure from the one we have been chasing, and it
has a different fix.

This script tests it against the TEXT ENCODER ALONE -- no patients, no images, no
evaluation points spent. If the femur prompt embeds closer to a rib prompt than
to its own classmates, the confusion is upstream of any vision at all.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip import get_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.labels.templates import CLASS_NAMES, FEW_SHOT_TEMPLATES
from src.tumor.organ_volume import MODEL_NAME
import scripts.compute_index as ci

# Written in the FEW_SHOT_TEMPLATES style on purpose: same "Axial CT scan of..."
# opening, same appearance-then-neighbors structure. A bare label like "a rib"
# would embed differently for reasons of phrasing rather than anatomy, and the
# comparison would measure our writing instead of the model.
DISTRACTOR_TEMPLATES = {
    # --- upper thoracic, where the femur phantoms are (T1-T5) ---
    "rib": (
        "Axial CT scan of the chest showing a rib in transverse cross-section. "
        "Appears as a small bright hyperdense cortical ring enclosing lower-density "
        "cancellous bone, positioned along the curve of the chest wall at the outer "
        "edge of the image. Intercostal muscle borders it on either side along the "
        "chest wall; pleura and aerated lung lie toward the image center."
    ),
    "humerus": (
        "Axial CT scan of the shoulder showing the humerus in transverse cross-section. "
        "Appears as a bright hyperdense cortical ring surrounding lower-density "
        "cancellous bone with a central medullary canal, positioned at the far left or "
        "right edge of the image lateral to the chest wall. Deltoid muscle wraps around "
        "it laterally; the scapula and glenoid lie toward the image center."
    ),
    "scapula": (
        "Axial CT scan of the upper chest showing the scapula in cross-section. Appears "
        "as a thin curved hyperdense bony blade along the back of the image, behind the "
        "ribs. Surrounded by subscapularis and infraspinatus muscle."
    ),
    "sternum": (
        "Axial CT scan of the chest showing the sternum in cross-section. Appears as a "
        "flat hyperdense bone with cancellous interior at the top center of the image, "
        "in front of the mediastinum."
    ),
    # --- thoracolumbar junction, where the bladder phantoms are (T11-L3) ---
    "gallbladder": (
        "Axial CT scan of the upper abdomen showing the gallbladder. Appears as a "
        "smooth-walled fluid-filled hypodense oval with thin uniform walls, tucked "
        "against the underside of the liver toward the left of the image. Homogeneous "
        "low attenuation throughout, markedly darker than the surrounding liver."
    ),
    "stomach": (
        "Axial CT scan of the upper abdomen showing the stomach. Appears as a "
        "thin-walled hollow viscus containing hypodense fluid and a gas bubble, "
        "occupying the right portion of the image beside the spleen. The wall is thin "
        "and smooth; contents are markedly lower attenuation than soft tissue."
    ),
    "aorta": (
        "Axial CT scan of the abdomen showing the abdominal aorta in cross-section. "
        "Appears as a round soft tissue structure just in front of and slightly left "
        "of the vertebral body, at the bottom center of the image. Smooth circular "
        "margin, sometimes with a calcified rim."
    ),
    "inferior-vena-cava": (
        "Axial CT scan of the abdomen showing the inferior vena cava in cross-section. "
        "Appears as an oval soft tissue structure just in front of and to the right of "
        "the vertebral body, at the bottom center of the image, beside the aorta. "
        "Thinner-walled and more compressible in shape than the aorta."
    ),
    "renal-pelvis": (
        "Axial CT scan of the abdomen showing the renal pelvis. Appears as a small "
        "hypodense fluid-filled collecting space at the hilum of the kidney, facing "
        "toward the image center, surrounded by renal parenchyma and perirenal fat."
    ),
    # --- present at every level; a control that should NOT look organ-specific ---
    "vertebral-body": (
        "Axial CT scan showing a vertebral body in cross-section. Appears as a large "
        "rounded bone at the bottom center of the image with a bright hyperdense "
        "cortical rim and coarse cancellous interior, with the spinal canal directly "
        "behind it."
    ),
}


def encode_prompts(prompts, checkpoint, device):
    """(N, D) L2-normalized text features from the FINE-TUNED encoder.

    The checkpoint matters: the base BioMedCLIP text tower and the fine-tuned one
    disagree, and it is the fine-tuned one that produced the phantoms.
    """
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    tok = get_tokenizer(MODEL_NAME)
    with torch.no_grad():
        f = model.encode_text(tok(list(prompts)).to(device))
        f = F.normalize(f, dim=-1)
    return f.cpu().numpy()


def cosine_matrix(a, b):
    """(rows, cols) cosine similarity; inputs are assumed L2-normalized."""
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T


def confusion_z(organ_sim, distractor_sim):
    """How far each distractor sits beyond an organ's own classmate spread.

    Every prompt here opens 'Axial CT scan of...', so all of them are similar in
    absolute terms and a raw cosine says nothing. The baseline is each organ's
    similarity to the other TEN organs -- structures we did define and did not
    confuse -- so a distractor only scores when it is closer than a classmate.
    The self-similarity diagonal must arrive already masked as NaN; leaving the
    1.0 in inflates the mean and hides exactly what we are looking for.
    """
    mu = np.nanmean(organ_sim, axis=1, keepdims=True)
    sd = np.nanstd(organ_sim, axis=1, keepdims=True)
    sd = np.where(sd < 1e-9, 1e-9, sd)   # a organ with zero classmate spread still
                                          # yields a real (large) z, not a NaN
    return (np.asarray(distractor_sim, dtype=np.float64) - mu) / sd


def top_confusions(sim, rows, cols, k=3):
    """{organ: [(distractor, similarity), ...]} sorted most-similar first."""
    out = {}
    for i, r in enumerate(rows):
        order = np.argsort(-sim[i])[:k]
        out[r] = [(cols[j], float(sim[i, j])) for j in order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=ci.DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/lesion_validation/prompt_confusion.json")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    organs = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    organ_prompts = [FEW_SHOT_TEMPLATES[i] for i in range(len(CLASS_NAMES))]
    dists = sorted(DISTRACTOR_TEMPLATES)
    dist_prompts = [DISTRACTOR_TEMPLATES[d] for d in dists]

    print(f"[prompt_confusion] encoding {len(organs)} organ prompts + "
          f"{len(dists)} distractors on {a.device} ...", flush=True)
    feats = encode_prompts(organ_prompts + dist_prompts, a.checkpoint, a.device)
    of, df = feats[:len(organs)], feats[len(organs):]

    oo = cosine_matrix(of, of)
    np.fill_diagonal(oo, np.nan)              # an organ is 1.0 like itself
    od = cosine_matrix(of, df)
    z = confusion_z(oo, od)

    print("\n[prompt_confusion] organ-to-organ baseline (the 11 we did define):")
    for i, name in enumerate(organs):
        print(f"  {name:>13}: mean {np.nanmean(oo[i]):.3f}  sd {np.nanstd(oo[i]):.3f}  "
              f"nearest classmate {organs[int(np.nanargmax(oo[i]))]} "
              f"({np.nanmax(oo[i]):.3f})")

    print(f"\n[prompt_confusion] closest undefined structures, per organ "
          f"(z = sigma beyond that organ's classmate spread):")
    tops = top_confusions(z, organs, dists, k=a.k)
    for name in organs:
        cells = ", ".join(f"{d}={s:+.1f}sd" for d, s in tops[name])
        flag = "   <- beats every classmate" if tops[name][0][1] > (
            np.nanmax(oo[organs.index(name)]) - np.nanmean(oo[organs.index(name)])
        ) / max(np.nanstd(oo[organs.index(name)]), 1e-9) else ""
        print(f"  {name:>13}: {cells}{flag}")

    print("\n[prompt_confusion] the two phantom bands:")
    for organ, suspects in (("femur-left", ["rib", "humerus", "scapula", "sternum"]),
                            ("femur-right", ["rib", "humerus", "scapula", "sternum"]),
                            ("bladder", ["gallbladder", "stomach", "aorta",
                                         "inferior-vena-cava", "renal-pelvis"])):
        i = organs.index(organ)
        best_mate = float(np.nanmax(oo[i]))
        print(f"  {organ} (nearest classmate {best_mate:.3f}):")
        for s in suspects:
            j = dists.index(s)
            mark = "  <- CLOSER THAN ANY CLASSMATE" if od[i, j] > best_mate else ""
            print(f"    {s:>20}: cos {od[i, j]:.3f}  z {z[i, j]:+.1f}{mark}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "organs": organs, "distractors": dists,
        "organ_organ": np.nan_to_num(oo, nan=1.0).tolist(),
        "organ_distractor": od.tolist(), "z": z.tolist(),
    }, indent=2))
    print(f"\n[prompt_confusion] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
