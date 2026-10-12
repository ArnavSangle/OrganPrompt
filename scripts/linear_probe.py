"""Equal-supervision baseline for the BSPC revision (Reviewer 5, point 1).

Reviewer 5: OrganPrompt trains on ground-truth masks downsampled to the 14x14
patch grid, so it should be compared "against methods with equivalent
supervision". This is that comparison, at the cheapest honest tier: a LINEAR
PROBE on frozen BioMedCLIP patch tokens, trained on exactly the 14x14 targets
PatchContrastiveLoss uses (area-pool, > 0.5), on the same TotalSegmentator split.
No text encoder anywhere. Same features, same labels -- so the gap between probe
and OrganPrompt is what the text alignment + fine-tuning add.

Features match the inference engine (src/explainability/patch_similarity.py):
mean of ViT blocks 8-11, CLS dropped, 768-d per patch. The probe is one
nn.Linear(768, 11); an item's loss is BCE on its OWN class's column only,
because each TotalSegmentator item carries one organ's mask.

Thresholds. Both methods are scored with per-class IoU-optimal thresholds TUNED
ON THE VALIDATION SPLIT and applied to the test split. The submitted paper's
0.653 used thresholds swept on the test split itself (generate_masks.py +
evaluate_masks.py --tune both run on test), so `eval` also reports each method
under test-tuned ("oracle") thresholds -- the organprompt oracle row should
reproduce 0.653 and ties this run to the paper.

Configuration is fixed here in advance (pre-registered in
docs/preregistration/2026-09-29-revision-rerun.md): pretrained BioMedCLIP
backbone, AdamW lr 1e-3, wd 1e-4, batch 64, 5 epochs, best epoch by val loss.

    python scripts/linear_probe.py train --root data/TotalSeg/Totalsegmentator_dataset_v201
    python scripts/linear_probe.py eval --method probe --root ...
    python scripts/linear_probe.py eval --method organprompt --root ...
    python scripts/linear_probe.py select      # checkpoint re-selection on val
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels.templates import CLASS_NAMES

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
FUSED_BLOCKS = (8, 9, 10, 11)
N_CLASSES = len(CLASS_NAMES)
METRICS = ("iou", "dice", "hd95", "sens", "prec", "rve", "nsd")


# --- pure pieces (unit-tested) ------------------------------------------------

def patch_targets(masks):
    """(B,224,224) {0,1} -> (B,196): PatchContrastiveLoss's own downsampling."""
    t = F.interpolate(masks.unsqueeze(1).float(), size=(14, 14), mode="area")
    return (t.squeeze(1).reshape(masks.shape[0], -1) > 0.5).float()


def probe_loss(logits, labels, targets):
    """BCE on each item's labelled class only. logits (B,196,C), labels (B,)."""
    own = logits.gather(2, labels.view(-1, 1, 1).expand(-1, logits.shape[1], 1)).squeeze(2)
    return F.binary_cross_entropy_with_logits(own, targets)


def upsample_patch_map(grid):
    """(196,) -> (224,224) float32, bilinear exactly as the inference engine."""
    g = grid.reshape(1, 1, 14, 14).float()
    up = F.interpolate(g, size=(224, 224), mode="bilinear", align_corners=False)
    return up.squeeze().cpu().numpy().astype(np.float32)


def tune_thresholds(pairs_by_class, n_candidates=100):
    """{class: IoU-optimal threshold}: evaluate_masks._tune_thresholds, in memory.

    Candidates span each class's own map range; images with empty ground truth
    carry no IoU and are skipped, so a class with none gets no threshold.
    """
    tuned = {}
    for c, pairs in pairs_by_class.items():
        pairs = [(m.astype(np.float32), g) for m, g in pairs if g.any()]
        if not pairs:
            continue
        lo = min(float(m.min()) for m, _ in pairs)
        hi = max(float(m.max()) for m, _ in pairs)
        best_thr, best = lo, -1.0
        for thr in np.linspace(lo, hi, n_candidates):
            ious = []
            for m, g in pairs:
                p = m >= thr
                u = (p | g).sum()
                if u:
                    ious.append((p & g).sum() / u)
            if ious and np.mean(ious) > best:
                best, best_thr = float(np.mean(ious)), float(thr)
        tuned[c] = best_thr
    return tuned


def evaluate_pairs(pairs_by_class, thresholds, nsd_tolerance=3.0, metrics=METRICS):
    """Per-class and pooled means of every metric eval_checkpoints reports.

    The pooled mean is over all images of all classes, as eval_checkpoints.main
    computes the paper's headline numbers.
    """
    from scripts.eval_checkpoints import (_compute_iou, _compute_dice, _compute_hd95,
                                          _compute_sensitivity, _compute_precision,
                                          _compute_rve, _compute_nsd)
    fns = {"iou": _compute_iou, "dice": _compute_dice, "hd95": _compute_hd95,
           "sens": _compute_sensitivity, "prec": _compute_precision, "rve": _compute_rve,
           "nsd": lambda p, g: _compute_nsd(p, g, nsd_tolerance)}
    fns = {k: fns[k] for k in metrics}
    per = {k: defaultdict(list) for k in metrics}
    for c, pairs in pairs_by_class.items():
        thr = thresholds.get(c, 0.0)
        for m, g in pairs:
            p = m.astype(np.float32) >= thr
            for k, fn in fns.items():
                v = fn(p, g)
                if v is not None:
                    per[k][c].append(v)
    out = {"per_class": {}, "mean": {}}
    for c in sorted(pairs_by_class):
        out["per_class"][str(c)] = {
            "name": CLASS_NAMES[c], "n": len(pairs_by_class[c]),
            **{k: float(np.mean(per[k][c])) if per[k][c] else None for k in metrics}}
    for k in metrics:
        allv = [v for vals in per[k].values() for v in vals]
        out["mean"][k] = float(np.mean(allv)) if allv else None
    return out


def pick_checkpoint(val_miou):
    """Best validation mIoU; ties go to the earlier (lexically smaller) checkpoint."""
    return min(val_miou, key=lambda k: (-val_miou[k], k))


class FusedPatchFeatures:
    """(B,3,224,224) -> (B,196,768): mean of ViT blocks 8-11, CLS dropped.

    The same fusion BioMedCLIPPatchSimilarity uses, batched and without the
    text-space projection -- the probe learns its own map from token space.
    """

    def __init__(self, model, blocks=FUSED_BLOCKS):
        self.model = model
        self._feats = []
        self._hooks = [model.visual.trunk.blocks[i].register_forward_hook(
            lambda m, i, o: self._feats.append(o.detach())) for i in blocks]

    @torch.no_grad()
    def __call__(self, images):
        self._feats.clear()
        self.model.encode_image(images)
        fused = torch.stack(self._feats, 0).mean(0)
        self._feats.clear()
        return fused[:, 1:, :]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# --- model-backed stages (run by the user) ------------------------------------

def _backbone(device, init_checkpoint=None):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    if init_checkpoint:
        ck = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _skip(paths, a):
    """True (and says so) if every output exists and --force was not given --
    lets an unattended command chain pass over work another window finished."""
    if not a.force and all(Path(p).exists() for p in paths):
        print(f"[probe] SKIP {a.cmd}: {', '.join(str(p) for p in paths)} already exist "
              f"(--force to redo)", flush=True)
        return True
    return False


def train(a):
    if _skip([Path(a.out_dir) / "head_best.pt", Path(a.out_dir) / "train_history.json"], a):
        return
    from torch.utils.data import DataLoader
    from src.data.totalsegmentator import load_totalsegmentator
    torch.manual_seed(a.seed)
    train_ds, val_ds, _ = load_totalsegmentator(a.root, top_k=a.top_k, include_masks=True)
    print(f"[probe] train {len(train_ds)} | val {len(val_ds)} items", flush=True)
    feats = FusedPatchFeatures(_backbone(a.device, a.init_checkpoint))
    head = nn.Linear(768, N_CLASSES).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers,
              pin_memory=a.device == "cuda")
    tl = DataLoader(train_ds, shuffle=True, **kw)
    vl = DataLoader(val_ds, shuffle=False, **kw)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for ep in range(a.epochs):
        head.train()
        t0, tot, n = time.time(), 0.0, 0
        for img, lab, gt in tl:
            img, lab, gt = img.to(a.device), lab.to(a.device), gt.to(a.device)
            loss = probe_loss(head(feats(img)), lab, patch_targets(gt))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(lab), n + len(lab)
        head.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for img, lab, gt in vl:
                img, lab, gt = img.to(a.device), lab.to(a.device), gt.to(a.device)
                vtot += probe_loss(head(feats(img)), lab, patch_targets(gt)).item() * len(lab)
                vn += len(lab)
        rec = {"epoch": ep, "train_loss": tot / n, "val_loss": vtot / vn,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[probe] {rec}", flush=True)
        if rec["val_loss"] < best:
            best = rec["val_loss"]
            torch.save({"head": head.state_dict(), "epoch": ep, "config": vars(a)},
                       out / "head_best.pt")
    (out / "train_history.json").write_text(json.dumps(
        {"config": vars(a), "history": history}, indent=2))
    print(f"[probe] best val loss {best:.4f} -> {out / 'head_best.pt'}", flush=True)


def _organprompt_scorer(model, device):
    """score(img, lab) -> raw patch-text similarity map, as eval_checkpoints does.
    Text features are re-encoded here because the text tower is fine-tuned too;
    call again after swapping weights."""
    from open_clip import get_tokenizer
    from src.explainability.patch_similarity import BioMedCLIPPatchSimilarity
    from src.labels.templates import FEW_SHOT_TEMPLATES
    tok = get_tokenizer(MODEL_NAME)
    with torch.no_grad():
        tf = F.normalize(model.encode_text(tok(
            [FEW_SHOT_TEMPLATES[i] for i in range(N_CLASSES)]).to(device)), dim=-1)
    ps = BioMedCLIPPatchSimilarity(model, device=device)

    def score(img, lab):
        return ps.generate(img.unsqueeze(0).to(device), tf, target_class=lab)
    score.remove_hooks = ps.remove_hooks
    return score


def select(a):
    """Re-select the OrganPrompt checkpoint on VALIDATION mIoU (val-tuned
    thresholds), then score the chosen one on test. The submitted checkpoint
    (epoch 11) was the max-test-mIoU entry of eval_checkpoints.py."""
    if _skip([Path(a.out or "results/equal_supervision/checkpoint_selection.json")], a):
        return
    from src.data.totalsegmentator import load_totalsegmentator
    _, val_ds, test_ds = load_totalsegmentator(a.root, top_k=a.top_k, include_masks=True)
    ckpts = sorted(Path(a.checkpoint_dir).glob("epoch_*.pt"))
    model = _backbone(a.device)
    val_miou = {}
    for ck in ckpts:
        state = torch.load(str(ck), map_location=a.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()
        score = _organprompt_scorer(model, a.device)
        val = _collect(val_ds, score, a.limit)
        # IoU only: selection needs nothing else, and HD95/NSD cost ~0.36 s per
        # slice (~27 min per checkpoint); the full suite runs once, on test
        val_miou[ck.name] = evaluate_pairs(val, tune_thresholds(val),
                                           metrics=("iou",))["mean"]["iou"]
        score.remove_hooks()
        print(f"[select] {ck.name}: val mIoU {val_miou[ck.name]:.4f}", flush=True)
    best = pick_checkpoint(val_miou)
    print(f"[select] chosen on validation: {best}", flush=True)
    state = torch.load(str(Path(a.checkpoint_dir) / best), map_location=a.device,
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    score = _organprompt_scorer(model, a.device)
    val = _collect(val_ds, score, a.limit)
    test = _collect(test_ds, score, a.limit)
    res = {"checkpoint_dir": a.checkpoint_dir, "val_miou": val_miou, "chosen": best,
           "test_val_tuned": evaluate_pairs(test, tune_thresholds(val))}
    out = Path(a.out or "results/equal_supervision/checkpoint_selection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[select] {best} test mIoU (val-tuned) "
          f"{res['test_val_tuned']['mean']['iou']:.4f} -> {out}", flush=True)


def _collect(ds, score, limit):
    """{class: [(map float16, gt bool)]} for the first `limit` items (0 = all)."""
    n = len(ds) if not limit else min(limit, len(ds))
    pairs = defaultdict(list)
    for i in range(n):
        img, lab, gt = ds[i]
        pairs[int(lab)].append((score(img, int(lab)).astype(np.float16),
                                np.asarray(gt) > 0.5))
        if (i + 1) % 500 == 0:
            print(f"[probe]   {i + 1}/{n}", flush=True)
    return pairs


def evaluate(a):
    if _skip([Path(a.out or f"results/equal_supervision/{a.method}.json")], a):
        return
    from src.data.totalsegmentator import load_totalsegmentator
    _, val_ds, test_ds = load_totalsegmentator(a.root, top_k=a.top_k, include_masks=True)
    if a.method == "probe":
        ck = torch.load(Path(a.probe_dir) / "head_best.pt", map_location="cpu",
                        weights_only=False)
        feats = FusedPatchFeatures(_backbone(a.device, ck["config"].get("init_checkpoint")))
        head = nn.Linear(768, N_CLASSES).to(a.device).eval()
        head.load_state_dict(ck["head"])

        @torch.no_grad()
        def score(img, lab):
            logit = head(feats(img.unsqueeze(0).to(a.device)))[0, :, lab]
            return upsample_patch_map(torch.sigmoid(logit))
    else:
        score = _organprompt_scorer(_backbone(a.device, a.checkpoint), a.device)

    print(f"[probe] scoring val ({a.method})", flush=True)
    val = _collect(val_ds, score, a.limit)
    print(f"[probe] scoring test ({a.method})", flush=True)
    test = _collect(test_ds, score, a.limit)
    thr_val = tune_thresholds(val)
    thr_test = tune_thresholds(test)
    res = {"method": a.method, "limit": a.limit,
           "thresholds_val_tuned": {CLASS_NAMES[c]: t for c, t in thr_val.items()},
           "thresholds_test_tuned": {CLASS_NAMES[c]: t for c, t in thr_test.items()},
           "test_val_tuned": evaluate_pairs(test, thr_val),
           "test_oracle_test_tuned": evaluate_pairs(test, thr_test)}
    out = Path(a.out or f"results/equal_supervision/{a.method}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for key in ("test_val_tuned", "test_oracle_test_tuned"):
        m = res[key]["mean"]
        print(f"[probe] {a.method} {key}: mIoU {m['iou']:.4f} Dice {m['dice']:.4f} "
              f"HD95 {m['hd95']:.2f} NSD {m['nsd']:.4f}", flush=True)
    print(f"[probe] wrote {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "select"):
        s = sub.add_parser(name)
        s.add_argument("--root", default="data/TotalSeg/Totalsegmentator_dataset_v201")
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        s.add_argument("--top_k", type=int, default=5)
        s.add_argument("--force", action="store_true",
                       help="rerun even if this command's outputs already exist")
    t = sub.choices["train"]
    t.add_argument("--init_checkpoint", default=None,
                   help="default: pretrained BioMedCLIP (pre-registered primary)")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=1e-4)
    t.add_argument("--num_workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out_dir", default="checkpoints/linear_probe")
    e = sub.choices["eval"]
    e.add_argument("--method", choices=["probe", "organprompt"], required=True)
    e.add_argument("--probe_dir", default="checkpoints/linear_probe")
    e.add_argument("--checkpoint",
                   default="checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt")
    e.add_argument("--limit", type=int, default=0, help="items per split, 0 = all")
    e.add_argument("--out", default=None)
    sel = sub.choices["select"]
    sel.add_argument("--checkpoint_dir", default="checkpoints/totalseg_patch_v5")
    sel.add_argument("--limit", type=int, default=0, help="items per split, 0 = all")
    sel.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    {"train": train, "eval": evaluate, "select": select}[a.cmd](a)


if __name__ == "__main__":
    main()
