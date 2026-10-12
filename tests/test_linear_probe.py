"""Equal-supervision baseline: a linear probe on frozen BioMedCLIP patch tokens
trained on the same 14x14 mask-derived targets (BSPC revision, Reviewer 5 pt 1)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

import scripts.linear_probe as lp


def test_patch_targets_match_the_patch_loss_downsampling():
    """Same targets OrganPrompt trains on: area-pool to 14x14, then > 0.5."""
    m = torch.zeros(1, 224, 224)
    m[0, :16, :16] = 1.0                   # exactly patch (0,0)
    m[0, :16, 16:24] = 1.0                 # half of patch (0,1) -> not > 0.5
    t = lp.patch_targets(m)
    assert t.shape == (1, 196)
    assert t[0, 0] == 1.0 and t[0, 1] == 0.0 and t.sum() == 1.0


def test_probe_loss_only_scores_the_labelled_class():
    logits = torch.zeros(2, 196, 11)
    logits[0, :, 5] = 50.0                  # confidently wrong on a class not labelled
    targets = torch.zeros(2, 196)
    labels = torch.tensor([3, 3])
    loss = lp.probe_loss(logits, labels, targets)
    assert loss.item() == pytest.approx(np.log(2), abs=1e-5)   # all-zero logits on class 3


def test_upsample_matches_the_inference_engine():
    grid = torch.arange(196, dtype=torch.float32)
    up = lp.upsample_patch_map(grid)
    assert up.shape == (224, 224) and up.dtype == np.float32
    ref = torch.nn.functional.interpolate(grid.reshape(1, 1, 14, 14), size=(224, 224),
                                          mode="bilinear", align_corners=False)
    assert np.allclose(up, ref.squeeze().numpy())


def test_tune_thresholds_picks_the_iou_optimal_cut_per_class():
    gt = np.zeros((4, 4), bool)
    gt[:2] = True
    m = np.where(gt, 0.8, 0.2).astype(np.float32)
    thr = lp.tune_thresholds({0: [(m, gt)]}, n_candidates=11)
    assert 0.2 < thr[0] <= 0.8
    assert ((m >= thr[0]) == gt).all()


def test_tune_thresholds_ignores_pairs_with_empty_ground_truth():
    gt = np.zeros((2, 2), bool)
    thr = lp.tune_thresholds({0: [(np.ones((2, 2), np.float32), gt)]}, n_candidates=5)
    assert 0 not in thr


class _Block(nn.Module):
    def __init__(self, add):
        super().__init__()
        self.add = add

    def forward(self, x):
        return x + self.add


class _FakeCLIP(nn.Module):
    """Twelve blocks; block i adds i. Tokens start at 0, so block 8..11 outputs
    are 36, 45, 55, 66 (cumulative sums) and their mean is 50.5."""
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.trunk = nn.Module()
        self.visual.trunk.blocks = nn.ModuleList([_Block(float(i)) for i in range(12)])

    def encode_image(self, images):
        x = torch.zeros(images.shape[0], 197, 768)
        for b in self.visual.trunk.blocks:
            x = b(x)
        return x[:, 0]


def test_fused_features_average_blocks_8_to_11_and_drop_cls():
    feats = lp.FusedPatchFeatures(_FakeCLIP())
    out = feats(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 196, 768)
    assert torch.allclose(out, torch.full_like(out, 50.5))
    feats.remove_hooks()


def test_fused_features_do_not_accumulate_across_calls():
    feats = lp.FusedPatchFeatures(_FakeCLIP())
    feats(torch.zeros(1, 3, 224, 224))
    out = feats(torch.zeros(1, 3, 224, 224))
    assert torch.allclose(out, torch.full_like(out, 50.5))


def test_summarize_metrics_macro_averages_like_eval_checkpoints():
    gt = np.zeros((8, 8), bool)
    gt[:4] = True
    pairs = {0: [(np.where(gt, 1.0, 0.0).astype(np.float32), gt)],
             1: [(np.zeros((8, 8), np.float32), gt)]}
    res = lp.evaluate_pairs(pairs, {0: 0.5, 1: 0.5})
    assert res["per_class"]["0"]["iou"] == pytest.approx(1.0)
    assert res["per_class"]["1"]["iou"] == pytest.approx(0.0)
    assert res["mean"]["iou"] == pytest.approx(0.5)      # pooled over images, as eval_checkpoints


def test_pick_checkpoint_takes_the_best_validation_miou():
    scores = {"epoch_003.pt": 0.61, "epoch_011.pt": 0.64, "epoch_019.pt": 0.63}
    assert lp.pick_checkpoint(scores) == "epoch_011.pt"


def test_pick_checkpoint_breaks_ties_toward_the_earlier_epoch():
    assert lp.pick_checkpoint({"epoch_009.pt": 0.6, "epoch_004.pt": 0.6}) == "epoch_004.pt"


def test_train_skips_when_its_outputs_exist(tmp_path, capsys):
    (tmp_path / "head_best.pt").write_bytes(b"x")
    (tmp_path / "train_history.json").write_text("{}")
    lp.main(["train", "--out_dir", str(tmp_path), "--device", "cpu"])
    assert "SKIP" in capsys.readouterr().out


def test_eval_skips_when_its_output_exists(tmp_path, capsys):
    out = tmp_path / "probe.json"
    out.write_text("{}")
    lp.main(["eval", "--method", "probe", "--out", str(out), "--device", "cpu"])
    assert "SKIP" in capsys.readouterr().out
    assert out.read_text() == "{}"


def test_evaluate_pairs_can_compute_iou_only_with_identical_values():
    gt = np.zeros((8, 8), bool)
    gt[:4] = True
    m = np.where(gt, 0.9, 0.1).astype(np.float32)
    m[5, 5] = 0.95
    pairs = {0: [(m, gt)], 1: [(np.zeros((8, 8), np.float32), gt)]}
    full = lp.evaluate_pairs(pairs, {0: 0.5, 1: 0.5})
    fast = lp.evaluate_pairs(pairs, {0: 0.5, 1: 0.5}, metrics=("iou",))
    assert fast["mean"]["iou"] == full["mean"]["iou"]
    assert set(fast["mean"]) == {"iou"}
