import scripts.analyze_validation as av


def test_pair_splits_counts_anatomically_impossible_halves():
    """A scan holds both sides of a pair or neither; exactly one is a bug.

    This is the headline metric for bilateral grouping. Measured at 37% before
    the fix, it should be 0 after -- and it is the number to read FIRST, because
    mean rank barely moved between the two runs even though the assignment was
    visibly broken.
    """
    per_case = {
        "a": {"absent": ["kidney-left"]},                      # split
        "b": {"absent": ["kidney-left", "kidney-right"]},      # both gone, fine
        "c": {"absent": ["bladder"]},                          # unpaired, fine
        "d": {"absent": ["femur-right", "lung-left"]},         # two splits
    }
    r = av.pair_splits(per_case)
    assert r["split"] == 3
    assert r["both_absent"] == 1
    assert r["both_present"] == 8      # 4 cases x 3 pairs = 12 total
    assert r["n_pairs"] == 12


def test_pair_splits_is_zero_when_pairs_move_together():
    per_case = {"a": {"absent": ["lung-left", "lung-right", "pancreas"]},
                "b": {"absent": []}}
    r = av.pair_splits(per_case)
    assert r["split"] == 0
