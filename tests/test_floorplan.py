from libpipe.floorplan import merge_duplicate_walls


def wall(x0, z0, x1, z1, support=400):
    return {"start": [x0, z0], "end": [x1, z1], "length_m": ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5, "support": support}


def test_near_duplicate_parallel_walls_merge_into_one():
    """Regression test: a real wall's depth noise sometimes splits it into two near-parallel, near-coincident
    RANSAC fits instead of one - a user spotted this as a doubled line in the top-down plot."""
    a = wall(0.0, 0.0, 4.0, 0.0, support=500)
    b = wall(0.05, 0.02, 3.95, 0.08, support=300)   # same wall, ~5cm offset, slightly different fit
    out = merge_duplicate_walls([a, b])
    assert len(out) == 1
    assert out[0]["support"] == 800


def test_opposite_walls_of_a_room_stay_separate():
    """Two genuinely different parallel walls (e.g. the two long sides of a room) are metres apart and
    must NOT be merged into one."""
    a = wall(0.0, 0.0, 4.0, 0.0, support=500)
    b = wall(0.0, 3.0, 4.0, 3.0, support=500)   # 3m away: a different wall, not noise on the same one
    out = merge_duplicate_walls([a, b])
    assert len(out) == 2


def test_perpendicular_walls_stay_separate():
    a = wall(0.0, 0.0, 4.0, 0.0, support=500)
    b = wall(0.0, 0.0, 0.0, 3.0, support=500)   # a corner, not a duplicate
    out = merge_duplicate_walls([a, b])
    assert len(out) == 2


def test_merge_is_transitive_across_more_than_two_segments():
    """RANSAC can split one noisy wall into 3+ near-duplicate segments, not just 2; all must collapse."""
    a = wall(0.0, 0.00, 4.0, 0.00, support=300)
    b = wall(0.1, 0.02, 3.9, 0.03, support=300)
    c = wall(0.2, -0.02, 3.8, -0.01, support=300)
    out = merge_duplicate_walls([a, b, c])
    assert len(out) == 1
    assert out[0]["support"] == 900


def test_real_scan_left_wall_pair_merges():
    """Exact coordinates from the r3d_2026-09-08 bedroom scan's floorplan.json before this fix: two RANSAC
    fits for the same left wall, 15-17cm apart (real depth noise, not the ~5cm synthetic case above) -
    what the user actually saw as a doubled line in the top-down plot. cos(angle) between them is
    0.99998; perp offset ~15-17cm, over the original 0.12m tolerance and why it needed raising to 0.20m."""
    a = {"start": [-3.07, -1.84], "end": [-2.65, -5.23], "support": 8078}
    b = {"start": [-2.9, -1.83], "end": [-2.59, -4.46], "support": 3982}
    out = merge_duplicate_walls([a, b])
    assert len(out) == 1
    assert out[0]["support"] == 12060


def test_merged_wall_spans_the_full_extent_of_both_inputs():
    a = wall(0.0, 0.0, 2.0, 0.0, support=400)
    b = wall(1.5, 0.01, 4.0, 0.0, support=400)   # overlaps and extends past `a`
    out = merge_duplicate_walls([a, b])
    assert len(out) == 1
    xs = [out[0]["start"][0], out[0]["end"][0]]
    assert min(xs) <= 0.1 and max(xs) >= 3.9
