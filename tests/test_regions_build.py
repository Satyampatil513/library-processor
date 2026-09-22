import numpy as np

from libpipe.db import DB
from libpipe.pieces import Piece
from libpipe.regions import Region, build_inputs


def piece(id_, center, size=(0.03, 0.2, 0.15), overlaps=()):
    c, s = np.array(center, float), np.array(size, float)
    return Piece(id_, c - s / 2, c + s / 2, 100, [0], overlaps=list(overlaps))


class NullSession:
    """No stills/notes: exercises build_inputs' text-only path (no images depend on real frames)."""
    id = "s"
    stills = []
    notes = []
    frames = []

    def still(self, st):
        raise AssertionError


def test_near_hint_excludes_true_overlaps_and_far_pieces():
    a = piece("p1", (0, 1, 0))
    b = piece("p2", (0.05, 1, 0), overlaps=["p3"])       # near a, and a true overlap partner of p3
    c = piece("p3", (0.05, 1, 0), overlaps=["p2"])
    far = piece("p4", (5, 1, 0))
    region = Region("r0", [a, b, c, far])
    prompt, _ = build_inputs(DB(":memory:"), NullSession(), region, [])
    lines = prompt.splitlines()
    p1_block = "\n".join(lines[lines.index("- p1: (no crop available)"):lines.index("- p1: (no crop available)") + 3])
    assert "near (not overlapping): p2, p3" in p1_block
    p2_block = "\n".join(lines[lines.index("- p2: (no crop available)"):lines.index("- p2: (no crop available)") + 4])
    assert "overlaps physically with: p3" in p2_block
    assert "near (not overlapping): p1" in p2_block   # p3 excluded: already an overlap partner
    assert "p4" not in prompt.split("- p1:")[1].split("- p2:")[0]   # far piece not listed as near p1
