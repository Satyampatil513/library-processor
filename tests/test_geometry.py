import pytest
import numpy as np
from PIL import Image

from libpipe.pieces import Piece, box_iou, merge_and_flag, split_by_spine_edges
from libpipe.session import pixel_ray_point, project_world

I3 = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]


def frame(pos=(0, 0, 0)):
    T = np.eye(4)
    T[:3, 3] = pos
    return {"transform": T.tolist(), "intrinsics": I3}


def test_project_inverts_unproject_arkit_convention():
    fr = frame((1.0, 1.5, -2.0))
    for u, v, d in [(320, 240, 2.0), (100, 400, 1.3), (600, 50, 3.1)]:
        w = pixel_ray_point(fr, u, v, d, (640, 480))
        u2, v2, z2 = project_world(fr, w[None])[0]
        assert abs(u - u2) < 1e-6 and abs(v - v2) < 1e-6 and abs(z2 - d) < 1e-9


def test_camera_looks_down_minus_z_and_y_is_up():
    fr = frame()
    ahead = pixel_ray_point(fr, 320, 240, 2.0, (640, 480))
    assert np.allclose(ahead, [0, 0, -2.0])
    upper = pixel_ray_point(fr, 320, 100, 2.0, (640, 480))     # smaller v = higher in image
    assert upper[1] > 0


def test_spine_edges_split_touching_books():
    rng = np.random.default_rng(0)
    gray = np.full((120, 200), 120, np.uint8)
    for x0, x1, val in [(10, 60, 40), (60, 110, 200), (110, 190, 90)]:      # three differently-coloured spines
        gray[10:110, x0:x1] = val
    gray = np.clip(gray + rng.integers(-3, 3, gray.shape), 0, 255).astype(np.uint8)
    mask = np.zeros(gray.shape, bool)
    mask[10:110, 10:190] = True
    parts = split_by_spine_edges(mask, gray)
    assert len(parts) == 3
    assert sum(p.sum() for p in parts) == mask.sum()       # no pixels lost or duplicated


def test_uniform_mask_is_not_split():
    gray = np.full((100, 100), 100, np.uint8)
    mask = np.zeros_like(gray, bool)
    mask[10:90, 10:90] = True
    assert len(split_by_spine_edges(mask, gray)) == 1


def box(c, s=(0.03, 0.2, 0.15), frames=(0,)):
    c, s = np.array(c, float), np.array(s, float)
    return Piece("", c - s / 2, c + s / 2, 100, list(frames))


def test_same_spot_pieces_merge_across_frames():
    raw = [box((0, 1, 0), frames=(0,)), box((0.002, 1, 0), frames=(5,)), box((0.5, 1, 0), frames=(0,))]
    out = merge_and_flag(raw)
    assert len(out) == 2
    two = next(p for p in out if len(p.frames) == 2)
    assert two.frames == [0, 5]


def test_partial_intersection_is_flagged_not_merged():
    a, b = box((0, 1, 0), s=(0.2, 0.2, 0.2)), box((0.17, 1, 0), s=(0.2, 0.2, 0.2))
    assert 0.02 < box_iou(a, b) < 0.3
    out = merge_and_flag([a, b])
    assert len(out) == 2 and out[0].overlaps == [out[1].id] and out[1].overlaps == [out[0].id]


def test_adjacent_books_do_not_overlap():
    out = merge_and_flag([box((0, 1, 0)), box((0.031, 1, 0))])
    assert len(out) == 2 and not out[0].overlaps


def test_best_group_still_rejects_edge_slivers_over_full_view():
    from libpipe.regions import best_group_still

    p = box((0, 1, -2))    # 3cm x 20cm x 15cm box, 2m in front of the camera
    centered = frame((0, 1, 0))
    centered.update(index=0, width=640, height=480)
    # a still where the box barely clips the corner (tiny visible fraction) must lose to one framing it fully
    edge = {"transform": np.eye(4).tolist(), "intrinsics": [[3000, 0, -600], [0, 3000, 240], [0, 0, 1]],
            "width": 640, "height": 480}
    stills = [edge, centered]

    class FakeSession:
        def __init__(self, stills):
            self.stills = stills

    best = best_group_still(FakeSession(stills), [p])
    assert best is centered


def test_best_group_still_none_when_nothing_visible():
    from libpipe.regions import best_group_still

    p = box((0, 1, 5))   # behind every camera below (they look down -z, so +z is behind them)
    fr = frame((0, 1, 0))
    fr["width"], fr["height"] = 640, 480

    class FakeSession:
        stills = [fr]

    assert best_group_still(FakeSession(), [p]) is None


def test_lift_mask_accepts_medium_confidence_not_just_high(tmp_path):
    """Regression test for the depth-confidence bug: requiring confidence==2 ('high' only) silently dropped
    small/distant/dark real objects whose depth pixels were mostly 'medium' (1). confidence>=1 must be enough."""
    from libpipe.pieces import lift_mask
    from libpipe.session import Session

    dh, dw, rw, rh = 20, 20, 20, 20
    conf = np.ones((dh, dw), np.uint8)          # every pixel is "medium" confidence, none "high"
    depth = np.full((dh, dw), 1.5, np.float32)
    (tmp_path / "frames").mkdir()
    depth.tofile(tmp_path / "frames" / "000000_depth.bin")
    conf.tofile(tmp_path / "frames" / "000000_conf.bin")
    manifest = {"depth_width": dw, "depth_height": dh, "rgb_width": rw, "rgb_height": rh,
               "session_id": "t", "country": "US", "frames": [], "stills": [], "notes": []}
    s = Session(tmp_path, manifest)
    fr = {"index": 0, "intrinsics": [[20, 0, 10], [0, 20, 10], [0, 0, 1]], "transform": np.eye(4).tolist(),
          "depth": "frames/000000_depth.bin", "confidence": "frames/000000_conf.bin"}
    mask = np.zeros((rh, rw), bool)
    mask[5:15, 5:15] = True   # 100px mask, well over min_pts once accepted

    assert lift_mask(s, fr, mask, min_conf=2) is None       # old behaviour: no confidence==2 pixels -> nothing
    pts = lift_mask(s, fr, mask, min_conf=1)                # fixed default: medium confidence is accepted
    assert pts is not None and len(pts) >= 15


def test_replicate_sam2_retries_transient_network_errors(monkeypatch):
    """Regression test: a bare DNS/connection hiccup used to kill a whole multi-frame run outright."""
    import httpx as _httpx
    from libpipe.pieces import ReplicateSAM2

    seg = ReplicateSAM2.__new__(ReplicateSAM2)
    seg.http = object.__new__(_httpx.Client)  # never actually used; .request is monkeypatched below
    calls = {"n": 0}

    def flaky_request(method, url, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _httpx.ConnectError("getaddrinfo failed")
        return _httpx.Response(200, json={"ok": True}, request=_httpx.Request(method, url))

    monkeypatch.setattr(seg.http, "request", flaky_request, raising=False)
    monkeypatch.setattr("time.sleep", lambda s: None)   # don't actually wait in the test
    r = seg._request_with_backoff("GET", "https://api.replicate.com/v1/models/meta/sam-2")
    assert r.json() == {"ok": True} and calls["n"] == 3


def test_replicate_sam2_gives_up_after_max_attempts(monkeypatch):
    import httpx as _httpx
    from libpipe.pieces import ReplicateSAM2

    seg = ReplicateSAM2.__new__(ReplicateSAM2)
    seg.http = object.__new__(_httpx.Client)

    def always_fails(method, url, **kw):
        raise _httpx.ConnectError("getaddrinfo failed")

    monkeypatch.setattr(seg.http, "request", always_fails, raising=False)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(_httpx.ConnectError):
        seg._request_with_backoff("GET", "https://api.replicate.com/v1/models/meta/sam-2", attempts=3)


def _fake_yolo_result(obb_corners: np.ndarray | None):
    """obb_corners: None (no detections) or an (N, 4, 2) array of OBB polygon corners in pixel coords."""

    class FakeTensor:
        def __init__(self, a):
            self._a = a

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    class FakeOBB:
        def __init__(self, a):
            self.xyxyxyxy = FakeTensor(a)

        def __len__(self):
            return len(self._array())

        def _array(self):
            return self.xyxyxyxy._a

    class FakeResult:
        obb = FakeOBB(obb_corners) if obb_corners is not None else None

    return FakeResult()


def test_spine_detector_is_presplit_and_skips_further_splitting():
    from libpipe.pieces import SpineDetector

    assert SpineDetector.presplit is True


def test_spine_detector_rasterizes_obb_corners_to_a_boolean_mask():
    from libpipe.pieces import SpineDetector

    seg = SpineDetector.__new__(SpineDetector)
    seg.conf = 0.6
    corners = np.array([[[10, 10], [50, 10], [50, 40], [10, 40]]], dtype=float)   # one axis-aligned box

    class FakeModel:
        def predict(self, img, conf, verbose):
            assert conf == 0.6
            return [_fake_yolo_result(corners)]

    seg.model = FakeModel()
    img = Image.new("RGB", (100, 80))
    masks = seg.segment(img)

    assert len(masks) == 1
    m = masks[0]
    assert m.shape == (80, 100) and m.dtype == bool
    assert m[20, 30] and not m[5, 5]                          # inside vs. clearly outside the box
    assert m.sum() == pytest.approx((50 - 10) * (40 - 10), rel=0.1)   # fillPoly includes boundary pixels


def test_spine_detector_returns_no_masks_below_confidence():
    from libpipe.pieces import SpineDetector

    seg = SpineDetector.__new__(SpineDetector)
    seg.conf = 0.6

    class FakeModel:
        def predict(self, img, conf, verbose):
            return [_fake_yolo_result(None)]      # ultralytics leaves .obb=None when nothing clears `conf`

    seg.model = FakeModel()
    assert seg.segment(Image.new("RGB", (50, 50))) == []


def test_find_pieces_splits_only_segmenters_without_presplit(monkeypatch):
    """A list of segmenters (diagram step 6: SAM2 + spine detector) must run split_by_spine_edges on a
    generic segmenter's masks but use a `presplit=True` segmenter's masks (SpineDetector) as-is."""
    import libpipe.pieces as pieces_mod
    from libpipe.pieces import find_pieces

    rng = np.random.default_rng(0)
    W, H = 800, 600
    gray = np.full((H, W), 120, np.uint8)
    for x0, x1, val in [(10, 60, 40), (60, 110, 200), (110, 190, 90)]:   # three differently-coloured spines
        gray[10:110, x0:x1] = val
    gray = np.clip(gray.astype(int) + rng.integers(-3, 3, gray.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([gray] * 3, axis=-1))

    same_region_mask = np.zeros((H, W), bool)
    same_region_mask[10:110, 10:190] = True   # spans all three spines: splittable iff not presplit

    class FakeSAM2:
        presplit = False

        def segment(self, im):
            return [same_region_mask.copy()]

    class FakeSpineDetector:
        presplit = True

        def segment(self, im):
            return [same_region_mask.copy()]

    class FakeSession:
        frames = [{"index": 0}]

        def rgb(self, fr):
            return img

    calls = {"n": 0}

    def fake_lift_mask(s, fr, sub, *a, **kw):
        calls["n"] += 1
        c = calls["n"] * 0.5
        return np.array([[c, c, c], [c + 0.05, c, c], [c, c + 0.05, c], [c, c, c + 0.05]] * 5, dtype=float)

    monkeypatch.setattr(pieces_mod, "lift_mask", fake_lift_mask)

    kept_per_frame = []
    find_pieces(FakeSession(), [FakeSAM2(), FakeSpineDetector()], every=1,
                on_frame=lambda fr, im, masks: kept_per_frame.append(len(masks)))

    assert kept_per_frame == [4]   # 3 split parts from the non-presplit segmenter + 1 unsplit from SpineDetector


def test_find_pieces_keeps_furniture_scale_objects(monkeypatch):
    """Regression test: an earlier book-only default (0.25 frame area / 0.6m max extent) silently discarded
    real furniture candidates SAM2 had already found - a user noticed SAM2 detecting "many things" but
    furniture never showing up as a piece, and library-insurance scope needs furniture (wardrobes, sofas,
    bookshelves), not just books. Confirms both the new default and that a wall/room-scale mask is still
    rejected by max_dim_m."""
    import libpipe.pieces as pieces_mod
    from libpipe.pieces import find_pieces

    W, H = 800, 600
    img = Image.new("RGB", (W, H), (120, 120, 120))
    furniture_mask = np.zeros((H, W), bool)
    furniture_mask[100:500, 100:600] = True   # ~42% of the frame: over the old 0.25 cap, under the new 0.6 one

    class FakeFurnitureSegmenter:
        presplit = True   # not spine-shaped; skip the vertical-edge splitter either way

        def segment(self, im):
            return [furniture_mask.copy()]

    class FakeSession:
        frames = [{"index": 0}]

        def rgb(self, fr):
            return img

    def lift_furniture(s, fr, sub, *a, **kw):
        # a wide, tall real object: ~1.2m x 2.0m x 0.5m - furniture-scale, not book-scale, not wall-scale
        return np.array([[0, 0, 0], [1.2, 0, 0], [0, 2.0, 0], [0, 0, 0.5]] * 5, dtype=float)

    def lift_wall(s, fr, sub, *a, **kw):
        return np.array([[0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 2.5]] * 5, dtype=float)

    monkeypatch.setattr(pieces_mod, "lift_mask", lift_furniture)
    kept = []
    find_pieces(FakeSession(), [FakeFurnitureSegmenter()], every=1, on_frame=lambda fr, im, m: kept.append(len(m)))
    assert kept == [1]   # old 0.25-frame-area / 0.6m defaults would have dropped this

    monkeypatch.setattr(pieces_mod, "lift_mask", lift_wall)
    kept.clear()
    find_pieces(FakeSession(), [FakeFurnitureSegmenter()], every=1, on_frame=lambda fr, im, m: kept.append(len(m)))
    assert kept == [0]   # room-scale extent must still be rejected by max_dim_m


def test_pending_regions_skips_already_priced_ones():
    """Regression test: a killed mid-run process must be resumable without re-paying for Fable/Astra/Jev
    calls on regions the DB already has stored objects for (found when a real final run got killed by a
    power outage partway through, at region 25 of 46, with 23 already safely persisted)."""
    from libpipe.pipeline import _pending_regions
    from libpipe.regions import Region

    regions = [Region(f"r{i:03d}", []) for i in range(5)]
    done = {"r000", "r001", "r002"}
    pending = _pending_regions(regions, done)
    assert [r.id for r in pending] == ["r003", "r004"]


def test_rotate_cw_exact_pixel_mapping():
    """Regression test: RoomCapture (this project's own app - see CaptureSession.swift) saves frames/stills
    in raw sensor orientation regardless of how the phone was physically held, confirmed by a real upload
    where a still's content was visibly sideways (a window lying on its side). `rotate_cw` corrects crops
    shown to vision models / saved for review without touching any depth/mask math. Verified empirically:
    a 90 correction (PIL ROTATE_270 under the hood) made the real photo upright."""
    from libpipe.pieces import rotate_cw

    img = Image.new("RGB", (4, 2))   # 4 wide, 2 tall; mark the top-left pixel to track orientation
    img.putpixel((0, 0), (255, 0, 0))

    assert rotate_cw(img, 0) is img   # no-op returns the same object, no needless re-encode

    r90 = rotate_cw(img, 90)
    assert r90.size == (2, 4)                          # dimensions swap on a 90/270 rotation
    assert r90.getpixel((1, 0)) == (255, 0, 0)          # top-left corner moves to top-right after a CW turn

    r180 = rotate_cw(img, 180)
    assert r180.size == (4, 2)
    assert r180.getpixel((3, 1)) == (255, 0, 0)         # top-left moves to bottom-right

    r270 = rotate_cw(img, 270)
    assert r270.size == (2, 4)
    assert r270.getpixel((0, 3)) == (255, 0, 0)         # top-left moves to bottom-left after a CCW-equivalent turn


def test_crop_for_piece_applies_rotation(monkeypatch):
    from libpipe.pieces import Piece, crop_for_piece

    I3 = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]
    T = np.eye(4).tolist()
    still_data = {"transform": T, "intrinsics": I3, "width": 640, "height": 480}
    img = Image.new("RGB", (640, 480), (10, 20, 30))

    class FakeSession:
        stills = [still_data]

        def still(self, st):
            return img

    p = Piece("p0", np.array([-0.3, -0.3, -1.15]), np.array([0.3, 0.3, -0.85]), 100)

    plain = crop_for_piece(FakeSession(), p, max_crops=1)
    rotated = crop_for_piece(FakeSession(), p, max_crops=1, rotate_deg=90)
    assert rotated[0].size == (plain[0].height, plain[0].width)


def test_load_cached_geometry_reconstructs_pieces_and_regions(tmp_path):
    """Regression test: --resume previously only skipped already-priced *regions*, but still re-ran the
    paid SAM2/spine-detector segmentation (step 6) from scratch on every attempt, since nothing cached its
    output - found when a real session needed several resumes in a row and kept re-paying Replicate for
    identical geometry each time."""
    from libpipe.artifacts import Recorder
    from libpipe.pipeline import _load_cached_geometry

    rec = Recorder(tmp_path / "out")
    rec.json("pieces/pieces.json", [
        {"id": "p0000", "min": [0, 0, 0], "max": [0.1, 0.2, 0.1], "size_cm": [20, 10, 10],
         "frames": [0, 1], "barcode": "9780000000000", "overlaps": ["p0001"]},
        {"id": "p0001", "min": [0.1, 0, 0], "max": [0.2, 0.2, 0.1], "size_cm": [10, 20, 10],
         "frames": [2], "barcode": None, "overlaps": ["p0000"]},
    ])
    rec.json("regions/regions.json", {"r000": ["p0000", "p0001"]})

    cached = _load_cached_geometry(rec)
    assert cached is not None
    pieces, regions = cached
    by_id = {p.id: p for p in pieces}
    assert set(by_id) == {"p0000", "p0001"}
    assert by_id["p0000"].barcode == "9780000000000" and by_id["p0000"].overlaps == ["p0001"]
    assert np.allclose(by_id["p0000"].box_max, [0.1, 0.2, 0.1])
    assert len(regions) == 1 and regions[0].id == "r000"
    assert [p.id for p in regions[0].pieces] == ["p0000", "p0001"]


def test_load_cached_geometry_returns_none_when_nothing_cached_yet(tmp_path):
    from libpipe.artifacts import Recorder
    from libpipe.pipeline import _load_cached_geometry

    assert _load_cached_geometry(Recorder(tmp_path / "out")) is None


def test_pending_regions_is_a_noop_when_nothing_is_done():
    from libpipe.pipeline import _pending_regions
    from libpipe.regions import Region

    regions = [Region(f"r{i:03d}", []) for i in range(3)]
    assert _pending_regions(regions, set()) == regions
