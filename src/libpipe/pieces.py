"""Step 6 FIND PIECES, no AI [F6]: SAM2 + spine-boundary detector on frames, lift masks to 3D via depth+pose,
merge same-spot pieces across frames, flag overlaps, decode visible barcodes from stills.

SOURCE DIAGRAM TARGET, not measured by this code: spine separation ~75-85% (UNVALIDATED in the diagram itself:
pilot on 15-20 real shelves). Never run on a real bookshelf. On two non-shelf test scans the free OpenCV
fallback badly over-segmented (1253 "pieces" in one room); real SAM2 did much better (129) but neither has been
scored against ground truth, and split_by_spine_edges is an untrained heuristic, not the paper-grade oriented
spine detector this number implies.
Not implemented: "unclaimed space" flagging (needs shelf-plane estimate) - listed in README gaps.
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np
from PIL import Image

from .session import Session, project_world


class Segmenter(Protocol):
    def segment(self, img: Image.Image) -> list[np.ndarray]:
        """Return boolean masks (H x W) for every candidate object in the image."""


class ReplicateSAM2:
    """Hosted SAM2 automatic mask generation. Slug/inputs follow Replicate's `meta/sam-2` model card;
    verify against the live schema on first run."""

    def __init__(self, model: str = "meta/sam-2"):
        self.http = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {os.environ['REPLICATE_API_TOKEN']}",
                                                       "Prefer": "wait"})
        self.model = model

    def segment(self, img: Image.Image) -> list[np.ndarray]:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        if not hasattr(self, "version"):     # community model: needs the version-id endpoint
            m = self._request_with_backoff("GET", f"https://api.replicate.com/v1/models/{self.model}")
            self.version = m.json()["latest_version"]["id"]
        r = self._request_with_backoff("POST", "https://api.replicate.com/v1/predictions",
                                       json={"version": self.version, "input": {"image": uri, "use_m2m": True, "points_per_side": 32}})
        pred = r.json()
        while pred["status"] in ("starting", "processing"):
            time.sleep(1.5)
            pred = self._request_with_backoff("GET", pred["urls"]["get"]).json()
        if pred["status"] != "succeeded":
            raise RuntimeError(f"SAM2 failed: {pred.get('error')}")
        masks = []
        for url in pred["output"]["individual_masks"]:
            content = self._request_with_backoff("GET", url).content
            m = np.array(Image.open(io.BytesIO(content)).convert("L").resize(img.size))
            masks.append(m > 127)
        return masks

    def _request_with_backoff(self, method: str, url: str, attempts: int = 6, **kw) -> httpx.Response:
        """Retries both HTTP 429 (rate limit) and transient network failures (DNS hiccup, connection reset,
        timeout) with exponential backoff. A multi-frame session can run for 20+ minutes across dozens of
        calls; a single transient DNS lookup failure used to kill the whole run - found by hitting exactly
        that in a real run partway through frame 15/37."""
        last: Exception | None = None
        for i in range(attempts):
            try:
                r = self.http.request(method, url, **kw)
            except httpx.TransportError as e:
                last = e
                time.sleep(2 ** i)
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 2 ** i)))
                continue
            r.raise_for_status()
            return r
        raise last or RuntimeError(f"SAM2 request to {url} failed after {attempts} attempts")


class OpenCVSegmenter:
    """Free local fallback (no API, no GPU): mean-shift smoothing, Canny edges, then connected regions between
    edges. Far cruder than SAM2 (splits/merges objects on texture and lighting), so use it only to exercise the
    pipeline until SAM2 is available."""

    def __init__(self, work_width: int = 480, canny: tuple[int, int] = (30, 90), min_px: int = 150):
        self.work_width, self.canny, self.min_px = work_width, canny, min_px

    def segment(self, img: Image.Image) -> list[np.ndarray]:
        import cv2

        a = np.array(img)
        scale = self.work_width / a.shape[1]
        small = cv2.resize(a, (self.work_width, int(a.shape[0] * scale)))
        sm = cv2.pyrMeanShiftFiltering(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), 8, 18)
        edges = cv2.Canny(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), *self.canny)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        n, lbl = cv2.connectedComponents((edges == 0).astype(np.uint8), connectivity=4)
        masks = []
        for i in range(1, n):
            m = lbl == i
            if m.sum() >= self.min_px:
                masks.append(cv2.resize(m.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0)
        return masks


class SpineDetector:
    """Trained YOLOv8-OBB book-spine detector (see results/README.md "Spine detector fine-tune" for the
    training run, dataset, and honest metrics - mAP50-95 76.3% on Roboflow's held-out split, NEVER run on a
    real bookshelf or any photo from this project's own capture pipeline before now).

    Returns one mask per detected spine, already correctly separated by the model - unlike SAM2's generic
    masks these are not passed through split_by_spine_edges (see find_pieces): a tight single-spine oriented
    box has no reason to need further splitting, and doing so risked cutting a real detection in half.

    Default confidence (0.6) is chosen from an empirical false-positive check, not from any recall data (none
    exists - no real shelf to test recall against): across 32 real frames from two book-free room scans,
    false "book-spine" detections dropped from 45 (conf>=0.25) to 9 (conf>=0.5) to 0 (conf>=0.8). 0.6 is a
    conservative middle ground pending real recall data to actually tune against."""

    presplit = True   # find_pieces must not run split_by_spine_edges on an already-precise detection

    def __init__(self, weights: str = "runs/obb/runs_spine/nano48/weights/best.pt", conf: float = 0.6):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf

    def segment(self, img: Image.Image) -> list[np.ndarray]:
        import cv2

        r = self.model.predict(img, conf=self.conf, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            return []
        w, h = img.size
        masks = []
        for corners in r.obb.xyxyxyxy.cpu().numpy():
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [corners.astype(np.int32)], 1)
            masks.append(m.astype(bool))
        return masks


def split_by_spine_edges(mask: np.ndarray, gray: np.ndarray, min_w: int = 14, k: float = 1.8) -> list[np.ndarray]:
    """Spine-boundary detector: vertical-edge energy across a mask's bbox; split at strong, well-spaced peaks.
    Assumes upright books (vertical spine boundaries); horizontally stacked books are left as one piece."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if x1 - x0 < 2 * min_w:
        return [mask]
    g = np.abs(np.diff(gray[y0:y1, x0:x1].astype(np.float32), axis=1))
    g = np.where(mask[y0:y1, x0 + 1:x1], g, 0)
    prof = np.convolve(g.sum(axis=0) / max(1, (y1 - y0)), np.ones(3) / 3, mode="same")
    thr = prof.mean() + k * prof.std()
    cuts, last = [], -min_w
    for i in range(min_w, len(prof) - min_w):
        if prof[i] > thr and prof[i] == prof[max(0, i - min_w):i + min_w].max() and i - last >= min_w:
            cuts.append(i)
            last = i
    if not cuts:
        return [mask]
    edges = [0, *[c + 1 for c in cuts], x1 - x0]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = np.zeros_like(mask)
        m[y0:y1, x0 + a:x0 + b] = mask[y0:y1, x0 + a:x0 + b]
        if m.sum() > 200:
            out.append(m)
    return out or [mask]


@dataclass
class Piece:
    id: str
    box_min: np.ndarray
    box_max: np.ndarray
    n_pts: int
    frames: list[int] = field(default_factory=list)
    barcode: str | None = None
    overlaps: list[str] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return (self.box_min + self.box_max) / 2

    @property
    def size(self) -> np.ndarray:
        return self.box_max - self.box_min

    def corners(self) -> np.ndarray:
        lo, hi = self.box_min, self.box_max
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def box_iou(a: Piece, b: Piece) -> float:
    lo, hi = np.maximum(a.box_min, b.box_min), np.minimum(a.box_max, b.box_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    union = np.prod(a.size) + np.prod(b.size) - inter
    return float(inter / union) if union > 0 else 0.0


def lift_mask(s: Session, frame: dict, mask: np.ndarray, min_conf: int = 1, min_pts: int = 15) -> np.ndarray | None:
    """Mask (RGB resolution) -> world points of its confident depth pixels.

    BUG FIXED: this used to require ARKit confidence ==2 ("high" only, discarding "medium") and >=30 points.
    On a real cluttered-room scan, checking every correctly-segmented SAM2 mask individually showed 6 of 10
    otherwise-valid object masks in one frame were silently dropped here - not a segmentation failure, a depth
    threshold failure. Relaxing to confidence>=1 ("medium" allowed) recovered 5 of those 6 immediately; the
    min-point floor was also lowered since it's evaluated on the much lower-resolution depth grid, so a
    legitimately small or distant object can have very few pixels there even when its mask is fine at RGB
    resolution. This disproportionately hit small, distant, or dark objects (a teddy bear, a spray can, small
    bottles) - exactly the kind of clutter that fills a typical room, and, more importantly for this project,
    the kind of thin object a book spine is."""
    d, conf = s.depth(frame), s.confidence(frame)
    dh, dw = d.shape
    mh, mw = mask.shape
    m = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((dw, dh), Image.NEAREST)) > 0
    ok = m & (conf >= min_conf) & (d > 0.1) & (d < 6) & np.isfinite(d)
    if ok.sum() < min_pts:
        return None
    K = np.array(frame["intrinsics"], dtype=float).reshape(3, 3)
    sx, sy = dw / mw, dh / mh
    v, u = np.where(ok)
    z = d[ok].astype(float)
    cam = np.stack([(u - K[0, 2] * sx) / (K[0, 0] * sx) * z, -(v - K[1, 2] * sy) / (K[1, 1] * sy) * z, -z,
                    np.ones_like(z)], axis=1)
    T = np.array(frame["transform"], dtype=float).reshape(4, 4)
    return (T @ cam.T).T[:, :3]


def _union_find(n: int, edges: list[tuple[int, int]]) -> list[int]:
    p = list(range(n))

    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for a, b in edges:
        p[f(a)] = f(b)
    return [f(i) for i in range(n)]


def find_pieces(s: Session, seg: Segmenter | list[Segmenter], every: int = 5, merge_iou: float = 0.3,
                on_frame=None, max_frame_frac: float = 0.6, max_dim_m: float = 3.0) -> list[Piece]:
    """`seg` may be a single segmenter or several (e.g. `[ReplicateSAM2(), SpineDetector()]`, matching the
    diagram's step 6 "SAM2 + spine-boundary detector"). A segmenter with `presplit = True` (SpineDetector) has
    its masks used as-is - already one precise detection each - instead of being run through
    split_by_spine_edges, which is for SAM2's generic, possibly-multi-spine masks.

    `max_frame_frac`/`max_dim_m` bound what counts as "one object" in 2D image area and 3D extent. This
    project's scope is now library-insurance inventory, not books alone (furniture, laptops, chargers,
    anything seen) - so these must admit large furniture (a wardrobe, a bookshelf, a sofa), not just
    book-scale objects. Raised from an earlier book-only default (0.25 frame / 0.6m) that was silently
    discarding furniture candidates SAM2 had already found - found by a user noticing SAM2 detected "many
    things" but furniture never showed up as a piece. Still bounded (not unlimited) so a wall/floor/ceiling
    mask - which spans meters in every direction - doesn't get lifted as "one object"; step 8's `is_object`
    judgment is the real backstop for that (see `_store_object`'s `not_an_object` path), not this heuristic."""
    segmenters = seg if isinstance(seg, list) else [seg]
    raw: list[Piece] = []
    for fr in s.frames[::every]:
        img = s.rgb(fr)
        gray = np.array(img.convert("L"))
        area = img.width * img.height
        kept_masks: list[np.ndarray] = []
        for segmenter in segmenters:
            presplit = getattr(segmenter, "presplit", False)
            for m in segmenter.segment(img):
                if not (0.002 * area < m.sum() < max_frame_frac * area):
                    continue  # too small = noise, too big = wall/floor/ceiling
                for sub in ([m] if presplit else split_by_spine_edges(m, gray)):
                    pts = lift_mask(s, fr, sub)
                    if pts is None:
                        continue
                    lo, hi = np.percentile(pts, 3, axis=0), np.percentile(pts, 97, axis=0)
                    if np.any(hi - lo > max_dim_m) or np.prod(hi - lo + 1e-3) < 1e-6:
                        continue  # implausible extent for one object (wall/room-scale, not furniture-scale)
                    raw.append(Piece("", lo, hi, len(pts), [fr["index"]]))
                    kept_masks.append(sub)
        if on_frame:
            on_frame(fr, img, kept_masks)
    return merge_and_flag(raw, merge_iou)


def merge_and_flag(raw: list[Piece], merge_iou: float = 0.3) -> list[Piece]:
    if not raw:
        return []
    edges = [(i, j) for i in range(len(raw)) for j in range(i + 1, len(raw))
             if box_iou(raw[i], raw[j]) >= merge_iou
             or np.linalg.norm(raw[i].center - raw[j].center) < 0.03]
    roots = _union_find(len(raw), edges)
    groups: dict[int, list[Piece]] = {}
    for r, p in zip(roots, raw):
        groups.setdefault(r, []).append(p)
    merged = []
    for g in groups.values():
        w = np.array([p.n_pts for p in g], dtype=float)
        lo = np.average([p.box_min for p in g], axis=0, weights=w)
        hi = np.average([p.box_max for p in g], axis=0, weights=w)
        merged.append(Piece("", lo, hi, int(w.sum()), sorted({f for p in g for f in p.frames})))
    # stable ids: sort along x then z
    merged.sort(key=lambda p: (round(p.center[0], 1), p.center[2], p.center[1]))
    for i, p in enumerate(merged):
        p.id = f"p{i:04d}"
    # overlap flags: boxes that intersect noticeably but were not the same piece
    for i, a in enumerate(merged):
        for b in merged[i + 1:]:
            if 0.02 < box_iou(a, b) < merge_iou:
                a.overlaps.append(b.id)
                b.overlaps.append(a.id)
    return merged


def decode_barcodes(s: Session, pieces: list[Piece]) -> None:
    """Decode barcodes on hi-res stills; assign to the piece whose projected box contains the barcode."""
    import zxingcpp

    for st in s.stills:
        img = s.still(st)
        found = zxingcpp.read_barcodes(np.array(img))
        if not found:
            continue
        for bc in found:
            c = bc.position
            cx = np.mean([c.top_left.x, c.bottom_right.x])
            cy = np.mean([c.top_left.y, c.bottom_right.y])
            best, best_d = None, 1e9
            for p in pieces:
                uvz = project_world(st, p.corners())
                if np.any(uvz[:, 2] <= 0.1):
                    continue
                x0, x1, y0, y1 = uvz[:, 0].min(), uvz[:, 0].max(), uvz[:, 1].min(), uvz[:, 1].max()
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    d = np.hypot(cx - (x0 + x1) / 2, cy - (y0 + y1) / 2)
                    if d < best_d:
                        best, best_d = p, d
            if best is not None and not best.barcode:
                best.barcode = bc.text


def crop_for_piece(s: Session, p: Piece, max_crops: int = 2, pad: float = 0.15) -> list[Image.Image]:
    """Crops from the hi-res stills where the piece is best framed (different angles = spread of camera positions)."""
    from .session import mat4

    scored = []
    for st in s.stills:
        uvz = project_world(st, p.corners())
        if np.any(uvz[:, 2] <= 0.15):
            continue
        x0, x1, y0, y1 = uvz[:, 0].min(), uvz[:, 0].max(), uvz[:, 1].min(), uvz[:, 1].max()
        if x1 < 0 or y1 < 0 or x0 > st["width"] or y0 > st["height"]:
            continue
        w, h = x1 - x0, y1 - y0
        vis = (min(x1, st["width"]) - max(x0, 0)) * (min(y1, st["height"]) - max(y0, 0)) / max(w * h, 1)
        if vis < 0.8:
            continue
        scored.append((w * h, st, (x0 - pad * w, y0 - pad * h, x1 + pad * w, y1 + pad * h), mat4(st["transform"])[:3, 3]))
    scored.sort(key=lambda t: -t[0])
    chosen: list = []
    for cand in scored:  # prefer a second crop from a different camera position
        if all(np.linalg.norm(cand[3] - c[3]) > 0.25 for c in chosen) or not chosen:
            chosen.append(cand)
        if len(chosen) == max_crops:
            break
    if len(chosen) < max_crops:
        chosen += [c for c in scored if c not in chosen][: max_crops - len(chosen)]
    out = []
    for _, st, box, _pos in chosen:
        img = s.still(st)
        out.append(img.crop((max(0, int(box[0])), max(0, int(box[1])), min(img.width, int(box[2])), min(img.height, int(box[3])))))
    return out
