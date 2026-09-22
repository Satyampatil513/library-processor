"""Saves showcase artifacts (overlays, plots, prompts, model answers, verdicts, prices) under results/<session>/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .session import Session, mat4, project_world

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]


class Recorder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def json(self, name: str, obj) -> None:
        self.path(name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    def text(self, name: str, s: str) -> None:
        self.path(name).write_text(s, encoding="utf-8")

    def image(self, name: str, img: Image.Image) -> None:
        img.convert("RGB").save(self.path(name), quality=90)

    def bytes(self, name: str, data: bytes) -> None:
        self.path(name).write_bytes(data)


def mask_overlay(img: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    a = np.array(img).astype(np.float32)
    for i, m in enumerate(masks):
        c = np.array(PALETTE[i % len(PALETTE)], np.float32)
        a[m] = a[m] * 0.45 + c * 0.55
    return Image.fromarray(a.astype(np.uint8))


def draw_piece_boxes(img: Image.Image, frame_like: dict, pieces, labels: dict[str, str] | None = None) -> Image.Image:
    """Project each piece's 3D box into `img` (frame_like has transform + intrinsics for THIS image size)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for k, p in enumerate(pieces):
        uvz = project_world(frame_like, p.corners())
        if np.any(uvz[:, 2] <= 0.15):
            continue
        col = PALETTE[k % len(PALETTE)]
        for a, b in edges:
            d.line([tuple(uvz[a, :2]), tuple(uvz[b, :2])], fill=col, width=3)
        d.text((uvz[:, 0].min() + 3, uvz[:, 1].min() + 3), (labels or {}).get(p.id, p.id), fill=col)
    return img


def _plot_font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >=10
    except TypeError:
        return ImageFont.load_default()


def objects_plot(fp: dict, objects: list[dict], size: int = 1000) -> Image.Image:
    """Top-down (x,z) plot of every priced/reviewable object's box footprint, labelled with its
    title/description and its guessed price - a photo or raw JSON is correct but not visual; this is the
    "does this actually work" demo view. Takes the same `fp` (step 5) and `objects` (step 14's scene.json
    "objects" list) shapes already produced by a run, so it can be rendered straight from a saved
    `<session>.scene.json` + `floorplans` row with no live Session or API call needed.

    Zoom is set from the OBJECT cluster, not the walls: floor-plan wall fitting is known-noisy (see README
    "1-2 of 6 estimates land inside the 10cm target") and a stray wall segment spanning metres away used to
    drag the whole plot's scale out, squeezing every real object into a tiny corner. Walls still get drawn
    for context, just not used to size the view."""
    STATUS_COLOR = {"ok": (60, 200, 100), "needs_review": (240, 190, 40)}
    shown = [o for o in objects if o.get("status") != "not_an_object" and o.get("box")]

    obj_pts = [[o["box"]["min"][0], o["box"]["min"][2]] for o in shown] + \
              [[o["box"]["max"][0], o["box"]["max"][2]] for o in shown]
    arr = np.array(obj_pts, float) if obj_pts else np.array([[0.0, 0.0], [1.0, 1.0]])
    lo, hi = arr.min(0) - 0.6, arr.max(0) + 0.6
    sc = (size - 60) / max(1e-6, max(hi - lo))
    tf = lambda xz: (np.asarray(xz, float) - lo) * sc + 30

    img = Image.new("RGB", (size, size), (20, 20, 24))
    d = ImageDraw.Draw(img)
    font, font_sm = _plot_font(15), _plot_font(13)
    for w in fp.get("walls", []):
        d.line([tuple(tf(w["start"])), tuple(tf(w["end"]))], fill=(80, 80, 90), width=3)

    placed: list[tuple[float, float, float, float]] = []   # already-drawn label boxes, to dodge overlap

    def place(x: float, y_above: float, y_below: float, text: str, fnt, col) -> None:
        w, h = d.textbbox((0, 0), text, font=fnt)[2:]
        for y in (y_above - h, y_below, y_above - h - h - 2, y_below + h + 2):
            box = (x, y, x + w, y + h)
            if not any(a < box[2] and box[0] < c and b < box[3] and box[1] < e for a, b, c, e in placed):
                d.text((x, y), text, fill=col, font=fnt)
                placed.append(box)
                return
        d.text((x, y_below), text, fill=col, font=fnt)   # give up dodging, draw anyway

    total, cur = 0.0, ""
    for o in sorted(shown, key=lambda o: -abs(o["box"]["max"][0] - o["box"]["min"][0])):
        b = o["box"]
        x0, x1 = sorted((b["min"][0], b["max"][0]))
        z0, z1 = sorted((b["min"][2], b["max"][2]))
        (px0, pz0), (px1, pz1) = tf((x0, z0)), tf((x1, z1))
        px1, pz1 = max(px1, px0 + 6), max(pz1, pz0 + 6)   # thin objects still get a visible box
        col = STATUS_COLOR.get(o["status"], (150, 150, 160))
        d.rectangle([px0, pz0, px1, pz1], outline=col, width=2)
        name = (o.get("title") or (o.get("description") or o["id"]).split(",")[0])[:22]
        if o.get("list_price") is not None:
            price = f'{o["list_price"]:.0f} {o.get("currency") or ""}'.strip()
            total += o["list_price"]
            cur = o.get("currency") or cur
        else:
            price = "in review" if o["status"] == "needs_review" else "unpriced"
        place(px0, pz0, pz1 + 2, price, font, col)
        place(px0, pz0 - 15, pz1 + 15, name, font_sm, col)
    d.text((16, 12), f"{len(shown)} objects - guessed total ~{total:.0f} {cur}".strip(), fill=(255, 255, 255), font=font)
    return img


_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
_TOP_FACE = (2, 3, 7, 6)   # y=max corners, wound into a proper quad (see the corner-order comment below)


def _iso(pts: np.ndarray) -> np.ndarray:
    """World (x, y-up, z) -> classic 2:1 isometric screen (x2d, y2d). Taller (larger y) draws higher on
    screen (smaller y2d) with no extra flip needed, since y is subtracted directly."""
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return np.stack([(x - z) * 0.866, (x + z) * 0.5 - y], axis=-1)


def _box_corners(b: dict) -> np.ndarray:
    """8 corners in the same x/y/z-bit order as `Piece.corners()`, so `_BOX_EDGES`/`_TOP_FACE` line up."""
    lo, hi = np.array(b["min"], float), np.array(b["max"], float)
    return np.array([[hi[0] if xb else lo[0], hi[1] if yb else lo[1], hi[2] if zb else lo[2]]
                     for xb in (0, 1) for yb in (0, 1) for zb in (0, 1)])


def isometric_plot(fp: dict, objects: list[dict], size: int = 1200) -> Image.Image:
    """Isometric ("a bit 3D") view of every priced/reviewable object's box - filled top face + wireframe
    sides, back-to-front painter's-algorithm draw order - instead of `objects_plot`'s flat top-down
    footprint. Same inputs (fp + scene.json's "objects"), no live Session or API call needed."""
    STATUS_COLOR = {"ok": (60, 200, 100), "needs_review": (240, 190, 40)}
    shown = [o for o in objects if o.get("status") != "not_an_object" and o.get("box")]
    floor_y = fp.get("floor_y")
    if floor_y is None and shown:
        floor_y = min(o["box"]["min"][1] for o in shown)
    floor_y = floor_y or 0.0

    corners_by_id = {o["id"]: _box_corners(o["box"]) for o in shown}
    wall_pts = [np.array([[w["start"][0], floor_y, w["start"][1]], [w["end"][0], floor_y, w["end"][1]]])
               for w in fp.get("walls", [])]
    all_2d = [_iso(c) for c in corners_by_id.values()] + [_iso(w) for w in wall_pts]
    arr = np.concatenate(all_2d) if all_2d else np.array([[0.0, 0.0], [1.0, 1.0]])
    lo, hi = arr.min(0) - 0.5, arr.max(0) + 0.5
    sc = (size - 60) / max(1e-6, max(hi - lo))
    tf = lambda p2: (np.asarray(p2, float) - lo) * sc + 30

    img = Image.new("RGBA", (size, size), (20, 20, 24, 255))
    d = ImageDraw.Draw(img)
    font, font_sm = _plot_font(15), _plot_font(13)
    for w in wall_pts:
        a2, b2 = tf(_iso(w[0])), tf(_iso(w[1]))
        d.line([tuple(a2), tuple(b2)], fill=(80, 80, 90), width=3)

    # painter's algorithm: draw objects further from the viewer (smaller x+z) first, nearer ones over them
    order = sorted(shown, key=lambda o: o["box"]["min"][0] + o["box"]["max"][0] + o["box"]["min"][2] + o["box"]["max"][2])
    placed: list[tuple[float, float, float, float]] = []

    def place(x: float, y_above: float, text: str, fnt, col) -> None:
        tw, th = d.textbbox((0, 0), text, font=fnt)[2:]
        for y in (y_above - th, y_above - 2 * th - 2, y_above):
            box = (x, y, x + tw, y + th)
            if not any(a < box[2] and box[0] < c and b < box[3] and box[1] < e for a, b, c, e in placed):
                d.text((x, y), text, fill=col, font=fnt)
                placed.append(box)
                return
        d.text((x, y_above), text, fill=col, font=fnt)

    total, cur = 0.0, ""
    for o in order:
        c2d = tf(_iso(corners_by_id[o["id"]]))
        col = STATUS_COLOR.get(o["status"], (150, 150, 160))
        d.polygon([tuple(c2d[i]) for i in _TOP_FACE], fill=col + (70,), outline=col)
        for a_, b_ in _BOX_EDGES:
            d.line([tuple(c2d[a_]), tuple(c2d[b_])], fill=col, width=2)
        top_y, left_x = c2d[:, 1].min(), c2d[:, 0].min()
        name = (o.get("title") or (o.get("description") or o["id"]).split(",")[0])[:22]
        if o.get("list_price") is not None:
            price = f'{o["list_price"]:.0f} {o.get("currency") or ""}'.strip()
            total += o["list_price"]
            cur = o.get("currency") or cur
        else:
            price = "in review" if o["status"] == "needs_review" else "unpriced"
        place(left_x, top_y, price, font, col)
        place(left_x, top_y - 16, name, font_sm, col)
    d.text((16, 12), f"{len(shown)} objects - guessed total ~{total:.0f} {cur}".strip(), fill=(255, 255, 255), font=font)
    return img.convert("RGB")


def floorplan_plot(s: Session, fp: dict, size: int = 900) -> Image.Image:
    """Top-down view (ARKit x,z): camera path, depth points, fitted walls."""
    from .stitch import cloud, voxel_down

    pts = voxel_down(cloud(s, every=6), 0.05)
    cam = np.array([mat4(f["transform"])[:3, 3] for f in s.frames])
    allxz = np.vstack([pts[:, [0, 2]], cam[:, [0, 2]]])
    lo, hi = allxz.min(0) - 0.5, allxz.max(0) + 0.5
    sc = (size - 20) / max(hi - lo)
    tf = lambda xz: ((np.asarray(xz) - lo) * sc + 10)
    img = Image.new("RGB", (size, size), (20, 20, 24))
    d = ImageDraw.Draw(img)
    for x, y in tf(pts[:, [0, 2]]):
        d.point((x, y), fill=(90, 110, 140))
    d.line([tuple(p) for p in tf(cam[:, [0, 2]])], fill=(255, 200, 0), width=2)
    for i, w in enumerate(fp.get("walls", [])):
        d.line([tuple(tf(w["start"])), tuple(tf(w["end"]))], fill=(255, 60, 60), width=3)
    d.text((14, 12), f"top-down: grey=depth points, yellow=camera path, red=RANSAC walls ({len(fp.get('walls', []))})", fill=(255, 255, 255))
    return img
