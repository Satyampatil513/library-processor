"""Step 7 (group into regions) and step 8 (region call inputs/outputs)."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .db import DB
from .pieces import Piece, crop_for_piece
from .prep import nearest_pose_time_window, words_between
from .session import Session, project_world


@dataclass
class Region:
    id: str
    pieces: list[Piece]

    @property
    def piece_ids(self) -> list[str]:
        return [p.id for p in self.pieces]


def make_regions(pieces: list[Piece], size: int = 15, gap_m: float = 0.4) -> list[Region]:
    """~`size` objects per region, spatially coherent, never splitting overlapping pieces."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not pieces:
        return []
    by_id ={p.id: i for i, p in enumerate(pieces)}
    # atoms: overlap-connected pieces that must stay together
    rows, cols = [], []
    for i, p in enumerate(pieces):
        for o in p.overlaps:
            rows.append(i)
            cols.append(by_id[o])
    n = len(pieces)
    _, atom_lbl = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    atoms: dict[int, list[Piece]] = {}
    for p, a in zip(pieces, atom_lbl):
        atoms.setdefault(int(a), []).append(p)
    atom_list = list(atoms.values())
    centers = np.array([np.mean([p.center for p in a], axis=0) for a in atom_list])
    # spatial components (separate shelves / walls)
    d = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    ii, jj = np.where(d < gap_m)
    _, comp = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(atom_list),) * 2), directed=False)
    regions: list[Region] = []
    for c in sorted(set(comp)):
        idx = np.where(comp == c)[0]
        pts = centers[idx][:, [0, 2]]
        # principal horizontal direction along the shelf, then shelf rows by height
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False) if len(idx) > 1 else (0, 0, np.eye(2))
        along = (pts - pts.mean(axis=0)) @ vt[0]
        row = np.round(centers[idx][:, 1] / 0.3).astype(int)
        order = idx[np.lexsort((along, -row))]
        chunk: list[Piece] = []
        for ai in order:
            chunk.extend(sorted(atom_list[ai], key=lambda p: p.center[0]))
            if len(chunk) >= size:
                regions.append(Region(f"r{len(regions):03d}", chunk))
                chunk = []
        if chunk:
            if regions and len(chunk) < size // 3 and len(regions[-1].pieces) + len(chunk) <= size * 4 // 3 \
                    and np.linalg.norm(regions[-1].pieces[-1].center - chunk[0].center) < gap_m * 3:
                regions[-1].pieces.extend(chunk)
            else:
                regions.append(Region(f"r{len(regions):03d}", chunk))
    return regions


# ------------------------------------------------------------------ step 8 ---

SYSTEM = """You identify physical objects on shelves from photos, for an inventory. Rules:
- Decide from the images, the voice/notes text and the barcode only. NO web search. NO prices.
- Every listed piece id must appear in exactly one object. Pieces that are parts of the same physical object
  (e.g. a spine and a cover seen separately, or a box set) go into the same object.
- "overlaps physically with" = their 3D boxes intersect: almost always the same object or a stack, so merge
  unless the images clearly show two separate touching items.
- "near" = close in 3D space but boxes do NOT intersect: usually separate objects sitting next to each other
  (e.g. two books side by side) - do NOT merge just because they are listed as near, decide from the images.
- Piece ids you were not given are outside this region; do not report gaps or missing shelf space, and do not
  invent objects that are not in one of the numbered crops or the overview image.
- If a title/author is not legible, use null - never guess from memory.
- is_old: true only for visibly old/antique/vintage editions.
- For non-books give a short physical description (what it is, colour, size).
- is_object: false when the piece is only bare wall, floor, ceiling, shadow, reflection or another non-item fragment.
Return ONLY JSON:
{"objects":[{"piece_ids":["p0001"],"is_object":true,"is_book":true,"title":"...","author":"...","is_old":false,
"description":null,"reasoning":"one or two sentences citing what you saw/heard"}]}"""


def _jpeg(img: Image.Image, max_side: int = 1024) -> bytes:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    b = io.BytesIO()
    img.save(b, "JPEG", quality=88)
    return b.getvalue()


def best_group_still(s: Session, pieces: list[Piece], min_visible_frac: float = 0.5) -> dict | None:
    """The still that shows the most of `pieces` at a usable size, not just with their centers in frame.

    BUG FIXED: the previous version scored a still by how many piece *centers* fell inside the image
    bounds, with no check on box size or occlusion. A still where every piece was a speck at the image
    edge scored the same as one showing them large and centred, so the overview/box-preview images could
    point labels at nothing (the reported "objects literally outside the region" symptom). Now each piece
    only counts if at least `min_visible_frac` of its projected box is inside frame, and stills are ranked
    by how many pieces clear that bar, tie-broken by total visible area.
    """
    best, best_key = None, (0, 0.0)   # a still with nothing visible must not "win" just for being first
    for st in s.stills:
        ok_count, area = 0, 0.0
        for p in pieces:
            uvz = project_world(st, p.corners())
            if np.any(uvz[:, 2] <= 0.15):
                continue
            x0, x1, y0, y1 = uvz[:, 0].min(), uvz[:, 0].max(), uvz[:, 1].min(), uvz[:, 1].max()
            full = max((x1 - x0) * (y1 - y0), 1e-6)
            cx0, cx1 = max(x0, 0), min(x1, st["width"])
            cy0, cy1 = max(y0, 0), min(y1, st["height"])
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            vis = (cx1 - cx0) * (cy1 - cy0)
            if vis / full < min_visible_frac:
                continue
            ok_count += 1
            area += vis
        key = (ok_count, area)
        if key > best_key:
            best, best_key = st, key
    return best


def overview_image(s: Session, region: Region) -> Image.Image | None:
    """Best still seeing most of the region, with piece ids drawn on it."""
    best = best_group_still(s, region.pieces)
    if best is None:
        return None
    img = s.still(best)
    dr = ImageDraw.Draw(img)
    for p in region.pieces:
        u, v, z = project_world(best, p.center[None])[0]
        if z > 0.15 and 0 <= u < img.width and 0 <= v < img.height:
            dr.text((u - 12, v - 6), p.id, fill=(255, 0, 0))
    return img


def build_inputs(db: DB, s: Session, region: Region, exemplars: list[dict], crop_rotate_deg: int = 0) -> tuple[str, list[bytes]]:
    images: list[bytes] = []
    lines = [f"REGION {region.id}: {len(region.pieces)} pieces, listed in shelf order.", ""]
    ov = overview_image(s, region)
    if ov is not None:
        images.append(_jpeg(ov, 1600))
        lines.append("Image 1 = overview of the region with piece ids drawn in red.")
    near_radius = 0.20   # metres; a real-world proximity hint distinct from the rare true-3D-box-overlap flag
    for p in region.pieces:
        crops = crop_for_piece(s, p, 2, rotate_deg=crop_rotate_deg)
        first = len(images) + 1
        for c in crops:
            images.append(_jpeg(c, 800))
        sz = np.sort(p.size)[::-1] * 100
        note_txt = [n["text"] for n in s.notes if n.get("target_point_world")
                    and np.linalg.norm(np.array(n["target_point_world"]) - p.center) < 0.35]
        lines.append(f"- {p.id}: images {first}-{len(images)}" if crops else f"- {p.id}: (no crop available)")
        lines.append(f"    3D box (cm, large to small): {sz[0]:.0f} x {sz[1]:.0f} x {sz[2]:.0f}")
        if p.overlaps:
            lines.append(f"    overlaps physically with: {', '.join(p.overlaps)} (likely the same object or stacked)")
        near = [o.id for o in region.pieces if o.id != p.id and o.id not in p.overlaps
               and np.linalg.norm(o.center - p.center) < near_radius]
        if near:
            lines.append(f"    near (not overlapping): {', '.join(near)}")
        if p.barcode:
            lines.append(f"    barcode read: {p.barcode}")
        if note_txt:
            lines.append(f"    typed notes at this object: {note_txt}")
    win = nearest_pose_time_window(s, np.mean([p.center for p in region.pieces], axis=0))
    if win:
        voice = words_between(db, s.id, *win)
        if voice:
            lines += ["", f"VOICE while filming this region: \"{voice}\""]
    if exemplars:
        lines += ["", "VERIFIED EXAMPLES from earlier regions (format reference and past corrections):"]
        for e in exemplars[:6]:
            lines.append("  " + json.dumps(e, ensure_ascii=False))
    return "\n".join(lines), images


def normalise(raw: dict, piece_ids: list[str]) -> list[dict]:
    """Enforce the partition: unknown ids dropped, duplicates kept once, missing ids become singletons."""
    seen, out = set(), []
    for o in raw.get("objects", []):
        ids = [i for i in o.get("piece_ids", []) if i in piece_ids and i not in seen]
        if not ids:
            continue
        seen.update(ids)
        out.append({"piece_ids": ids, "is_object": bool(o.get("is_object", True)), "is_book": bool(o.get("is_book")), "title": o.get("title") or None,
                    "author": o.get("author") or None, "is_old": bool(o.get("is_old")),
                    "description": o.get("description") or None, "reasoning": o.get("reasoning", "")})
    for i in piece_ids:
        if i not in seen:
            out.append({"piece_ids": [i], "is_object": True, "is_book": False, "title": None, "author": None, "is_old": False,
                        "description": None, "reasoning": "model omitted this piece"})
    return out
