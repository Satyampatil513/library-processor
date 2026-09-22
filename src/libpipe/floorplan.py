"""Step 5 FLOOR PLAN [F4]: walls + room dimensions.

NOTE: the iOS app does not currently save ARKit mesh classification (walls/doors/windows), so walls are
recovered from the depth point cloud: ARKit y is up, project points between 0.3m and 1.8m above the
floor onto the XZ plane, then RANSAC-fit line segments. Doors/windows are NOT detectable from depth alone
(they need mesh classification on device or a vision pass) and are returned empty, marked unavailable.
SOURCE DIAGRAM TARGET, not measured by this code: wall/door ~90%+, dimensions within 10cm.
Measured once against a hand-measured room: 1 of 3 dimensions was within 10cm (see results/README.md).
"""
from __future__ import annotations

import numpy as np

from .session import Session
from .stitch import cloud, voxel_down


def floor_height(p: np.ndarray) -> float:
    return float(np.percentile(p[:, 1], 2))


def ransac_lines(xz: np.ndarray, n_lines: int = 8, tol: float = 0.05, iters: int = 400,
                 min_inliers: int = 300, rng=None):
    rng = rng or np.random.default_rng(0)
    pts, lines = xz.copy(), []
    for _ in range(n_lines):
        if len(pts) < min_inliers:
            break
        best = None
        for _ in range(iters):
            a, b = pts[rng.choice(len(pts), 2, replace=False)]
            d = b - a
            n = np.linalg.norm(d)
            if n < 0.5:
                continue
            nrm = np.array([-d[1], d[0]]) / n
            dist = np.abs((pts - a) @ nrm)
            inl = dist < tol
            if best is None or inl.sum() > best[0].sum():
                best = (inl, a, d / n)
        if best is None or best[0].sum() < min_inliers:
            break
        inl, a, dirv = best
        proj = (pts[inl] - a) @ dirv
        lo, hi = np.percentile(proj, [1, 99])
        lines.append({"start": (a + lo * dirv).tolist(), "end": (a + hi * dirv).tolist(),
                      "length_m": float(hi - lo), "support": int(inl.sum())})
        pts = pts[~inl]
    return lines


def merge_duplicate_walls(lines: list[dict], angle_cos: float = 0.97, perp_tol: float = 0.20) -> list[dict]:
    """A real wall's depth points are noisier than RANSAC's fit tolerance (5cm), so the same physical wall
    sometimes comes back as two separate near-parallel, near-coincident segments instead of one - a user
    noticed exactly this in the top-down plot ("wall should be single line"). Repeatedly merges any pair of
    lines whose directions are within ~14 degrees (angle_cos) and whose perpendicular offset is under
    `perp_tol`, support-weighted, until no such pair remains. Two genuinely separate parallel walls (e.g.
    opposite sides of a room) are metres apart and never meet `perp_tol`, so they are left as two lines."""
    out = [dict(l) for l in lines]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                sa, ea, sb, eb = np.array(a["start"]), np.array(a["end"]), np.array(b["start"]), np.array(b["end"])
                da, db = ea - sa, eb - sb
                na, nb = np.linalg.norm(da), np.linalg.norm(db)
                if na < 1e-6 or nb < 1e-6:
                    continue
                ua, ub = da / na, db / nb
                if abs(float(np.dot(ua, ub))) < angle_cos:
                    continue
                base, other, u = (a, b, ua) if a["support"] >= b["support"] else (b, a, ub)
                bs = np.array(base["start"])
                os_, oe = np.array(other["start"]) - bs, np.array(other["end"]) - bs
                perp = lambda v: v - np.dot(v, u) * u
                if max(np.linalg.norm(perp(os_)), np.linalg.norm(perp(oe))) > perp_tol:
                    continue
                pts = np.array([sa, ea, sb, eb]) - bs
                w = np.repeat([a["support"], b["support"]], 2).astype(float)
                proj, perp_v = pts @ u, pts - np.outer(pts @ u, u)
                center_perp = np.average(perp_v, axis=0, weights=w)
                lo, hi = proj.min(), proj.max()
                out[i] = {"start": (bs + center_perp + lo * u).tolist(), "end": (bs + center_perp + hi * u).tolist(),
                         "length_m": float(hi - lo), "support": int(a["support"] + b["support"])}
                del out[j]
                changed = True
                break
            if changed:
                break
    return out


def _plane_peaks(vals: np.ndarray, bin_m: float = 0.02, min_frac: float = 0.25):
    """Positions of dense planes along an axis (walls / floor / ceiling show up as histogram peaks)."""
    lo, hi = np.percentile(vals, [0.5, 99.5])
    hist, edges = np.histogram(vals, bins=max(int((hi - lo) / bin_m), 10), range=(lo, hi))
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    ctr = (edges[:-1] + edges[1:]) / 2
    peaks = [ctr[i] for i in range(1, len(hist) - 1)
             if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= min_frac * hist.max()]
    return np.array(peaks)


def room_dims_from_planes(p: np.ndarray, cam_xz: np.ndarray, dom_dir: np.ndarray, floor: float) -> dict:
    """Wall-to-wall distances: along each horizontal axis take the nearest dense wall plane on either side of the
    camera path; height = floor peak to ceiling peak. More robust than raw percentile extents (which include doorways
    and corridors seen through them)."""
    band = p[(p[:, 1] > floor + 0.3) & (p[:, 1] < floor + 1.8)][:, [0, 2]]
    out = {}
    for name, axis in (("width_m", dom_dir), ("depth_m", np.array([-dom_dir[1], dom_dir[0]]))):
        pk, cam = _plane_peaks(band @ axis), float(np.median(cam_xz @ axis))   # median: inside the room even if the walk continues into a corridor
        below, above = pk[pk < cam], pk[pk > cam]
        out[name] = float(above.min() - below.max()) if len(below) and len(above) else None   # nearest wall each side
    ypk = _plane_peaks(p[:, 1], min_frac=0.15)
    out["height_m"] = float(ypk.max() - ypk.min()) if len(ypk) >= 2 else None
    return out


def floor_plan(s: Session) -> dict:
    p = voxel_down(cloud(s, every=3), 0.005)
    if len(p) < 500:
        return {"walls": [], "doors": [], "windows": [], "dimensions": None, "note": "too little depth"}
    fh = floor_height(p)
    band = p[(p[:, 1] > fh + 0.3) & (p[:, 1] < fh + 1.8)]
    walls = merge_duplicate_walls(ransac_lines(band[:, [0, 2]]))
    dims = None
    if walls:
        d = np.array(walls[0]["end"]) - np.array(walls[0]["start"])
        u = d / np.linalg.norm(d)
        v = np.array([-u[1], u[0]])
        a, b = band[:, [0, 2]] @ u, band[:, [0, 2]] @ v
        dims = {"width_m": float(np.percentile(a, 99) - np.percentile(a, 1)),
                "depth_m": float(np.percentile(b, 99) - np.percentile(b, 1)),
                "height_m": float(np.percentile(p[:, 1], 99) - fh)}
    plane_dims = None
    if walls:
        from .session import mat4
        cam = np.array([mat4(f["transform"])[:3, 3] for f in s.frames])
        d0 = np.array(walls[0]["end"]) - np.array(walls[0]["start"])
        plane_dims = room_dims_from_planes(p, cam[:, [0, 2]], d0 / np.linalg.norm(d0), fh)
    return {"floor_y": fh, "walls": walls, "doors": [], "windows": [], "dimensions": dims,
            "dimensions_from_planes": plane_dims,
            "note": "doors/windows unavailable without ARKit mesh classification"}
