"""Step 4 STITCH [F5]: ARKit relocalization gives a shared world frame; verify the overlap on the
server with depth point clouds, spread residual drift across the new room, flag failures.

SOURCE DIAGRAM TARGET, not measured by this code: overlap within 2-3cm. Never run on two real overlapping
sessions - unit-tested on synthetic point clouds only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .session import Session

VOXEL = 0.005  # 0.5cm


def cloud(s: Session, every: int = 5) -> np.ndarray:
    pts = [s.unproject_depth(f, stride=4) for f in s.frames[::every]]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def voxel_down(p: np.ndarray, v: float = VOXEL) -> np.ndarray:
    if len(p) == 0:
        return p
    _, idx = np.unique(np.floor(p / v).astype(np.int64), axis=0, return_index=True)
    return p[idx]


@dataclass
class StitchResult:
    ok: bool
    median_err_m: float | None
    overlap_points: int
    translation: np.ndarray        # correction applied at end of the session (drift spread)
    reason: str = ""


def check_overlap(prev: np.ndarray, new: np.ndarray, tol: float) -> tuple[float | None, int, np.ndarray]:
    """Median nearest-neighbour distance new->prev over points that have a neighbour within 15cm,
    plus the mean offset vector (used as the drift estimate)."""
    from scipy.spatial import cKDTree

    if len(prev) == 0 or len(new) == 0:
        return None, 0, np.zeros(3)
    d, i = cKDTree(prev).query(new, distance_upper_bound=0.15)
    m = np.isfinite(d)
    if m.sum() < 200:
        return None, int(m.sum()), np.zeros(3)
    offset = (prev[i[m]] - new[m]).mean(axis=0)
    # Median residual after removing the rigid offset is the honest local agreement;
    # the raw median is what drift looks like.
    return float(np.median(d[m])), int(m.sum()), offset


def stitch(prev: Session, new: Session, tol: float = 0.03) -> StitchResult:
    """Pose-based stitch: both sessions share a world frame when ARKit relocalized [F5]."""
    if new.relocalized_against != prev.id:
        return StitchResult(False, None, 0, np.zeros(3), "not relocalized against this session")
    p, n = voxel_down(cloud(prev)), voxel_down(cloud(new))
    med, cnt, off = check_overlap(p, n, tol)
    if med is None:
        return StitchResult(False, None, cnt, np.zeros(3), "insufficient overlap - re-record room")
    return StitchResult(med <= tol, med, cnt, off,
                        "" if med <= tol else f"overlap {med*100:.1f}cm > {tol*100:.0f}cm - re-record room")


def spread_drift(points_world: np.ndarray, times: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Distribute the measured end-of-overlap offset linearly over the session, so early points
    (near the relocalization anchor) stay put and late points absorb the drift."""
    if len(points_world) == 0:
        return points_world
    w = (times - times.min()) / max(times.max() - times.min(), 1e-9)
    return points_world + w[:, None] * offset[None, :]
