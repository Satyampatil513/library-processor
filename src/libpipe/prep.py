"""Step 3 PREP: align frames + stills into frames.db; transcribe audio with word timestamps."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .db import DB
from .session import Session, mat4


def align(db: DB, s: Session) -> None:
    """Each frame -> nearest still in time (same session clock, ARKit timestamps)."""
    still_ts = np.array([st["ar_timestamp"] for st in s.stills]) if s.stills else None
    for fr in s.frames:
        near = None
        if still_ts is not None:
            near = s.stills[int(np.argmin(np.abs(still_ts - fr["ar_timestamp"])))]["index"]
        q = fr.get("quality") or {}
        db.upsert("frames", {"session_id": s.id, "idx": fr["index"], "ar_timestamp": fr["ar_timestamp"],
                             "rgb": fr["rgb"], "tracking": fr["tracking_state"],
                             "blur": q.get("blur_score"), "nearest_still": near})


def transcribe_session(db: DB, s: Session, transcribe: Callable[[str], list[dict]]) -> list[dict]:
    """Word times are audio-relative; shift onto the AR clock with audio.start_ar_timestamp."""
    a = s.manifest["audio"]
    if not a.get("file") or not (s.root / a["file"]).exists():
        return []      # e.g. imported scans without audio
    words = transcribe(str(s.root / a["file"]))
    db.x("DELETE FROM words WHERE session_id=?", (s.id,))
    for w in words:
        db.upsert("words", {"session_id": s.id, "start": a["start_ar_timestamp"] + w["start"],
                            "end": a["start_ar_timestamp"] + w["end"], "word": w["word"]})
    return words


def words_between(db: DB, session_id: str, t0: float, t1: float) -> str:
    rows = db.q("SELECT word FROM words WHERE session_id=? AND end>=? AND start<=? ORDER BY start",
                (session_id, t0, t1))
    return " ".join(r["word"] for r in rows).strip()


def nearest_pose_time_window(s: Session, point: np.ndarray, radius_m: float = 1.2, max_frames: int = 40):
    """AR-time window(s) when the camera was aimed near a world point (for voice/notes lookup)."""
    hits = []
    for fr in s.frames:
        T = mat4(fr["transform"])
        pos, fwd = T[:3, 3], -T[:3, 2]
        v = point - pos
        d = np.linalg.norm(v)
        if 0.2 < d < 4 and np.dot(v / d, fwd) > np.cos(np.radians(35)):
            hits.append(fr["ar_timestamp"])
    if not hits:
        return None
    hits = sorted(hits)[:max_frames] if len(hits) > max_frames else hits
    return hits[0] - 1.0, hits[-1] + 1.0
