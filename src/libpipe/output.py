"""Step 14 OUTPUT: scene graph + floor plan.
id, 3D box, is_book, title, author, is_old, ISBN, list_price, currency, price_concept, price_source, 2 crops, provenance."""
from __future__ import annotations

import json
from pathlib import Path

from .db import DB

FIELDS = ("id", "box", "is_book", "title", "author", "is_old", "description", "isbn", "list_price", "currency",
          "price_concept", "price_source", "price_note", "status", "crops", "provenance")


def export(db: DB, session_id: str, out_dir: Path) -> Path:
    objs = []
    for r in db.q("SELECT * FROM objects WHERE session_id=? ORDER BY id", (session_id,)):
        o = {k: r[k] for k in FIELDS}
        for k in ("box", "crops", "provenance"):
            o[k] = json.loads(o[k]) if o[k] else None
        o["is_book"], o["is_old"] = bool(o["is_book"]), bool(o["is_old"])
        objs.append(o)
    fp = db.q("SELECT data FROM floorplans WHERE session_id=?", (session_id,))
    doc = {"session_id": session_id, "floor_plan": json.loads(fp[0]["data"]) if fp else None, "objects": objs,
           "summary": {"objects": len(objs), "needs_review": sum(o["status"] == "needs_review" for o in objs),
                       "not_an_object": sum(o["status"] == "not_an_object" for o in objs),
                       "unpriced": sum(o["list_price"] is None and o["status"] != "not_an_object" for o in objs)}}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{session_id}.scene.json"
    p.write_text(json.dumps(doc, indent=2))
    return p
