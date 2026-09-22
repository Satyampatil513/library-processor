"""Step 12 HITL review [F16]: unreadable spines, Jev ties, low-confidence fields, unpriced items; one-tap fix.
SOURCE DIAGRAM TARGET, not measured by this code: ~8-12% of objects at start, falling over time."""
from __future__ import annotations

import json
from pathlib import Path

from .db import DB
from .learning import add_exemplar

IDENTITY_COLS = {"is_book", "title", "author", "isbn"}


def enqueue(db: DB, object_id: str, reason: str, payload: dict | None = None) -> None:
    db.x("INSERT INTO hitl(object_id, reason, payload) VALUES(?,?,?)", (object_id, reason, json.dumps(payload or {}, default=str)))


def open_items(db: DB) -> list[dict]:
    return [dict(r) | {"payload": json.loads(r["payload"] or "{}")} for r in db.q("SELECT * FROM hitl WHERE status='open'")]


def _reprice(db: DB, object_id: str, isbndb, pricer, image_base_url: str | None, output_dir: Path | None) -> None:
    """Re-runs step 10/11 (identity + price) for one object after a HITL edit changed its identity
    without the reviewer typing a price by hand. Best-effort: any failure (no isbndb/pricer configured,
    network error, nothing found) just leaves the object as the reviewer left it - never re-opens the
    dispute the human just closed."""
    o = dict(db.q("SELECT * FROM objects WHERE id=?", (object_id,))[0])
    country_rows = db.q("SELECT country FROM sessions WHERE id=?", (o["session_id"],))
    country = country_rows[0]["country"] if country_rows else "US"
    crops = json.loads(o["crops"]) if o["crops"] else []

    if o["is_book"]:
        isbn = o["isbn"]
        if not isbn and isbndb is not None:
            isbn, _how = isbndb.resolve({"barcode": o["barcode"], "title": o["title"], "author": o["author"]})
        if not isbn:
            return
        pr = pricer.price_book(isbn, country)
        if pr:
            db.x("UPDATE objects SET isbn=?, list_price=?, currency=?, price_concept=?, price_source=?, "
                "price_note=?, status='reviewed' WHERE id=?",
                (isbn, pr.list_price, pr.currency, pr.price_concept, pr.price_source,
                 ("estimate; " if pr.estimate else "") + (pr.fx_note or ""), object_id))
        elif isbn != o["isbn"]:
            db.x("UPDATE objects SET isbn=? WHERE id=?", (isbn, object_id))
    else:
        url = f"{image_base_url.rstrip('/')}/{crops[0]}" if image_base_url and crops else None
        crop_bytes = (output_dir / crops[0]).read_bytes() if crops and output_dir else None
        pr = pricer.price_nonbook(url, o["description"] or "", country, crop_bytes)
        if pr:
            db.x("UPDATE objects SET list_price=?, currency=?, price_concept=?, price_source=?, price_note=?, "
                "status='reviewed' WHERE id=?",
                (pr.list_price, pr.currency, pr.price_concept, pr.price_source,
                 ("estimate; " if pr.estimate else "") + (pr.fx_note or ""), object_id))


def resolve(db: DB, item_id: int, fix: dict, *, isbndb=None, pricer=None,
           image_base_url: str | None = None, output_dir: Path | None = None) -> None:
    """`fix` holds the corrected fields (title, author, is_book, is_old, isbn, list_price, ...) or
    {"pick": "fable"|"astra"} to accept one candidate (one-tap). Also feeds learning [F13].

    If the edit changes identity (title/author/is_book/isbn) and the reviewer did NOT also type a
    list_price by hand, and `isbndb`/`pricer` are supplied, step 10/11 are re-run so the object doesn't
    stay silently priced (or mispriced) against its old, wrong identity - the gap noted in the README as
    "editing an identity in HITL does not re-price the object"."""
    row = db.q("SELECT * FROM hitl WHERE id=?", (item_id,))[0]
    payload = json.loads(row["payload"] or "{}")
    if "pick" in fix:
        cand = (payload.get("candidates") or {}).get(fix["pick"])
        if cand is None:
            raise ValueError("no such candidate on this item")
        fix = {k: cand[k] for k in ("is_book", "title", "author", "is_old", "description") if k in cand}
    # is_object isn't a real `objects` column (Astra/Fable's is_object:false only ever shows up as
    # status='not_an_object' - see pipeline._store_object) - a reviewer saying "not actually an object" must
    # route the same way, which the HITL form previously had no way to express at all.
    not_object = fix.get("is_object") is False
    cols = {"is_book", "title", "author", "is_old", "description", "isbn", "list_price", "currency", "price_concept", "price_source"}
    upd = {k: v for k, v in fix.items() if k in cols}
    status = "not_an_object" if not_object else "reviewed"
    if upd:
        sets = ",".join(f"{k}=?" for k in upd)
        db.x(f"UPDATE objects SET {sets}, status=? WHERE id=?", [*upd.values(), status, row["object_id"]])
    else:
        db.x("UPDATE objects SET status=? WHERE id=?", (status, row["object_id"]))
    db.x("UPDATE hitl SET status='done', resolution=? WHERE id=?", (json.dumps(fix), item_id))
    if not not_object and ({"title", "author", "is_book", "is_old"} & set(upd)):
        add_exemplar(db, "hitl", {"reason": row["reason"], "correct": {k: upd[k] for k in upd if k in ("is_book", "title", "author", "is_old")}})
    if not not_object and (IDENTITY_COLS & set(upd)) and "list_price" not in upd and pricer is not None:
        try:
            _reprice(db, row["object_id"], isbndb, pricer, image_base_url, output_dir)
        except Exception:
            pass   # best-effort, per the docstring above: a pricing hiccup must never lose the review just saved
