"""Step 12 HITL review, the actual reviewable page - was previously just a SQLite table + `libpipe hitl`
JSON dump, with nothing a human could look at and judge. GET /hitl now shows each flagged object's real
crop photos, the reason it was flagged, and (when Jev logged a tie/low-confidence dispute) both models'
candidate values side by side with one-tap resolve buttons; everything else falls back to a manual edit
form pre-filled with the object's current values."""
from __future__ import annotations

import html
import json

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from . import hitl
from .config import Config
from .db import DB

router = APIRouter()

_pricing_deps = None   # lazy (isbndb, pricer): built on first resolve so an unconfigured server still starts


def _get_pricing_deps():
    """Best-effort, each piece independent: if ISBNdb/Keepa/Shopping/web-search can't be constructed
    (e.g. an API key is missing), whichever half fails is just skipped - the reviewer's edit is always
    saved regardless, same as before this existed."""
    global _pricing_deps
    if _pricing_deps is None:
        isbndb = pricer = None
        try:
            from .identity import ISBNdb
            isbndb = ISBNdb()
        except Exception:
            pass
        try:
            from .pricing import Pricer, WebSearchPricer
            web = None
            try:
                web = WebSearchPricer()
            except Exception:
                pass
            pricer = Pricer(web=web)
        except Exception:
            pass
        _pricing_deps = (isbndb, pricer)
    return _pricing_deps

CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; background: #f4f4f6; margin: 0; padding: 24px; color: #1c1c1f; }
h1 { font-size: 20px; }
.empty { color: #666; padding: 40px; text-align: center; }
.card { background: #fff; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.crops { display: flex; gap: 8px; margin: 8px 0; }
.crops img { height: 140px; border-radius: 6px; border: 1px solid #ddd; object-fit: cover; }
.reason { display: inline-block; background: #fff3cd; color: #7a5b00; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-bottom: 6px; }
.cand-row { display: flex; gap: 12px; margin: 10px 0; }
.cand { flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
.cand h4 { margin: 0 0 6px; font-size: 13px; color: #555; }
.cand .field { font-size: 13px; margin: 2px 0; }
button, input[type=submit] { cursor: pointer; border: none; border-radius: 6px; padding: 6px 14px; font-size: 13px; background: #2563eb; color: #fff; }
button.secondary { background: #e5e7eb; color: #1c1c1f; }
.manual { margin-top: 10px; }
.manual input, .manual textarea, .manual select { width: 100%; box-sizing: border-box; margin: 3px 0 8px; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
.manual label { font-size: 12px; color: #555; }
.badge { font-size: 11px; color: #999; }
"""


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _object_row(db: DB, object_id: str) -> dict | None:
    rows = db.q("SELECT * FROM objects WHERE id=?", (object_id,))
    if not rows:
        return None
    o = dict(rows[0])
    o["crops"] = json.loads(o["crops"]) if o["crops"] else []
    return o


def _candidate_block(label: str, cand) -> str:
    """cand is either one object-value dict (field dispute) or a list of them (partition dispute)."""
    items = cand if isinstance(cand, list) else [cand]
    rows = "".join(
        f'<div class="field"><b>{_e(o.get("title") or o.get("description") or "(untitled)")}</b> '
        f'- book: {_e(o.get("is_book"))}, old: {_e(o.get("is_old"))}, author: {_e(o.get("author"))}</div>'
        for o in items
    )
    return f'<div class="cand"><h4>{_e(label)}</h4>{rows}</div>'


def _item_html(db: DB, item: dict) -> str:
    obj = _object_row(db, item["object_id"]) or {}
    crops = obj.get("crops") or []
    img_tags = "".join(f'<img src="/img/{_e(c)}">' for c in crops) or '<span class="badge">no crop saved</span>'
    payload = item["payload"]
    candidates = payload.get("candidates")

    cand_html = ""
    pick_buttons = ""
    if candidates:
        cand_html = '<div class="cand-row">' + "".join(_candidate_block(k, v) for k, v in candidates.items()) + "</div>"
        pick_buttons = "".join(
            f'<form method="post" action="/hitl/{item["id"]}/resolve" style="display:inline">'
            f'<input type="hidden" name="pick" value="{_e(k)}">'
            f'<button type="submit">Use {_e(k)}</button></form>'
            for k in candidates
        )

    return f"""
    <div class="card">
      <div class="reason">{_e(item['reason'])}</div>
      <div class="badge">object {_e(item['object_id'])} - hitl #{item['id']}</div>
      <div class="crops">{img_tags}</div>
      <div><b>{_e(obj.get('title') or obj.get('description') or '(no description)')}</b>
        {'| book' if obj.get('is_book') else '| non-book'}
        {'| price ' + _e(obj.get('list_price')) + ' ' + _e(obj.get('currency')) if obj.get('list_price') else '| unpriced'}
      </div>
      {cand_html}
      <div>{pick_buttons}</div>
      <details class="manual">
        <summary>Manual correction</summary>
        <form method="post" action="/hitl/{item['id']}/resolve">
          <label>Title</label><input name="title" value="{_e(obj.get('title'))}">
          <label>Author</label><input name="author" value="{_e(obj.get('author'))}">
          <label>Description</label><textarea name="description" rows="2">{_e(obj.get('description'))}</textarea>
          <label>Is a book?</label>
          <select name="is_book"><option value="1" {'selected' if obj.get('is_book') else ''}>Yes</option><option value="0" {'selected' if not obj.get('is_book') else ''}>No</option></select>
          <label>Is old/antique?</label>
          <select name="is_old"><option value="1" {'selected' if obj.get('is_old') else ''}>Yes</option><option value="0" {'selected' if not obj.get('is_old') else ''}>No</option></select>
          <label>Is this actually an object? (No = bare wall/floor/ceiling/shadow/reflection)</label>
          <select name="is_object"><option value="1" {'selected' if obj.get('status') != 'not_an_object' else ''}>Yes</option><option value="0" {'selected' if obj.get('status') == 'not_an_object' else ''}>No - not a real object</option></select>
          <label>List price</label><input name="list_price" value="{_e(obj.get('list_price'))}">
          <label>Currency</label><input name="currency" value="{_e(obj.get('currency'))}">
          <button type="submit">Save correction</button>
        </form>
      </details>
      <form method="post" action="/hitl/{item['id']}/resolve" style="margin-top:8px">
        <button type="submit" class="secondary">Dismiss (keep as-is, just close this item)</button>
      </form>
    </div>"""


@router.get("/hitl", response_class=HTMLResponse)
def hitl_page():
    db = DB(Config().db_path)
    items = hitl.open_items(db)
    body = "".join(_item_html(db, i) for i in items) or '<div class="empty">Nothing to review right now.</div>'
    return f"<html><head><title>HITL review</title><style>{CSS}</style></head><body>" \
           f"<h1>HITL review ({len(items)} open)</h1>{body}</body></html>"


@router.post("/hitl/{item_id}/resolve")
def hitl_resolve(item_id: int, pick: str | None = Form(None), title: str | None = Form(None),
                 author: str | None = Form(None), description: str | None = Form(None),
                 is_book: str | None = Form(None), is_old: str | None = Form(None),
                 is_object: str | None = Form(None),
                 list_price: str | None = Form(None), currency: str | None = Form(None)):
    db = DB(Config().db_path)
    if pick:
        fix = {"pick": pick}
    else:
        fix: dict = {}
        if title is not None:
            fix["title"] = title or None
        if author is not None:
            fix["author"] = author or None
        if description is not None:
            fix["description"] = description or None
        if is_book is not None:
            fix["is_book"] = is_book == "1"
        if is_old is not None:
            fix["is_old"] = is_old == "1"
        if is_object is not None:
            fix["is_object"] = is_object == "1"
        if list_price:
            try:
                fix["list_price"] = float(list_price)
            except ValueError:
                pass
        if currency:
            fix["currency"] = currency
    cfg = Config()
    isbndb, pricer = _get_pricing_deps()
    hitl.resolve(db, item_id, fix, isbndb=isbndb, pricer=pricer,
                image_base_url=cfg.key("IMAGE_BASE_URL"), output_dir=cfg.output_dir)
    return RedirectResponse("/hitl", status_code=303)
