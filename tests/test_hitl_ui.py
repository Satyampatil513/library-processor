import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBPIPE_DATA", str(tmp_path))
    monkeypatch.delenv("UPLOAD_TOKEN", raising=False)
    import importlib

    from libpipe import config, hitl_ui, server
    importlib.reload(config)
    importlib.reload(hitl_ui)
    importlib.reload(server)

    from libpipe.db import DB
    db = DB(server.cfg.db_path)
    return TestClient(server.app), db, tmp_path


def seed(db, oid: str, reason: str, payload: dict, **obj_fields):
    row = {"id": oid, "session_id": "s", "region_id": "r0", "status": "needs_review", "crops": [],
          "title": None, "author": None, "is_book": 0, "is_old": 0, "description": None}
    row.update(obj_fields)
    db.upsert("objects", row)
    db.x("INSERT INTO hitl(object_id, reason, payload) VALUES(?,?,?)", (oid, reason, json.dumps(payload)))
    return db.q("SELECT id FROM hitl WHERE object_id=?", (oid,))[0]["id"]


def test_empty_queue(client):
    c, db, _ = client
    r = c.get("/hitl")
    assert r.status_code == 200 and "Nothing to review" in r.text


def test_candidates_render_and_pick_resolves(client):
    c, db, _ = client
    cand = {"fable": {"title": "Dune", "author": "Frank Herbert", "is_book": True, "is_old": False},
           "astra": {"title": "Dyne", "author": "F. Herbert", "is_book": True, "is_old": False}}
    item_id = seed(db, "obj1", "Jev tie / low confidence", {"candidates": cand}, title="Dune", author="Frank Herbert", is_book=1)

    page = c.get("/hitl")
    assert "Use fable" in page.text and "Use astra" in page.text and "Dyne" in page.text

    r = c.post(f"/hitl/{item_id}/resolve", data={"pick": "astra"}, follow_redirects=False)
    assert r.status_code == 303
    row = db.q("SELECT title, author, status FROM objects WHERE id='obj1'")[0]
    assert row["title"] == "Dyne" and row["author"] == "F. Herbert" and row["status"] == "reviewed"
    assert db.q("SELECT status FROM hitl WHERE id=?", (item_id,))[0]["status"] == "done"
    assert db.q("SELECT * FROM hitl WHERE status='open'") == []


def test_manual_correction_resolves(client):
    c, db, _ = client
    item_id = seed(db, "obj2", "unpriced", {})
    r = c.post(f"/hitl/{item_id}/resolve", data={"title": "Fixed", "author": "Someone", "description": "desc",
                                                  "is_book": "1", "is_old": "0", "list_price": "12.5", "currency": "USD"})
    assert r.status_code in (200, 303)
    row = db.q("SELECT title, list_price, currency, is_book, status FROM objects WHERE id='obj2'")[0]
    assert row["title"] == "Fixed" and row["list_price"] == 12.5 and row["currency"] == "USD"
    assert row["is_book"] == 1 and row["status"] == "reviewed"


def test_manual_correction_can_set_is_book_false(client):
    """Regression guard: booleans here are <select> Yes/No, not checkboxes - HTML checkboxes silently omit
    themselves from form data when unchecked, which would make 'turn this off' impossible to submit."""
    c, db, _ = client
    item_id = seed(db, "obj3", "unpriced", {}, is_book=1, title="Maybe A Book")
    r = c.post(f"/hitl/{item_id}/resolve", data={"is_book": "0", "is_old": "0"})
    assert r.status_code in (200, 303)
    assert db.q("SELECT is_book FROM objects WHERE id='obj3'")[0]["is_book"] == 0


def test_manual_correction_can_mark_not_an_object(client):
    c, db, _ = client
    item_id = seed(db, "obj6", "low confidence", {}, title="Blurry patch")
    r = c.post(f"/hitl/{item_id}/resolve", data={"is_object": "0", "is_book": "0", "is_old": "0"})
    assert r.status_code in (200, 303)
    assert db.q("SELECT status FROM objects WHERE id='obj6'")[0]["status"] == "not_an_object"


def test_dismiss_with_no_fields_just_closes_item(client):
    c, db, _ = client
    item_id = seed(db, "obj4", "sample flagged by Jev", {}, title="Untouched")
    r = c.post(f"/hitl/{item_id}/resolve", data={})
    assert r.status_code in (200, 303)
    assert db.q("SELECT title, status FROM objects WHERE id='obj4'")[0]["title"] == "Untouched"
    assert db.q("SELECT status FROM hitl WHERE id=?", (item_id,))[0]["status"] == "done"


def test_crops_are_served_and_escaped(client):
    c, db, tmp_path = client
    (tmp_path / "output" / "crops").mkdir(parents=True)
    (tmp_path / "output" / "crops" / "a.jpg").write_bytes(b"\xff\xd8fake")
    seed(db, "obj5", "unpriced", {}, crops=["crops/a.jpg"], title="<script>bad</script>")
    page = c.get("/hitl")
    assert '<img src="/img/crops/a.jpg">' in page.text
    assert "<script>bad</script>" not in page.text   # must be HTML-escaped
    img = c.get("/img/crops/a.jpg")
    assert img.status_code == 200
