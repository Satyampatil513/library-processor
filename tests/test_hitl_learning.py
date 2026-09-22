from libpipe import hitl, learning
from libpipe.db import DB
from libpipe.settle import jev_verdict


class ScriptedJev:
    def __init__(self, pick):
        self.pick = pick

    def choose(self, state, instructions, options):
        return {"choice": self.pick(state), "confidence": 0.9, "probabilities": {}}


def add_obj(db, oid="o1"):
    db.upsert("objects", {"id": oid, "session_id": "s", "region_id": "r", "title": "wrong", "status": "needs_review"})


def test_hitl_pick_candidate_updates_object_and_learns(tmp_path):
    db = DB(tmp_path / "t.db")
    add_obj(db)
    cand = {"fable": {"is_book": 1, "title": "Dune", "author": "F. Herbert", "is_old": 0},
            "astra": {"is_book": 1, "title": "Dyne", "author": "X", "is_old": 0}}
    hitl.enqueue(db, "o1", "Jev tie", {"candidates": cand})
    item = hitl.open_items(db)[0]
    hitl.resolve(db, item["id"], {"pick": "fable"})
    row = db.q("SELECT title,status FROM objects WHERE id='o1'")[0]
    assert (row["title"], row["status"]) == ("Dune", "reviewed")
    assert hitl.open_items(db) == []
    assert learning.exemplars(db)[0]["correct"]["title"] == "Dune"


def test_hitl_manual_fix_price(tmp_path):
    db = DB(tmp_path / "t.db")
    add_obj(db)
    hitl.enqueue(db, "o1", "unpriced")
    hitl.resolve(db, hitl.open_items(db)[0]["id"], {"list_price": 12.5, "currency": "USD", "price_source": "manual"})
    assert db.q("SELECT list_price FROM objects WHERE id='o1'")[0]["list_price"] == 12.5


class FakePrice:
    def __init__(self, list_price, currency="USD", price_concept="list", price_source="fake", estimate=False):
        self.list_price, self.currency = list_price, currency
        self.price_concept, self.price_source, self.estimate, self.fx_note = price_concept, price_source, estimate, None


def test_hitl_reprice_on_identity_edit_when_no_manual_price(tmp_path):
    """Correcting is_book (or title/author/isbn) without the reviewer typing a price must re-run step
    10/11, not leave the object silently priced against its old identity - the documented HITL gap."""
    db = DB(tmp_path / "t.db")
    db.upsert("sessions", {"id": "s", "country": "US"})
    db.upsert("objects", {"id": "o1", "session_id": "s", "region_id": "r", "is_book": 0, "status": "needs_review",
                          "description": "a book-shaped thing", "crops": []})
    hitl.enqueue(db, "o1", "identity unclear")

    class FakeISBNdb:
        def resolve(self, obj):
            assert obj["title"] == "Dune"
            return "9780441172719", "isbndb"

    class FakePricer:
        def __init__(self):
            self.book_calls = []

        def price_book(self, isbn, country):
            self.book_calls.append((isbn, country))
            return FakePrice(9.99, "USD", "list", "keepa")

    pricer = FakePricer()
    hitl.resolve(db, hitl.open_items(db)[0]["id"], {"is_book": True, "title": "Dune", "author": "F. Herbert"},
                isbndb=FakeISBNdb(), pricer=pricer)

    row = db.q("SELECT isbn, list_price, currency, price_source, status FROM objects WHERE id='o1'")[0]
    assert row["isbn"] == "9780441172719" and row["list_price"] == 9.99
    assert row["price_source"] == "keepa" and row["status"] == "reviewed"
    assert pricer.book_calls == [("9780441172719", "US")]


def test_hitl_reprice_skipped_when_reviewer_gives_manual_price(tmp_path):
    """A manually-typed price is the reviewer's explicit answer and must not be overwritten by a re-price."""
    db = DB(tmp_path / "t.db")
    db.upsert("sessions", {"id": "s", "country": "US"})
    db.upsert("objects", {"id": "o1", "session_id": "s", "region_id": "r", "is_book": 0, "status": "needs_review"})
    hitl.enqueue(db, "o1", "identity unclear")

    class ExplodingPricer:
        def price_book(self, *a, **kw):
            raise AssertionError("must not be called when a manual price was given")

    hitl.resolve(db, hitl.open_items(db)[0]["id"], {"is_book": True, "title": "Dune", "list_price": 30.0},
                pricer=ExplodingPricer())
    row = db.q("SELECT list_price FROM objects WHERE id='o1'")[0]
    assert row["list_price"] == 30.0


def test_hitl_can_mark_not_an_object(tmp_path):
    """Regression test: Astra/Fable can return is_object:false (bare wall/floor/shadow), but until now HITL
    had no way for a human reviewer to correct something the SAME way by hand."""
    db = DB(tmp_path / "t.db")
    db.upsert("objects", {"id": "o1", "session_id": "s", "region_id": "r", "is_book": 0, "status": "needs_review",
                          "title": "Maybe a book spine?"})
    hitl.enqueue(db, "o1", "low confidence")

    class ExplodingPricer:
        def price_book(self, *a, **kw):
            raise AssertionError("must not price something just marked not_an_object")

        def price_nonbook(self, *a, **kw):
            raise AssertionError("must not price something just marked not_an_object")

    hitl.resolve(db, hitl.open_items(db)[0]["id"], {"is_object": False, "title": "wall smudge"},
                pricer=ExplodingPricer())
    row = db.q("SELECT status FROM objects WHERE id='o1'")[0]
    assert row["status"] == "not_an_object"
    assert hitl.open_items(db) == []


def test_hitl_reprice_failure_does_not_break_resolve(tmp_path):
    """A pricer that errors (missing API key, network failure, etc.) must not stop the reviewer's edit
    from being saved - re-pricing is strictly best-effort."""
    db = DB(tmp_path / "t.db")
    db.upsert("sessions", {"id": "s", "country": "US"})
    db.upsert("objects", {"id": "o1", "session_id": "s", "region_id": "r", "is_book": 0, "status": "needs_review"})
    hitl.enqueue(db, "o1", "identity unclear")

    class BrokenISBNdb:
        def resolve(self, obj):
            raise RuntimeError("ISBNDB_API_KEY not set")

    class UnusedPricer:
        def price_book(self, *a, **kw):
            raise AssertionError("should never be reached: isbn resolution failed first")

    hitl.resolve(db, hitl.open_items(db)[0]["id"], {"is_book": True, "title": "Dune"},
                isbndb=BrokenISBNdb(), pricer=UnusedPricer())
    row = db.q("SELECT title, status FROM objects WHERE id='o1'")[0]
    assert row["title"] == "Dune" and row["status"] == "reviewed"


def test_gold_drift_pauses_learning(tmp_path):
    db = DB(tmp_path / "t.db")
    cand = {"fable": {"t": "right"}, "astra": {"t": "wrong"}}
    for _ in range(4):
        learning.add_gold(db, "ev", cand, "fable")
    good = ScriptedJev(lambda s: "A" if s["answer_A"]["t"] == "right" else "B")
    r = learning.run_gold(db, good, 0.85)
    assert r["agreement"] == 1.0 and not r["paused"]
    learning.add_exemplar(db, "x", {"a": 1})
    assert learning.exemplars(db) == [{"a": 1}]

    bad = ScriptedJev(lambda s: "A" if s["answer_A"]["t"] == "wrong" else "B")   # Jev now systematically wrong
    r = learning.run_gold(db, bad, 0.85)
    assert r["agreement"] == 0.0 and r["paused"]
    assert learning.exemplars(db) == []           # paused: nothing fed back
    learning.add_exemplar(db, "x", {"a": 2})
    assert db.q("SELECT COUNT(*) c FROM exemplars")[0]["c"] == 1   # nothing new stored either
    r = learning.run_gold(db, good, 0.85)
    assert not r["paused"]                         # recovers when Jev agrees with humans again
