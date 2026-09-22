import httpx
import pytest

from libpipe.identity import ISBNdb, clean_isbn, isbn10_to_13, isbn13_ok
from libpipe.pricing import FX, Keepa, Pricer, SerpApi, asset_register_lookup, concept_for


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for k in ("ISBNDB_API_KEY", "KEEPA_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.setenv(k, "x")


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_isbn_checksums():
    assert isbn13_ok("9780306406157")
    assert not isbn13_ok("9780306406158")
    assert isbn10_to_13("0306406152") == "9780306406157"
    assert clean_isbn("978-0-306-40615-7") == "9780306406157"
    assert clean_isbn("0-306-40615-2") == "9780306406157"
    assert clean_isbn("12345") is None and clean_isbn(None) is None


def test_barcode_isbn_skips_isbndb():
    def boom(req):
        raise AssertionError("no network when barcode has the ISBN")
    assert ISBNdb(client(boom)).resolve({"barcode": "9780306406157", "title": "x"}) == ("9780306406157", "barcode")


def test_isbndb_title_author_match():
    def h(req):
        return httpx.Response(200, json={"books": [
            {"title": "Other Book", "authors": ["Z"], "isbn13": "9780306406157"},
            {"title": "Dune", "authors": ["Frank Herbert"], "isbn13": "9780441172719"}]})
    assert ISBNdb(client(h)).resolve({"title": "Dune", "author": "Frank Herbert"}) == ("9780441172719", "isbndb")


def test_unreadable_spine_and_no_match():
    assert ISBNdb(client(lambda r: httpx.Response(404))).resolve({"title": None}) == (None, "unreadable spine")
    assert ISBNdb(client(lambda r: httpx.Response(404))).resolve({"title": "Zzz"}) == (None, "no isbndb match")


def test_concepts_by_country():
    assert (concept_for("IN"), concept_for("GB"), concept_for("US"), concept_for("SE")) == ("MRP", "RRP", "list", "fixed")


def test_keepa_list_price_cents_and_yen():
    def h(req):
        return httpx.Response(200, json={"products": [{"csv": [None, None, None, None, [1, 1999, 2, 2499]]}]})
    k = Keepa(client(h))
    assert k.list_price("9780441172719", "US") == (24.99, "USD")
    assert k.list_price("9780441172719", "JP") == (2499, "JPY")
    assert k.list_price("9780441172719", "SE") is None     # no Amazon locale


def test_keepa_no_list_price():
    h = lambda r: httpx.Response(200, json={"products": [{"csv": [None] * 4 + [[1, -1]]}]})
    assert Keepa(client(h)).list_price("x", "US") is None


def test_book_price_falls_back_to_shopping():
    def h(req):
        assert req.url.host == "serpapi.com"
        return httpx.Response(200, json={"shopping_results": [{"extracted_price": 10}, {"extracted_price": 30}, {"extracted_price": 20}]})
    p = Pricer(keepa=Keepa(client(lambda r: httpx.Response(200, json={"products": []}))), serp=SerpApi(client(h)))
    pr = p.price_book("9780441172719", "SE")
    # SE is not in the currency map yet, so it defaults to USD - add it before using Sweden for real
    assert (pr.list_price, pr.currency, pr.price_source, pr.price_concept) == (20, "USD", "google_shopping", "fixed")


def test_book_unpriced_returns_none():
    p = Pricer(keepa=Keepa(client(lambda r: httpx.Response(200, json={"products": []}))),
               serp=SerpApi(client(lambda r: httpx.Response(200, json={}))))
    assert p.price_book("9780441172719", "IN") is None


def test_india_keepa_uses_mrp():
    h = lambda r: httpx.Response(200, json={"products": [{"csv": [None] * 4 + [[1, 39900]]}]})
    pr = Pricer(keepa=Keepa(client(h))).price_book("9780441172719", "IN")
    assert (pr.list_price, pr.currency, pr.price_concept, pr.price_source) == (399.0, "INR", "MRP", "keepa")


def test_nonbook_lens_marked_estimate():
    h = lambda r: httpx.Response(200, json={"visual_matches": [{"price": {"extracted_value": 50}}, {"title": "no price"},
                                                              {"price": {"extracted_value": 70}}]})
    pr = Pricer(serp=SerpApi(client(h))).price_nonbook("https://x/y.jpg", "lamp", "US")
    assert pr.list_price == 60 and pr.estimate and pr.price_source == "google_lens"


def test_nonbook_register_fallback_with_fx(tmp_path):
    reg = tmp_path / "r.csv"
    reg.write_text("description,price,currency\nblue ceramic table lamp,40,USD\n")
    fx = FX(client(lambda r: httpx.Response(200, json={"date": "2026-09-19", "rates": {"INR": 84.0}})))
    pr = Pricer(serp=SerpApi(client(lambda r: httpx.Response(200, json={}))), fx=fx, asset_register=reg) \
        .price_nonbook(None, "blue ceramic table lamp", "IN")
    assert pr.list_price == 3360.0 and pr.currency == "INR" and "USD->INR @ 84.0 on 2026-09-19" in pr.fx_note


def test_register_no_match(tmp_path):
    reg = tmp_path / "r.csv"
    reg.write_text("description,price,currency\nblue ceramic table lamp,40,USD\n")
    assert asset_register_lookup(reg, "antique globe") is None
