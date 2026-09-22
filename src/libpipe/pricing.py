"""Step 11 PRICE, routed by session country, deterministic [F8, F9, F11].

Books [F8, F9]: Keepa by ISBN (if the country has an Amazon locale) -> local Google Shopping by ISBN -> HITL.
  list price = MRP / RRP / list / fixed depending on country.
SOURCE DIAGRAM TARGET, not measured by this code: ~95%+ auto-priced
Non-books [F10]: Google Lens by image in country -> asset register -> HITL. replacement price new, marked estimate.
SOURCE DIAGRAM TARGET, not measured by this code: ~50-70% auto-priced, UNVALIDATED in the diagram itself.
Measured once, informally: 11 of 12 non-book objects in one real room got a web-search price (see results/README.md);
not checked against any reference price, and web-search pricing is a substitute for the diagram's Google Lens step,
not the same method.
Stored: list_price, currency (ISO), price_concept, price_source; foreign currency converted with rate + date.
"""
from __future__ import annotations

import csv
import difflib
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import httpx

# Keepa domain ids (Amazon locale) by ISO country; countries absent here have no Keepa coverage.
KEEPA_DOMAIN = {"US": 1, "GB": 2, "DE": 3, "FR": 4, "JP": 5, "CA": 6, "IT": 8, "ES": 9, "IN": 10, "MX": 11, "BR": 12}
CURRENCY = {"US": "USD", "GB": "GBP", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR", "IE": "EUR", "NL": "EUR",
            "JP": "JPY", "CA": "CAD", "IN": "INR", "MX": "MXN", "BR": "BRL", "AU": "AUD", "NZ": "NZD",
            "CH": "CHF", "SG": "SGD", "AE": "AED", "ZA": "ZAR"}
CONCEPT = {"IN": "MRP", "GB": "RRP", "AU": "RRP", "IE": "RRP", "NZ": "RRP", "US": "list", "CA": "list"}


def concept_for(country: str) -> str:
    return CONCEPT.get(country, "fixed")


@dataclass
class Price:
    list_price: float
    currency: str
    price_concept: str
    price_source: str
    estimate: bool = False
    fx_note: str | None = None


class FX:
    """Frankfurter (ECB rates). Rate + date are recorded on every conversion."""

    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=15)

    def convert(self, amount: float, src: str, dst: str) -> tuple[float, str]:
        if src == dst:
            return amount, ""
        r = self.http.get("https://api.frankfurter.app/latest", params={"from": src, "to": dst})
        r.raise_for_status()
        j = r.json()
        rate = j["rates"][dst]
        return round(amount * rate, 2), f"{src}->{dst} @ {rate} on {j['date']}"


class Keepa:
    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=30)

    def list_price(self, isbn: str, country: str) -> tuple[float, str] | None:
        dom = KEEPA_DOMAIN.get(country)
        if dom is None:
            return None
        r = self.http.get("https://api.keepa.com/product", params={
            "key": os.environ["KEEPA_API_KEY"], "domain": dom, "code": isbn})
        r.raise_for_status()
        prods = r.json().get("products") or []
        if not prods:
            return None
        csv_ = (prods[0].get("csv") or [])
        lp = csv_[4] if len(csv_) > 4 and csv_[4] else None      # index 4 = LISTPRICE history [t, v, t, v, ...]
        if not lp or lp[-1] in (-1, None):
            return None
        cur = CURRENCY.get(country, "USD")
        v = lp[-1]
        return (v if cur == "JPY" else v / 100.0), cur         # Keepa stores cents (JPY in whole yen)


class SerpApi:
    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=60)

    def _get(self, **params) -> dict:
        r = self.http.get("https://serpapi.com/search.json", params={**params, "api_key": os.environ["SERPAPI_API_KEY"]})
        r.raise_for_status()
        return r.json()

    def shopping(self, q: str, country: str) -> list[float]:
        j = self._get(engine="google_shopping", q=q, gl=country.lower())
        return [x["extracted_price"] for x in j.get("shopping_results", []) if x.get("extracted_price")]

    def lens(self, image_url: str, country: str) -> list[float]:
        j = self._get(engine="google_lens", url=image_url, country=country.lower())
        return [m["price"]["extracted_value"] for m in j.get("visual_matches", []) if (m.get("price") or {}).get("extracted_value")]


def asset_register_lookup(path: Path | None, description: str, cutoff: float = 0.6) -> tuple[float, str] | None:
    """CSV with columns: description,price,currency."""
    if not path or not Path(path).exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    best = max(rows, key=lambda r: difflib.SequenceMatcher(None, r["description"].lower(), description.lower()).ratio(),
               default=None)
    if best and difflib.SequenceMatcher(None, best["description"].lower(), description.lower()).ratio() >= cutoff:
        return float(best["price"]), best["currency"]
    return None


class WebSearchPricer:
    """Non-book replacement price without image hosting: Astra (OpenAI) with its built-in web search tool looks at
    the crop, identifies the item and returns a new-price estimate in the local currency."""

    def __init__(self, model: str = "gpt-6-astra"):
        from openai import OpenAI

        self.client, self.model = OpenAI(), model

    def price(self, crop_jpeg: bytes | None, description: str, country: str, currency: str) -> float | None:
        import base64
        import json

        from .llm import parse_json

        content = [{"type": "input_text", "text":
                    f"Item on a shelf: {description or 'see image'}. Search the web for what this exact or closest "
                    f"equivalent costs NEW in {country} today, in {currency}. Reply ONLY JSON: "
                    f'{{"price": <number or null>, "matched": "<what you matched>"}}. null if you cannot find one.'}]
        if crop_jpeg:
            content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(crop_jpeg).decode()})
        r = self.client.responses.create(model=self.model, tools=[{"type": "web_search"}],
                                         input=[{"role": "user", "content": content}])
        p = parse_json(r.output_text).get("price")
        return float(p) if isinstance(p, (int, float)) else None


class Pricer:
    def __init__(self, keepa: Keepa | None = None, serp: SerpApi | None = None, fx: FX | None = None,
                 asset_register: Path | None = None, web: WebSearchPricer | None = None,
                 use_keepa: bool | None = None):
        # Keepa is optional (paid plan): skipped unless a key is set or one is injected.
        if keepa is None and (use_keepa if use_keepa is not None else bool(os.environ.get("KEEPA_API_KEY"))):
            keepa = Keepa()
        self.keepa, self.serp, self.fx, self.register, self.web = keepa, serp or SerpApi(), fx or FX(), asset_register, web

    def price_book(self, isbn: str, country: str) -> Price | None:
        local, concept = CURRENCY.get(country, "USD"), concept_for(country)
        hit = self.keepa.list_price(isbn, country) if self.keepa else None
        if hit:
            return Price(hit[0], hit[1], concept, "keepa")
        prices = self.serp.shopping(isbn, country)
        if prices:
            return Price(round(statistics.median(prices), 2), local, concept, "google_shopping")
        return None

    def price_nonbook(self, image_url: str | None, description: str, country: str,
                      crop_jpeg: bytes | None = None) -> Price | None:
        local = CURRENCY.get(country, "USD")
        if image_url:      # Google Lens needs a publicly reachable image URL
            prices = self.serp.lens(image_url, country)
            if prices:
                return Price(round(statistics.median(prices), 2), local, "replacement_new", "google_lens", estimate=True)
        elif self.web is not None:
            p = self.web.price(crop_jpeg, description, country, local)
            if p:
                return Price(round(p, 2), local, "replacement_new", "web_search", estimate=True)
        reg = asset_register_lookup(self.register, description)
        if reg:
            amt, cur = reg
            note = None
            if cur != local:
                amt, note = self.fx.convert(amt, cur, local)
            return Price(amt, local, "replacement_new", "asset_register", estimate=True, fx_note=note)
        return None
