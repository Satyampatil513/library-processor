"""Step 10 BOOK IDENTITY [F7]: ISBN from barcode if visible, else ISBNdb by title + author; unreadable spine -> HITL.

SOURCE DIAGRAM TARGET, not measured by this code: coverage gaps 8-12%; ISBNdb match assumed ~100%.
Never run against a real book yet - unit-tested against mocked ISBNdb only.
"""
from __future__ import annotations

import os
import re
from urllib.parse import quote

import httpx

from .settle import text_match


def isbn13_ok(s: str) -> bool:
    if not re.fullmatch(r"97[89]\d{10}", s):
        return False
    t = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(s[:12]))
    return (10 - t % 10) % 10 == int(s[12])


def isbn10_to_13(s: str) -> str | None:
    if not re.fullmatch(r"\d{9}[\dXx]", s):
        return None
    core = "978" + s[:9]
    chk = (10 - sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(core)) % 10) % 10
    return core + str(chk)


def clean_isbn(raw: str | None) -> str | None:
    if not raw:
        return None
    d = re.sub(r"[^0-9Xx]", "", raw)
    if isbn13_ok(d):
        return d
    return (lambda x: x if x and isbn13_ok(x) else None)(isbn10_to_13(d)) if len(d) == 10 else None


class ISBNdb:
    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=20)

    def _h(self):
        return {"Authorization": os.environ["ISBNDB_API_KEY"]}

    def search(self, title: str, author: str | None) -> list[dict]:
        q = f"{title} {author}" if author else title
        r = self.http.get(f"https://api2.isbndb.com/books/{quote(q, safe='')}",
                          params={"pageSize": 10}, headers=self._h())
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("books", [])

    def resolve(self, obj: dict) -> tuple[str | None, str]:
        """-> (isbn13 | None, provenance)."""
        isbn = clean_isbn(obj.get("barcode"))
        if isbn:
            return isbn, "barcode"
        if not obj.get("title"):
            return None, "unreadable spine"
        for b in self.search(obj["title"], obj.get("author")):
            title_ok = text_match(b.get("title"), obj["title"], 0.8)
            auth_ok = (not obj.get("author")) or any(text_match(a, obj["author"], 0.7) for a in b.get("authors", []) or [""])
            isbn = clean_isbn(b.get("isbn13") or b.get("isbn"))
            if title_ok and auth_ok and isbn:
                return isbn, "isbndb"
        return None, "no isbndb match"
