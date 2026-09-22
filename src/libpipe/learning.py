"""Step 13 LEARNING + GOLD SET [F13, F14]: Jev verdicts and HITL fixes become exemplars for future region calls;
a weekly human-labelled gold set measures Jev and pauses learning on drift."""
from __future__ import annotations

import json

from .db import DB
from .llm import JevJudge
from .settle import jev_verdict

PAUSED = "learning_paused"


def paused(db: DB) -> bool:
    r = db.q("SELECT v FROM learning_state WHERE k=?", (PAUSED,))
    return bool(r) and r[0]["v"] == "1"


def add_exemplar(db: DB, source: str, payload: dict) -> None:
    if paused(db):
        return
    db.x("INSERT INTO exemplars(source, payload) VALUES(?,?)", (source, json.dumps(payload, default=str)))


def exemplars(db: DB, limit: int = 6) -> list[dict]:
    """Most recent exemplars, fed into the next region calls (the gold dashed arrow in the diagram)."""
    if paused(db):
        return []
    return [json.loads(r["payload"]) for r in db.q("SELECT payload FROM exemplars ORDER BY id DESC LIMIT ?", (limit,))]


def record_verdict_exemplar(db: DB, obj: dict, decided_by: str) -> None:
    add_exemplar(db, f"jev:{decided_by}", {k: obj[k] for k in ("is_book", "title", "author", "is_old") if k in obj})


def add_gold(db: DB, evidence: str, cand: dict, truth_winner: str) -> None:
    """One human-labelled judged-pair case: which candidate is actually right ('fable' / 'astra')."""
    db.x("INSERT INTO gold(object_id, truth) VALUES(?,?)",
         ("gold", json.dumps({"evidence": evidence, "cand": cand, "winner": truth_winner})))


def run_gold(db: DB, jev: JevJudge, min_agreement: float = 0.85) -> dict:
    """Weekly: how often does Jev pick the human-verified winner? Below the threshold, pause learning."""
    rows = [json.loads(r["truth"]) for r in db.q("SELECT truth FROM gold")]
    if not rows:
        return {"agreement": None, "paused": paused(db), "n": 0}
    hit = 0
    for g in rows:
        v = jev_verdict(jev, g["evidence"], "Which answer better matches the evidence?", g["cand"])
        hit += v.winner == g["winner"]
    agree = hit / len(rows)
    pause = agree < min_agreement
    db.x("INSERT OR REPLACE INTO learning_state(k,v) VALUES(?,?)", (PAUSED, "1" if pause else "0"))
    db.x("INSERT INTO gold_runs(agreement, paused) VALUES(?,?)", (agree, int(pause)))
    return {"agreement": agree, "paused": pause, "n": len(rows)}
