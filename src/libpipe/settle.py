"""Step 9 COMPARE AND SETTLE [F12, F13] - the JUDGED-PAIR box.

1. Compare, no model: join on piece id, canonical partitions, +-1 boundary shift auto-fixed.
2. agree -> done (5% sampled to Jev [F12]);  disagree -> Jev: blind, 2x order-swapped, flip = tie, logged [F13].
3. Re-share disputed items + Jev's assessment to both models, one round, tagged `_crosscheck`.
4. Jev final verdict; still tie / low confidence -> HITL.

Jev returns typed choices + probabilities (no prose, no images), so its "reasoning" is its probability
distribution over the two candidate answers, judged from the models' text answers + text evidence.

Jev is only ever shown the disputed *values* (title/author/is_book/is_old/description/piece_ids), never
either model's free-text `reasoning` - see `_values_only`. A verdict must turn on which value the shared
evidence supports, not on which model wrote a more persuasive rationale. This matters because the
order-swap check (a flip = tie) only cancels *position* bias (always preferring slot A); it does nothing
for a bias tied to a model's writing voice, since that voice would stay attached to the same content in
both swapped runs.
"""
from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable

from .db import DB
from .llm import JevJudge, RegionModel
from .regions import normalise

FIELDS = ("is_book", "title", "author", "is_old")


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def text_match(a: str | None, b: str | None, thr: float = 0.9) -> bool:
    a, b = _norm(a), _norm(b)
    if not a and not b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= thr


def objects_agree(a: dict, b: dict) -> list[str]:
    """Return the list of disagreeing fields ([] = agree)."""
    bad = []
    if a.get("is_object", True) != b.get("is_object", True):
        return ["is_object"]
    if a["is_book"] != b["is_book"]:
        bad.append("is_book")
    if a["is_old"] != b["is_old"]:
        bad.append("is_old")
    if a["is_book"] and b["is_book"]:
        if not text_match(a["title"], b["title"]):
            bad.append("title")
        if not text_match(a["author"], b["author"]):
            bad.append("author")
    return bad


def canonical(objs: list[dict]) -> frozenset:
    return frozenset(frozenset(o["piece_ids"]) for o in objs)


def _cuts(objs: list[dict], order: list[str]) -> set[int] | None:
    """Boundary positions in the shelf-ordered piece list; None if an object is non-contiguous."""
    pos = {p: i for i, p in enumerate(order)}
    cuts = set()
    for o in objs:
        idx = sorted(pos[p] for p in o["piece_ids"])
        if idx != list(range(idx[0], idx[-1] + 1)):
            return None
        cuts.add(idx[-1] + 1)
    cuts.discard(len(order))
    return cuts


def shift_equivalent(a: list[dict], b: list[dict], order: list[str], tol: int = 1) -> bool:
    ca, cb = _cuts(a, order), _cuts(b, order)
    if ca is None or cb is None or len(ca) != len(cb):
        return False
    return all(abs(x - y) <= tol for x, y in zip(sorted(ca), sorted(cb)))


VALUE_FIELDS = ("piece_ids", "is_object", "is_book", "title", "author", "is_old", "description")


def _values_only(o: dict) -> dict:
    """Strip everything except the typed fields Jev should judge - in particular `reasoning`, so a verdict
    can't be swayed by how persuasively one model argued rather than by which value the evidence supports."""
    return {k: o[k] for k in VALUE_FIELDS if k in o}


def _values_only_list(objs: list[dict]) -> list[dict]:
    return [_values_only(o) for o in objs]


def match_objects(a: list[dict], b: list[dict]) -> list[tuple[dict, dict | None]]:
    """Pair each object in a with the b object sharing the most pieces."""
    out = []
    for oa in a:
        sa = set(oa["piece_ids"])
        best = max(b, key=lambda ob: len(sa & set(ob["piece_ids"])), default=None)
        out.append((oa, best if best and sa & set(best["piece_ids"]) else None))
    return out


# --------------------------------------------------------------------- Jev ---

@dataclass
class Verdict:
    winner: str            # "fable" | "astra" | "tie"
    confidence: float
    reasoning: str


def jev_verdict(jev: JevJudge, evidence: str, question: str, cand: dict[str, dict]) -> Verdict:
    """Blind (labels A/B, never model names), run twice with the order swapped. A flip = position bias = tie."""
    names = list(cand)  # ["fable", "astra"]
    runs = []
    for first, second in ((names[0], names[1]), (names[1], names[0])):
        r = jev.choose(
            state={"evidence": evidence[:20000], "answer_A": cand[first], "answer_B": cand[second]},
            instructions=question,
            options={"A": "Answer A is the more likely correct reading of the evidence",
                     "B": "Answer B is the more likely correct reading of the evidence",
                     "tie": "Cannot tell, or both equally plausible"})
        pick = {"A": first, "B": second}.get(r["choice"], "tie")
        runs.append((pick, r))
    (p1, r1), (p2, r2) = runs
    conf = min(r1["confidence"], r2["confidence"])
    reasoning = f"run1 {r1['choice']} {json.dumps(r1['probabilities'])}; run2(swapped) {r2['choice']} {json.dumps(r2['probabilities'])}"
    if p1 != p2:
        return Verdict("tie", conf, "order flip: " + reasoning)
    return Verdict(p1, conf, reasoning)


# ------------------------------------------------------------------ settle ---

@dataclass
class Settled:
    objects: list[dict]                       # final objects for the region
    hitl: list[tuple[str, dict]] = field(default_factory=list)   # (reason, payload)
    stats: dict = field(default_factory=dict)


def settle_region(db: DB, region_id: str, order: list[str], fable_out: dict, astra_out: dict, evidence: str,
                  jev: JevJudge, reshare: Callable[[str, str], tuple[dict, dict]],
                  sample_rate: float = 0.05, min_conf: float = 0.6, rng: random.Random | None = None) -> Settled:
    """`reshare(disputes_text, tag)` -> (fable_out, astra_out) re-answers; supplied by the orchestrator so
    this function stays pure of image handling."""
    rng = rng or random.Random()
    F, A = normalise(fable_out, order), normalise(astra_out, order)
    res = Settled([], [], {"agree": 0, "jev": 0, "hitl": 0, "sampled": 0})

    def log(obj_id, field_, v: Verdict, stage):
        db.x("INSERT INTO verdicts(region_id,object_id,field,winner,confidence,reasoning,stage) VALUES(?,?,?,?,?,?,?)",
             (region_id, obj_id, field_, v.winner, v.confidence, v.reasoning, stage))

    part_ok = canonical(F) == canonical(A)
    if not part_ok and shift_equivalent(F, A, order):
        part_ok, A = True, _align_to(F, A)      # +-1 shift auto-fixed: keep Fable's boundaries
        res.stats["shift_fixed"] = True

    disputes: list[dict] = []
    if not part_ok:
        disputes.append({"kind": "partition", "cand": {"fable": _values_only_list(F), "astra": _values_only_list(A)}, "ids": order})
    else:
        for of, oa in match_objects(F, A):
            bad = objects_agree(of, oa) if oa else ["partition"]
            if not bad:
                res.objects.append({**of, "_by": "models_agree"})
                res.stats["agree"] += 1
                continue
            disputes.append({"kind": "object", "fields": bad,
                             "cand": {"fable": _values_only(of), "astra": _values_only(oa)}, "ids": of["piece_ids"]})
    # 5% of agreements are sampled to Jev as a quality check [F12]
    for o in list(res.objects):
        if rng.random() < sample_rate:
            res.stats["sampled"] += 1
            v = jev.choose({"evidence": evidence[:20000], "answer": _values_only(o)}, "Is this answer consistent with the evidence?",
                           {"correct": "Consistent or not contradicted", "incorrect": "Contradicted by the evidence"})
            db.x("INSERT INTO verdicts(region_id,object_id,field,winner,confidence,reasoning,stage) VALUES(?,?,?,?,?,?,?)",
                 (region_id, ",".join(o["piece_ids"]), "sample", v["choice"], v["confidence"],
                  json.dumps(v["probabilities"]), "sample"))
            if v["choice"] == "incorrect" and v["confidence"] >= min_conf:
                res.objects.remove(o)
                res.stats["agree"] -= 1
                res.hitl.append(("sample flagged by Jev", {"object": o}))

    if not disputes:
        return res

    # --- Jev first verdict, then one re-share round, then Jev final --------------------------------
    first = [(d, jev_verdict(jev, evidence, _question(d), d["cand"])) for d in disputes]
    for d, v in first:
        log(",".join(d["ids"]), d["kind"], v, "first")
    text = "\n".join(f"- {d['kind']} {d['ids']} fields={d.get('fields')}: Jev leaned {v.winner} ({v.confidence:.2f}); {v.reasoning}"
                     for d, v in first)
    f2, a2 = reshare(text, "_crosscheck")
    F2, A2 = normalise(f2, order), normalise(a2, order)

    for d, v1 in first:
        ids = set(d["ids"])
        rf = [o for o in F2 if set(o["piece_ids"]) & ids] or [o for o in F if set(o["piece_ids"]) & ids]
        ra = [o for o in A2 if set(o["piece_ids"]) & ids] or [o for o in A if set(o["piece_ids"]) & ids]
        if d["kind"] == "partition":
            cand = {"fable": _values_only_list(rf), "astra": _values_only_list(ra)}
        else:
            cand = {"fable": _values_only(rf[0]), "astra": _values_only(ra[0])}
        same = (canonical(rf) == canonical(ra)) if d["kind"] == "partition" else (not objects_agree(rf[0], ra[0]))
        v = Verdict("fable", 1.0, "models converged after re-share") if same else jev_verdict(jev, evidence, _question(d), cand)
        log(",".join(d["ids"]), d["kind"], v, "final")
        winner_objs = cand[v.winner if v.winner != "tie" else "fable"]
        winner_objs = winner_objs if isinstance(winner_objs, list) else [winner_objs]
        if v.winner == "tie" or v.confidence < min_conf:
            res.stats["hitl"] += len(winner_objs)
            for o in winner_objs:   # provisional = Fable's reading, flagged
                res.hitl.append(("Jev tie / low confidence", {"object": o, "candidates": cand, "verdict": v.__dict__}))
        else:
            res.stats["jev"] += len(winner_objs)
            res.objects.extend({**o, "_by": "jev"} for o in winner_objs)
    return res


def _question(d: dict) -> str:
    if d["kind"] == "partition":
        return "Which grouping of pieces into physical objects, with their identities, better matches the evidence?"
    return f"Which answer better matches the evidence for the disputed field(s) {d['fields']}?"


def _align_to(F: list[dict], A: list[dict]) -> list[dict]:
    """After a +-1 shift, adopt Fable's boundaries but keep Astra's fields where available."""
    out = []
    for of, oa in match_objects(F, A):
        out.append({**(oa or of), "piece_ids": of["piece_ids"]})
    return out
