import json
import random

from libpipe.db import DB
from libpipe.settle import objects_agree, settle_region, shift_equivalent, text_match

ORDER = ["p1", "p2", "p3", "p4"]


def obj(ids, title="Dune", author="Frank Herbert", is_book=True, is_old=False, reasoning=""):
    return {"piece_ids": ids, "is_book": is_book, "title": title, "author": author, "is_old": is_old,
            "description": None, "reasoning": reasoning}


class FakeJev:
    """Scripted judge. `pick(state)` returns 'A'|'B'|'tie'; confidence fixed."""

    def __init__(self, pick, conf=0.9):
        self.pick, self.conf, self.calls = pick, conf, []

    def choose(self, state, instructions, options):
        self.calls.append(state)
        return {"choice": self.pick(state), "confidence": self.conf, "probabilities": {"A": .5, "B": .5}}


def db(tmp_path):
    return DB(tmp_path / "t.db")


def no_reshare(_t, _tag):
    raise AssertionError("re-share must not run when models agree")


def test_text_match_fuzzy():
    assert text_match("The Hobbit", "the hobbit!")
    assert text_match(None, "")
    assert not text_match("Dune", "Emma")
    assert not text_match("Dune", None)


def test_agree_no_jev_call(tmp_path):
    out = {"objects": [obj(["p1", "p2"]), obj(["p3", "p4"], "Emma", "Austen")]}
    j = FakeJev(lambda s: "A")
    r = settle_region(db(tmp_path), "r0", ORDER, out, out, "ev", j, no_reshare, sample_rate=0)
    assert len(r.objects) == 2 and r.stats["agree"] == 2 and not j.calls and not r.hitl


def test_sample_rate_sends_agreements_to_jev(tmp_path):
    out = {"objects": [obj(ORDER)]}
    j = FakeJev(lambda s: "correct")
    r = settle_region(db(tmp_path), "r0", ORDER, out, out, "ev", j, no_reshare, sample_rate=1.0)
    assert r.stats["sampled"] == 1 and len(r.objects) == 1


def test_sample_flag_goes_to_hitl(tmp_path):
    out = {"objects": [obj(ORDER)]}
    r = settle_region(db(tmp_path), "r0", ORDER, out, out, "ev", FakeJev(lambda s: "incorrect"), no_reshare, sample_rate=1.0)
    assert r.objects == [] and r.hitl[0][0] == "sample flagged by Jev"


def test_shift_of_one_is_auto_fixed(tmp_path):
    f = {"objects": [obj(["p1", "p2"]), obj(["p3", "p4"], "Emma", "Austen")]}
    a = {"objects": [obj(["p1"]), obj(["p2", "p3", "p4"], "Emma", "Austen")]}
    assert shift_equivalent(f["objects"], a["objects"], ORDER)
    j = FakeJev(lambda s: "A")
    r = settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", j, no_reshare, sample_rate=0)
    assert not j.calls and r.stats.get("shift_fixed")
    assert [o["piece_ids"] for o in r.objects] == [["p1", "p2"], ["p3", "p4"]]   # Fable boundaries kept


def test_shift_of_two_is_not_a_shift():
    f = [obj(["p1", "p2", "p3"]), obj(["p4"])]
    a = [obj(["p1"]), obj(["p2", "p3", "p4"])]
    assert not shift_equivalent(f, a, ORDER)


def test_field_dispute_jev_picks_winner_position_independent(tmp_path):
    """Jev always picks whichever answer says 'Dune' regardless of slot -> consistent -> decisive."""
    f = {"objects": [obj(ORDER, "Dune")]}
    a = {"objects": [obj(ORDER, "Dyne")]}
    pick = lambda s: "A" if s["answer_A"]["title"] == "Dune" else "B"
    j = FakeJev(pick)
    r = settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", j, lambda t, tag: (f, a), sample_rate=0)
    assert r.objects[0]["title"] == "Dune" and r.objects[0]["_by"] == "jev"
    assert not r.hitl
    assert len(j.calls) == 4   # first verdict (2 swapped runs) + final verdict (2 swapped runs)


def test_position_bias_flip_is_a_tie_and_goes_to_hitl(tmp_path):
    f = {"objects": [obj(ORDER, "Dune")]}
    a = {"objects": [obj(ORDER, "Dyne")]}
    j = FakeJev(lambda s: "A")   # always slot A -> flips when order swapped
    r = settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", j, lambda t, tag: (f, a), sample_rate=0)
    assert r.objects == [] and r.hitl and r.hitl[0][0] == "Jev tie / low confidence"
    assert r.hitl[0][1]["object"]["title"] == "Dune"      # provisional = Fable


def test_low_confidence_goes_to_hitl(tmp_path):
    f = {"objects": [obj(ORDER, "Dune")]}
    a = {"objects": [obj(ORDER, "Dyne")]}
    pick = lambda s: "A" if s["answer_A"]["title"] == "Dune" else "B"
    r = settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", FakeJev(pick, conf=0.3), lambda t, tag: (f, a),
                      sample_rate=0, min_conf=0.6)
    assert r.hitl and not r.objects


def test_reshare_convergence_skips_final_jev(tmp_path):
    f = {"objects": [obj(ORDER, "Dune")]}
    a = {"objects": [obj(ORDER, "Dyne")]}
    j = FakeJev(lambda s: "A" if s["answer_A"]["title"] == "Dune" else "B")
    seen = {}

    def reshare(text, tag):
        seen["text"], seen["tag"] = text, tag
        return f, f   # both models now agree

    r = settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", j, reshare, sample_rate=0)
    assert seen["tag"] == "_crosscheck" and "title" in seen["text"]
    assert len(j.calls) == 2 and len(r.objects) == 1


def test_verdicts_are_logged(tmp_path):
    d = db(tmp_path)
    f = {"objects": [obj(ORDER, "Dune")]}
    a = {"objects": [obj(ORDER, "Dyne")]}
    settle_region(d, "r0", ORDER, f, a, "ev", FakeJev(lambda s: "A"), lambda t, tag: (f, a), sample_rate=0)
    assert {r["stage"] for r in d.q("SELECT stage FROM verdicts")} == {"first", "final"}


def test_jev_is_blind_to_model_names(tmp_path):
    f = {"objects": [obj(ORDER, "Dune")]}
    a = {"objects": [obj(ORDER, "Dyne")]}
    j = FakeJev(lambda s: "tie")
    settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", j, lambda t, tag: (f, a), sample_rate=0)
    blob = str(j.calls).lower()
    assert "fable" not in blob and "astra" not in blob


def test_partition_dispute_routes_through_jev(tmp_path):
    f = {"objects": [obj(["p1", "p2", "p3", "p4"])]}
    a = {"objects": [obj(["p1"]), obj(["p2"], "B", "C"), obj(["p3"], "D", "E"), obj(["p4"], "F", "G")]}
    j = FakeJev(lambda s: "A" if len(s["answer_A"]) == 1 else "B")     # prefers the 1-object grouping
    r = settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", j, lambda t, tag: (f, a), sample_rate=0)
    assert len(r.objects) == 1 and r.objects[0]["piece_ids"] == ORDER


def test_missing_piece_becomes_singleton_object(tmp_path):
    out = {"objects": [obj(["p1", "p2", "p3"])]}      # model forgot p4
    r = settle_region(db(tmp_path), "r0", ORDER, out, out, "ev", FakeJev(lambda s: "A"), no_reshare, sample_rate=0)
    assert sorted(i for o in r.objects for i in o["piece_ids"]) == ORDER


def test_non_book_ignores_title_differences():
    a = obj(["p1"], is_book=False, title=None, author=None)
    b = obj(["p1"], is_book=False, title="whatever", author="x")
    assert objects_agree(a, b) == []


def test_is_object_disagreement_is_a_dispute():
    a, b = obj(["p1"], is_book=False), obj(["p1"], is_book=False)
    b["is_object"] = False
    assert objects_agree(a, b) == ["is_object"]


def test_jev_never_sees_model_reasoning_text(tmp_path):
    """A verdict must turn on values, not on which model wrote a more persuasive rationale - and the
    order-swap flip-to-tie check only catches POSITION bias, not a bias tied to reasoning style, since the
    same reasoning text would stay attached to the same content in both swapped runs."""
    f = {"objects": [obj(ORDER, "Dune", reasoning="Extremely confident: the spine clearly reads 'Dune' in bold "
                        "gold lettering, cross-checked against the cover art and the author's signature style.")]}
    a = {"objects": [obj(ORDER, "Dyne", reasoning="idk")]}
    seen = []

    class RecordingJev(FakeJev):
        def choose(self, state, instructions, options):
            seen.append(state)
            return super().choose(state, instructions, options)

    pick = lambda s: "A" if s["answer_A"]["title"] == "Dune" else "B"
    settle_region(db(tmp_path), "r0", ORDER, f, a, "ev", RecordingJev(pick), lambda t, tag: (f, a), sample_rate=0)
    assert seen, "Jev should have been called for this dispute"
    for state in seen:
        for side in ("answer_A", "answer_B"):
            assert "reasoning" not in state[side], f"Jev was shown model reasoning text: {state[side]}"
            assert "confident" not in json.dumps(state).lower() and "idk" not in json.dumps(state).lower()
