"""Runs steps 3-14 for one uploaded session. Every external dependency is injected via `Deps` so the whole
pipeline can run offline against mocks (see tests/)."""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import floorplan, hitl, learning, output, prep, stitch
from .config import Config
from .db import DB
from .identity import ISBNdb
from .llm import JevJudge, RegionModel
from .artifacts import Recorder, draw_piece_boxes, floorplan_plot, isometric_plot, mask_overlay, objects_plot
from .pieces import Piece, Segmenter, crop_for_piece, decode_barcodes, find_pieces
from .pricing import Pricer
from .regions import SYSTEM, Region, best_group_still, build_inputs, make_regions
from .session import Session
from .settle import settle_region


@dataclass
class Deps:
    fable: RegionModel
    astra: RegionModel
    jev: JevJudge
    segmenter: Segmenter | list[Segmenter]
    transcribe: Callable[[str], list[dict]]
    isbndb: ISBNdb
    pricer: Pricer
    image_base_url: str | None = os.environ.get("IMAGE_BASE_URL")   # public URL prefix for saved crops (Google Lens)


def _union_piece(pieces: list[Piece]) -> Piece:
    return Piece("u", np.min([p.box_min for p in pieces], axis=0), np.max([p.box_max for p in pieces], axis=0),
                 sum(p.n_pts for p in pieces))


def run_region(cfg: Config, db: DB, s: Session, region: Region, deps: Deps, rng: random.Random, rec: Recorder | None = None):
    prompt, images = build_inputs(db, s, region, learning.exemplars(db), crop_rotate_deg=cfg.crop_rotate_deg)
    if rec:
        rec.text(f"regions/{region.id}/prompt.txt", prompt)
        for i, im in enumerate(images):
            rec.bytes(f"regions/{region.id}/input_image_{i + 1}.jpg", im)
    with ThreadPoolExecutor(2) as ex:      # independent: neither model sees the other
        ff, fa = ex.submit(deps.fable.call, SYSTEM, prompt, images), ex.submit(deps.astra.call, SYSTEM, prompt, images)
        fable_out, astra_out = ff.result(), fa.result()
    for name, o in (("fable", fable_out), ("astra", astra_out)):
        db.upsert("model_calls", {"region_id": region.id, "model": name, "output": o, "tag": "initial"})
        if rec:
            rec.json(f"regions/{region.id}/{name}_initial.json", o)

    def reshare(disputes: str, tag: str):
        p2 = prompt + "\n\nDISPUTED ITEMS after independent answers (a judge's assessment follows). Reconsider the images " \
                      "and return your full JSON again, changing only what the evidence supports:\n" + disputes
        with ThreadPoolExecutor(2) as ex:
            a, b = ex.submit(deps.fable.call, SYSTEM, p2, images), ex.submit(deps.astra.call, SYSTEM, p2, images)
            fo, ao = a.result(), b.result()
        for name, o in (("fable", fo), ("astra", ao)):
            db.upsert("model_calls", {"region_id": region.id, "model": name, "output": o, "tag": tag})
            if rec:
                rec.json(f"regions/{region.id}/{name}{tag}.json", o)
        return fo, ao

    return settle_region(db, region.id, region.piece_ids, fable_out, astra_out, prompt, deps.jev, reshare,
                         cfg.agree_sample_rate, cfg.jev_min_confidence, rng)


def _store_object(cfg: Config, db: DB, s: Session, region: Region, o: dict, status: str, decided: str,
                  by_id: dict[str, Piece], deps: Deps, hitl_reason: str | None, hitl_payload: dict | None):
    pcs = [by_id[i] for i in o["piece_ids"]]
    u = _union_piece(pcs)
    oid = f"{s.id}-{o['piece_ids'][0]}"
    crops = []
    (cfg.output_dir / "crops").mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(crop_for_piece(s, u, 2, rotate_deg=cfg.crop_rotate_deg)):
        rel = f"crops/{oid}_{k}.jpg"
        c.save(cfg.output_dir / rel, quality=90)
        crops.append(rel)
    barcode = next((p.barcode for p in pcs if p.barcode), None)
    row = {"id": oid, "session_id": s.id, "region_id": region.id,
           "box": {"min": u.box_min.tolist(), "max": u.box_max.tolist()}, "pieces": o["piece_ids"],
           "overlaps": sorted({x for p in pcs for x in p.overlaps} - set(o["piece_ids"])), "barcode": barcode,
           "crops": crops, "is_book": int(o["is_book"]), "title": o["title"], "author": o["author"],
           "is_old": int(o["is_old"]), "description": o["description"], "status": status,
           "provenance": {"decided_by": decided, "session": s.id, "country": s.country, "region": region.id,
                          "frames": sorted({f for p in pcs for f in p.frames})}}
    # step 10 identity, step 11 price
    if not o.get("is_object", True):        # bare wall/floor/shadow fragment: keep for the record, never price
        row["status"] = "not_an_object"
        db.upsert("objects", row)
        return
    if o["is_book"]:
        isbn, how = deps.isbndb.resolve({**o, "barcode": barcode})
        row["isbn"] = isbn
        row["provenance"]["isbn_from"] = how
        if isbn:
            pr = deps.pricer.price_book(isbn, s.country)
        else:
            pr = None
            hitl.enqueue(db, oid, f"book identity: {how}", {})
            row["status"] = "needs_review"
    else:
        url = f"{deps.image_base_url.rstrip('/')}/{crops[0]}" if deps.image_base_url and crops else None
        crop_bytes = (cfg.output_dir / crops[0]).read_bytes() if crops else None
        pr = deps.pricer.price_nonbook(url, o["description"] or "", s.country, crop_bytes)
    if pr:
        row.update({"list_price": pr.list_price, "currency": pr.currency, "price_concept": pr.price_concept,
                    "price_source": pr.price_source, "price_note": ("estimate; " if pr.estimate else "") + (pr.fx_note or "")})
    else:
        hitl.enqueue(db, oid, "unpriced", {})
        row["status"] = "needs_review"
    if hitl_reason:
        hitl.enqueue(db, oid, hitl_reason, hitl_payload)
        row["status"] = "needs_review"
    db.upsert("objects", row)


def _pending_regions(regions: list[Region], done_ids: set[str]) -> list[Region]:
    """Regions not yet in `done_ids` (already-priced regions from a prior, interrupted run of the same
    session) - keeps `process_session(resume=True)` from re-paying for Fable/Astra/Jev calls a killed run
    already made and persisted to the DB."""
    return [r for r in regions if r.id not in done_ids]


def process_session(cfg: Config, db: DB, session_dir: Path, deps: Deps, seed: int | None = None,
                    rec: Recorder | None = None, every: int = 5, max_regions: int | None = None,
                    resume: bool = False) -> dict:
    rng = random.Random(seed)
    s = Session.load(session_dir)
    db.upsert("sessions", {"id": s.id, "path": str(session_dir), "country": s.country, "status": "processing"})
    report: dict = {"session": s.id, "steps": {}}

    prep.align(db, s)                                                           # 3
    prep.transcribe_session(db, s, deps.transcribe)
    report["steps"]["prep"] = {"frames": len(s.frames), "words": db.q("SELECT COUNT(*) c FROM words WHERE session_id=?", (s.id,))[0]["c"]}

    if s.relocalized_against:                                                   # 4
        prev_row = db.q("SELECT path FROM sessions WHERE id=?", (s.relocalized_against,))
        if prev_row:
            r = stitch.stitch(Session.load(prev_row[0]["path"]), s, cfg.stitch_tolerance_m)
            report["steps"]["stitch"] = {"ok": r.ok, "median_err_m": r.median_err_m, "reason": r.reason}
            if not r.ok:
                db.upsert("sessions", {"id": s.id, "path": str(session_dir), "country": s.country, "status": "needs_rerecord"})
                report["status"] = "needs_rerecord"
                return report
        else:
            report["steps"]["stitch"] = {"ok": None, "reason": "previous session not uploaded yet"}

    fp = floorplan.floor_plan(s)                                                # 5
    db.upsert("floorplans", {"session_id": s.id, "data": fp})
    if rec:
        rec.json("floorplan/floorplan.json", fp)
        rec.image("floorplan/topdown.png", floorplan_plot(s, fp))

    def on_frame(fr, img, masks):
        if rec and masks:
            rec.image(f"segmentation/frame_{fr['index']:06d}.jpg", mask_overlay(img, masks))

    pieces = find_pieces(s, deps.segmenter, every=every, merge_iou=cfg.merge_iou, on_frame=on_frame,
                        max_frame_frac=cfg.piece_max_frame_frac, max_dim_m=cfg.piece_max_dim_m)   # 6
    decode_barcodes(s, pieces)
    all_regions = make_regions(pieces, cfg.region_size)                         # 7
    regions = all_regions[:max_regions] if max_regions else all_regions
    if rec:
        rec.json("pieces/pieces.json", [{"id": p.id, "min": p.box_min.tolist(), "max": p.box_max.tolist(),
                                         "size_cm": (p.size * 100).round(1).tolist(), "frames": p.frames,
                                         "barcode": p.barcode, "overlaps": p.overlaps} for p in pieces])
        rec.json("regions/regions.json", {r.id: r.piece_ids for r in all_regions})
        for r in all_regions:   # 3D boxes projected on the still that actually shows the region well (see best_group_still)
            best = best_group_still(s, r.pieces)
            if best:
                rec.image(f"pieces/{r.id}_boxes.jpg", draw_piece_boxes(s.still(best), best, r.pieces))
    by_id = {p.id: p for p in pieces}
    report["steps"]["pieces"] = {"pieces": len(pieces), "regions_found": len(all_regions),
                                 "regions_processed": len(regions), "barcodes": sum(bool(p.barcode) for p in pieces)}
    if len(regions) < len(all_regions):
        report["steps"]["pieces"]["skipped_regions"] = [r.id for r in all_regions[len(regions):]]

    agg = {"agree": 0, "jev": 0, "hitl": 0, "sampled": 0}
    to_process = regions
    if resume:
        done_ids = {r["region_id"] for r in db.q("SELECT DISTINCT region_id FROM objects WHERE session_id=?", (s.id,))}
        to_process = _pending_regions(regions, done_ids)
        report["steps"]["pieces"]["resumed_skipped"] = len(regions) - len(to_process)
    for region in to_process:                                                   # 8, 9, 10, 11, 12
        res = run_region(cfg, db, s, region, deps, rng, rec)
        for k in agg:
            agg[k] += res.stats.get(k, 0)
        for o in res.objects:
            decided = o.get("_by", "jev")
            _store_object(cfg, db, s, region, o, "ok", decided, by_id, deps, None, None)
            learning.record_verdict_exemplar(db, o, decided)
        for reason, payload in res.hitl:
            o = payload["object"]
            _store_object(cfg, db, s, region, o, "needs_review", "hitl", by_id, deps, reason, payload)
    report["steps"]["settle"] = agg

    db.upsert("sessions", {"id": s.id, "path": str(session_dir), "country": s.country, "status": "done"})
    p = output.export(db, s.id, cfg.output_dir)                                 # 14
    report["output"] = str(p)
    report["status"] = "done"
    if rec:
        scene = json.loads(p.read_text())
        rec.json("scene.json", scene)
        rec.image("objects_topdown.png", objects_plot(fp, scene["objects"]))
        rec.image("objects_isometric.png", isometric_plot(fp, scene["objects"]))
        for o in scene["objects"]:
            for c in o["crops"] or []:
                rec.bytes(f"objects/{Path(c).name}", (cfg.output_dir / c).read_bytes())
        rec.json("verdicts.json", [dict(r) for r in db.q("SELECT * FROM verdicts")])
        rec.json("hitl_queue.json", hitl.open_items(db))
        rec.json("report.json", report)
    return report
