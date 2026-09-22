# libpipe

Server pipeline for RoomCapture sessions (steps 2-14 of the diagram). Cloud-API based: no GPU needed.

**Every accuracy number in this codebase's docstrings is the source diagram's *target*, not something this
code has achieved.** They're marked `SOURCE DIAGRAM TARGET, not measured by this code` at the point they're
quoted. Where a real (non-synthetic) test run exists, that file also says what was actually observed, and
it has so far always been worse than the target. See `results/README.md` for the real runs and their
known bugs (some fixed, some open).

    pip install -e .[dev]          # numpy scipy pillow httpx fastapi uvicorn anthropic openai zxing-cpp pyliblzfse
    cp .env.example .env           # fill keys
    libpipe serve                  # step 2: PUT /upload  (point Uploader.uploadURLString at it)
    libpipe process data/sessions/<session_folder>
    libpipe hitl                   # open review items
    libpipe gold                   # weekly gold-set check (pauses learning on drift)

No real RoomCapture session exists yet, so `scripts/run_demo.py` runs the real pipeline (real Fable/Astra/
Jev, real web-search pricing) against LiDAR scans imported from other apps, for testing the geometry and
judged-pair logic end to end. Importers: `stray.py` (Stray Scanner) and `r3d.py` (Record3D, needs
`pyliblzfse`) — both verify the imported camera poses empirically (adjacent-frame depth clouds must line up)
because each app uses a different pose convention and getting it wrong silently corrupts every 3D box.

    PYTHONPATH=src python scripts/run_demo.py data/sessions/<imported_session> --segmenter sam2 --every 8

| Step | Module | Status |
|---|---|---|
| 2 upload | `server.py` | written, **untested** (needs fastapi) |
| 3 prep | `prep.py`, `llm.transcribe` | align run on 2 real scans; Whisper run once on real (near-silent) audio - hallucinated |
| 4 stitch | `stitch.py` | written, **untested on real data** (needs two overlapping sessions; unit-tested on synthetic clouds) |
| 5 floor plan | `floorplan.py` | run on 2 real scans; against a hand-measured room only 1 of 3 dimensions was within the 10cm target (see `results/README.md`). **doors/windows not available** (app saves no ARKit mesh classification) |
| 6 find pieces | `pieces.py` | geometry/merge/spine-split unit-tested on synthetic data; run on 2 real scans with real SAM2 and the free OpenCV fallback - **never run on an actual bookshelf**, so spine separation is completely unmeasured |
| 7 regions | `regions.py` | run on 2 real scans; region *content* was previously wrong (see "Bugs found and fixed") |
| 8 region call | `regions.py`, `llm.py` | run on 2 real scans with real Fable + Astra |
| 9 settle | `settle.py` | unit-tested (16 cases incl. shift-fix, order-flip = tie, reshare, HITL routing) + run on 2 real scans |
| 10 identity | `identity.py` | unit-tested against mocked ISBNdb; **never run against a real book** |
| 11 price | `pricing.py` | unit-tested against mocked Keepa/SerpApi/FX; run on real non-book objects via web search (see `results/README.md`); book path (Keepa/Shopping) never run for real |
| 12-13 HITL, learning, gold | `hitl.py`, `hitl_ui.py`, `learning.py` | unit-tested; reviewable UI live at `GET /hitl` (crops, one-tap candidate pick, manual correction form); resolving an identity edit now re-runs pricing automatically unless the reviewer typed a price by hand |
| 14 output | `output.py` | run end to end on 2 real scans |

## Bugs found and fixed (real runs, not unit tests)
- **Stray Scanner poses were the wrong convention** (OpenCV, not ARKit): every 3D box was built from
  misaligned depth+pose. Found by checking that adjacent-frame depth clouds coincide (they didn't: 15.9cm
  median residual); fixed in `stray.py`, now 0.65cm. `r3d.py` checked the same thing for Record3D scans
  (poses there were already correct) and additionally checked the recovered room size against a hand-measured
  room to pick the right axis mapping.
- **The region overview/box-preview image picked the wrong photo.** It scored a still by how many piece
  *centers* fell inside frame, with no check on size or occlusion, so a still where every piece was a sliver
  at the edge scored the same as one framing them well. This is almost certainly why region content looked
  "outside the region" in early runs: the labelled overview and the box-preview images (and therefore the
  visual evidence Fable/Astra actually saw) didn't reliably show the pieces they were meant to. Fixed by
  `regions.best_group_still`, which requires most of each piece's projected box to be visible, not just its
  centre; both the overview image and the pieces-preview image now use it.
- **Adjacency between pieces was under-signalled.** The only proximity hint in the prompt was `overlaps`,
  which needs true 3D-box intersection and almost never fires between distinct nearby objects (most real
  clutter sits next to, not inside, its neighbour). Added a `near (not overlapping)` hint (within 20cm) so
  Fable/Astra can reason about adjacency; the system prompt now explains the difference and explicitly says
  not to report gaps or objects outside the given crops (true "unclaimed space" detection - a shelf-plane
  estimate - is still not implemented at all; see below).
- **`lift_mask` silently dropped most real objects, not just noise.** On a real cluttered-room frame, SAM2
  itself found 14 candidate masks and our own area filter correctly kept 10 of them - but only 4 of those 10
  survived depth-lifting into a 3D piece, because `lift_mask` required ARKit confidence `==2` ("high" only,
  discarding "medium") and >=30 confident points. This wasn't a segmentation problem at all: it silently threw
  away correctly-segmented masks for small, distant, or dark objects (a teddy bear, a spray can, several small
  bottles) purely on a depth-confidence technicality. Relaxing to confidence`>=1` and lowering the point floor
  to 15 recovered 9 of the 10 masks as real pieces, with no new false positives (all still passed the
  size-plausibility check). Found by checking whether the pipeline was finding all the *real* objects it was
  shown, not just whether the ones it did report were correct - see `results/README.md`. This affects every
  frame, so it's likely also part of why spine separation on a real bookshelf (book spines are thin, exactly
  the kind of small-footprint geometry this bug penalised) is unmeasured but suspect.
- **`find_pieces` silently discarded furniture-scale objects.** The size-plausibility filters (`0.25` of
  frame area, `0.6m` max 3D extent) were sized for books and never revisited when scope expanded to
  library-insurance inventory (furniture, laptops, chargers - anything seen). Found by a user noticing SAM2
  detecting "many things" in a frame but furniture never showing up as a piece: SAM2 was finding it fine,
  `find_pieces` was throwing the candidate away afterward as "too big = wall/shelf." Now `piece_max_frame_frac`
  (0.6) / `piece_max_dim_m` (3.0m) in `Config`, wide enough for a wardrobe or sofa; a real wall/floor/ceiling
  mask (which spans meters in every direction) still gets rejected, and step 8's `is_object` judgment remains
  the actual backstop either way. **Not yet re-run on real data** - only unit-tested with synthetic masks.
- **`--regions` silently dropped whole rooms.** `run_demo.py` defaulted to processing only the first 3-4
  regions found, with no indication more existed; on a 222-frame scan sampled every 16th frame, that meant
  most of the room's segmented pieces never reached a model. Default is now 0 (process every region found);
  the report also lists `regions_found` vs `regions_processed` and names any skipped regions.

## Known gaps / deviations from the diagram
- **Jev is text-only and returns no prose** (choice + probabilities + confidence). Step 9's "verdict + reasoning" is
  therefore the probability distribution, and Jev judges from the two models' text answers + text evidence
  (voice, notes, barcode, sizes), not from images. Early access / waitlist.
- Step 5 doors/windows: not implemented (needs ARKit mesh classification the app doesn't save).
- Step 6 "unclaimed space" flagging: not implemented at all (needs a shelf-plane estimate). The `near` hint
  added above helps models reason about clutter but is not the same thing.
- Step 9 always runs the re-share round on a disagreement (literal reading of the diagram); this doubles cost per
  dispute. Skipping it when Jev's first verdict is decisive is a one-line change in `settle_region`.
- Keepa is optional/skipped by default (paid plan) - book pricing currently falls through to Google Shopping
  search only, which returns retail prices, not necessarily the MRP/RRP the `price_concept` field claims.
- Google Lens needs a public image URL, which this setup doesn't have; non-book pricing uses Astra web-search
  on the crop directly instead (`WebSearchPricer`), which is close in spirit but not the same method the
  diagram specifies.
- Currency map covers ~20 countries; unknown countries default to USD.
- Spine detection now has a trained detector (`pieces.SpineDetector`, YOLOv8n-OBB, see "Spine detector
  fine-tuning" below) available alongside the original `split_by_spine_edges` heuristic - pass a list of
  segmenters (e.g. `[ReplicateSAM2(), SpineDetector()]`, or `--spine` on `run_demo.py`) to use both. Still
  **never run on a real bookshelf or any frame from this project's own capture pipeline** - only validated
  on Roboflow's held-out split (see `results/README.md`) and a false-positive check on two book-free scans.
- SAM2 via Replicate (`meta/sam-2`) now confirmed working end to end; the free `OpenCVSegmenter` fallback is
  much cruder (over-segments walls/texture) and exists only for testing without Replicate credit.

## Spine detector fine-tuning
Not started. Needs, from whoever owns the 132-image dataset: the export format (YOLO-OBB txt, COCO, VOC-XML,
or a Roboflow export zip), whether boxes are axis-aligned or oriented, and the class list. Also needs a
decision on where training runs (this machine's GPU, if any, vs. a cloud GPU/Ultralytics HUB/Roboflow's
hosted training) - none of that is set up yet.
