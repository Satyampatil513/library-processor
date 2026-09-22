# libpipe

Server pipeline for RoomCapture sessions (steps 2-14 of the diagram). Cloud-API based (Fable/Claude,
Astra/GPT, Jev/TypeSafe, SAM2 via Replicate) — no GPU needed, except the trained spine detector, which
runs locally via Ultralytics/PyTorch (CPU is fine for inference; training needed a GPU, see below).

**Every accuracy number in this codebase's docstrings is the source diagram's *target*, not something this
code has achieved.** They're marked `SOURCE DIAGRAM TARGET, not measured by this code` at the point they're
quoted. Where a real (non-synthetic) test run exists, that file also says what was actually observed. See
`results/README.md` for the real runs and their known bugs (some fixed, some open).

## Install

    pip install -e .[dev]          # add .[dev,r3d] instead if you need the Record3D importer too
    cp .env.example .env           # fill in keys - see Configuration below

## Configuration (`.env`)

| Key | Required for | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Fable (step 8/9 judge) | |
| `OPENAI_API_KEY` | Astra (step 8/9 judge) | separate billing/credits from Anthropic |
| `TYPESAFE_API_KEY` | Jev (step 9 arbiter) | |
| `REPLICATE_API_TOKEN` | SAM2 (step 6, `--segmenter sam2`) | |
| `ISBNDB_API_KEY` | book identity (step 10) | |
| `SERPAPI_API_KEY` | Google Shopping (books) + Google Lens (non-books) pricing | |
| `KEEPA_API_KEY` | book pricing, preferred over Shopping | optional, paid plan |
| `IMAGE_BASE_URL` | routes non-book pricing to Google Lens instead of Astra web-search | see "Real upload setup" below - **set this**, it avoids burning OpenAI credits on pricing |
| `ASSET_REGISTER` | offline CSV fallback (`description,price,currency`) if Lens/web-search find nothing | optional |
| `LIBPIPE_DATA` | where sessions/db/output live | defaults to `./data` |
| `UPLOAD_TOKEN` | shared secret the app's Settings screen must match | optional |

## Real workflow: phone → server → reviewed inventory

1. **Start the server**: `libpipe serve` (or `uvicorn libpipe.server:app --host 0.0.0.0 --port 8000`)
2. **Expose it publicly** so the phone can reach it: `cloudflared tunnel --url http://localhost:8000`
   (or ngrok, or a real domain). Free tunnel URLs change on every restart.
3. **Point the app at it**: in the RoomCapture app, gear icon → Settings → paste `<tunnel-url>/upload`.
   Tap Test Connection first.
4. **Set `IMAGE_BASE_URL`** in `.env` to `<tunnel-url>/img` (the server already serves saved crops there via
   `StaticFiles`) — this routes non-book pricing through Google Lens instead of Astra's web-search
   workaround, which otherwise doubles your OpenAI spend (region judging *and* pricing on the same
   account). Restart the server after changing `.env`.
5. **Record and upload** a session from the app. It lands in `data/sessions/<session_id>/`, status
   `uploaded` in the DB.
6. **Process it**: `libpipe process data/sessions/<session_id>`
7. **Review**: `libpipe hitl` (JSON dump) or the actual reviewable page at `GET /hitl` on the same server/
   tunnel — crops, one-tap candidate pick, manual correction, and a "remove object entirely" option for
   duplicates (see "Known gaps").

`scripts/run_demo.py` is the equivalent for testing/demoing against an already-imported or already-uploaded
session folder, with more flags than the bare `libpipe process` currently exposes:

    PYTHONPATH=src python scripts/run_demo.py data/sessions/<session_id> \
        --segmenter sam2 --spine --every 10 --rotate 90 --out results/<name> [--resume]

- `--spine` — also run the trained spine detector (`pieces.SpineDetector`) alongside SAM2/OpenCV
- `--rotate 90|180|270` — clockwise correction applied to crops shown to Fable/Astra and saved for review;
  needed for this project's own RoomCapture app, which saves frames/stills in raw sensor orientation
  regardless of how the phone was held (confirmed on a real upload - see "Bugs found and fixed"). 3D
  geometry is unaffected either way; only what a human/vision-model actually looks at needs this.
- `--resume` — skip regions (and the SAM2/spine-detector pass itself) already completed in a prior,
  interrupted attempt at the same `--out` path, instead of re-paying for them. **Not yet on the production
  `libpipe process` command** - only `run_demo.py`, see "Known gaps".
- `--regions N` — cap how many *found* regions get sent to the models (cost control); 0 (default) = all.

Importers for LiDAR scans captured by *other* apps (useful before you have a real RoomCapture upload, or
for comparison): `stray.py` (Stray Scanner) and `r3d.py` (Record3D, needs the `r3d` extra) — both verify
imported camera poses empirically (adjacent-frame depth clouds must line up), because each app uses a
different pose convention and getting it wrong silently corrupts every 3D box.

| Step | Module | Status |
|---|---|---|
| 2 upload | `server.py` | working - real uploads received from the actual iOS app |
| 3 prep | `prep.py`, `llm.transcribe` | run on real sessions incl. real audio transcription |
| 4 stitch | `stitch.py` | **still untested on real data** (needs two overlapping real room sessions; unit-tested on synthetic clouds) |
| 5 floor plan | `floorplan.py` | run on real scans; a duplicate-wall bug was found and fixed (see below). Dimension accuracy still unverified against a real hand-measured room since that fix. **doors/windows not available** (app saves no ARKit mesh classification) |
| 6 find pieces | `pieces.py` | run on real scans with real SAM2 + the trained spine detector; furniture-scale objects (a bed, a wall-mounted TV) now correctly detected after a real bug fix (see below) |
| 7 regions | `regions.py` | run on real scans |
| 8 region call | `regions.py`, `llm.py` | run on real scans with real Fable + Astra, incl. real book spines |
| 9 settle | `settle.py` | unit-tested (16+ cases incl. shift-fix, order-flip = tie, reshare, HITL routing) + run on real scans |
| 10 identity | `identity.py` | **run against real books for the first time**: 2 of 4 real spine detections resolved a real ISBN via ISBNdb |
| 11 price | `pricing.py` | run on real objects, books and non-books; one real book successfully priced via Google Shopping |
| 12-13 HITL, learning, gold | `hitl.py`, `hitl_ui.py`, `learning.py` | reviewable UI live at `GET /hitl` - crops, one-tap candidate pick, manual correction, "not a real object" override, and "remove object entirely" (for duplicates); identity edits auto-reprice; gold-set drift check (`libpipe gold`) still never actually run |
| 14 output | `output.py` | run end to end on real scans |

## Bugs found and fixed (real runs, not unit tests)
- **Stray Scanner poses were the wrong convention** (OpenCV, not ARKit): every 3D box was built from
  misaligned depth+pose. Found by checking that adjacent-frame depth clouds coincide (they didn't: 15.9cm
  median residual); fixed in `stray.py`, now 0.65cm. `r3d.py` checked the same thing for Record3D scans
  (poses there were already correct) and additionally checked the recovered room size against a hand-measured
  room to pick the right axis mapping.
- **The region overview/box-preview image picked the wrong photo.** It scored a still by how many piece
  *centers* fell inside frame, with no check on size or occlusion. Fixed by `regions.best_group_still`,
  which requires most of each piece's projected box to be visible, not just its centre.
- **Adjacency between pieces was under-signalled.** Added a `near (not overlapping)` hint (within 20cm) so
  Fable/Astra can reason about real-world clutter that sits next to, not inside, its neighbour.
- **`lift_mask` silently dropped most real objects, not just noise.** Required ARKit confidence `==2`
  ("high" only) and >=30 points, which threw away correctly-segmented masks for small/distant/dark objects
  on a depth-confidence technicality alone. Relaxing to confidence `>=1` and lowering the point floor to 15
  recovered 9 of 10 previously-dropped real pieces in one test frame, no new false positives.
- **`find_pieces` silently discarded furniture-scale objects.** The size-plausibility filters (0.25 of frame
  area, 0.6m max 3D extent) were sized for books and never revisited when scope expanded to library-insurance
  inventory (furniture, laptops, chargers). Now `piece_max_frame_frac` (0.6) / `piece_max_dim_m` (3.0m) in
  `Config`. **Confirmed fixed on real data**: a real bed and wall-mounted TV, previously invisible to the
  pipeline, now show up as detected, priced objects.
- **`--regions` silently dropped whole rooms.** Default is now 0 (process every region found); the report
  lists `regions_found` vs `regions_processed` and names any skipped regions.
- **Frames/stills were saved sideways.** RoomCapture (this project's own app) keeps captured images in raw
  sensor orientation regardless of how the phone was physically held. 3D reconstruction is unaffected
  (ARKit world space is gravity-aligned Y-up regardless of buffer orientation), but every crop shown to a
  vision model or a human reviewer was rotated 90°. Fixed with `pieces.rotate_cw`, applied only to final
  crops (never to a full frame or a mask, to avoid touching the depth-aligned math in `find_pieces`) —
  `--rotate` on `run_demo.py`, `Config.crop_rotate_deg` otherwise.
- **The same physical wall was sometimes detected twice.** RANSAC line-fitting on noisy real depth data
  occasionally split one wall into two near-parallel, near-coincident segments (confirmed: 15-17cm apart,
  more than the original synthetic-noise tolerance assumed). Fixed by `floorplan.merge_duplicate_walls`.
- **The production `libpipe process` path never got the trained spine detector.** It was wired into
  `run_demo.py`'s `--spine` flag but `cli.py`'s `real_deps()` still built `Deps` with SAM2 alone - a real
  upload processed via the actual production command would have silently skipped it. Fixed.
- **`--resume` still re-paid for SAM2/spine-detector segmentation on every retry.** It already skipped
  regions with stored objects, but recomputed step 6/7 geometry from scratch every time - itself a paid
  Replicate call, for identical results, on the exact same frames. Now cached to the `--out` directory's
  `pieces.json`/`regions.json` and reloaded on `--resume` instead of recomputed.
- **HITL had no way to mark something "not a real object," or to remove a duplicate outright.** A reviewer
  could edit book/price fields or dismiss an item as-is, but not correct a wall/floor fragment the models
  missed, nor delete a confirmed duplicate detection (found for real: the same physical book, split across
  two different regions by size-based chunking, got identified and priced twice). Both options now on the
  HITL review page.

## Known gaps / deviations from the diagram
- **Spine detector recall was never tuned against real data** - only against false positives on two
  book-free rooms. Spot-checked against a real bookshelf photo for the first time: of 3 clearly visible
  spines, it correctly boxed 1 (the other 2 *were* detected, just below the 0.6 confidence cutoff - likely
  partial occlusion + motion blur, not a blind spot), and had one false positive on a power strip (its
  repeated grid-of-holes pattern resembles the vertical-line texture learned for spines). Needs real-shelf
  recall data to retune the threshold properly.
- **Cross-region duplicate detection is manual only.** Two pieces of the same physical object can land in
  different regions (make_regions' size-based chunking doesn't guarantee non-overlapping-but-adjacent
  pieces stay together) and get identified/priced independently. A reviewer has to spot and remove the
  duplicate by hand via HITL; there's no automatic cross-region judged-pair check.
- **`--resume`/`--rotate`/`--spine`/`--every` are `run_demo.py`-only.** The production `libpipe process`
  command doesn't expose any of them yet - always full SAM2+spine, no rotation correction, default
  `every=5`, no resume support.
- **Jev is text-only and returns no prose** (choice + probabilities + confidence). Jev judges from the two
  models' text answers + text evidence (voice, notes, barcode, sizes), not from images.
- Step 5 doors/windows: not implemented (needs ARKit mesh classification the app doesn't save).
- Step 6 "unclaimed space" flagging: not implemented at all (needs a shelf-plane estimate).
- Step 9 always runs the re-share round on a disagreement (literal reading of the diagram); doubles cost
  per dispute.
- Keepa is optional/paid - book pricing falls through to Google Shopping if not configured, which returns
  retail prices, not necessarily the MRP/RRP the `price_concept` field claims.
- Non-book pricing accuracy has no ground truth to check against - prices look plausible, nothing confirms
  they're actually accurate.
- Currency map covers ~20 countries; unknown countries default to USD.
- The weekly gold-set drift check (`libpipe gold`, `learning.run_gold`) has never actually been run.

## Spine detector fine-tune

Trained: `yolov8n-obb`, 48 epochs, on Roboflow's "Book Spines" dataset (2,003 images, real oriented boxes).
Training artifacts and metrics are in `runs/obb/runs_spine/nano48/` (weights, PR curves, confusion matrix).

**On the training distribution's held-out split**: 96.2% precision, 93.9% recall, mAP50 97.8%, mAP50-95
76.3% - see `results/README.md` for the full metrics and how they compare to the source paper.

**On a real photo, outside the training distribution, for the first time**: see "Known gaps" above - 1 of
3 real spines correctly boxed at the default threshold, the other 2 detected but under-confident, plus one
false positive on unrelated clutter. Real-world generalization is now partially measured, not
"completely unverified" as before, but still needs real-shelf recall data before the confidence threshold
can be trusted.
