# Test results (kept for showcase)

**Read the target-vs-measured caveat in the main README first** — every accuracy number quoted in the code
docstrings is the source diagram's target, not something measured here. This file records what was actually
observed.

All runs use the real Fable (`claude-fable-5-1`), Astra (`gpt-6-astra`), Jev (`jev-latest`), Astra web-search
pricing, and (from run 4 onward) real SAM2 on Replicate.

| Folder | What it is |
|---|---|
| `stray_c7d28f72c6*/` | Stray Scanner home-interior scan, no books, no audio |
| `r3d_2026-09-08--00-36-25/` | **Latest / best run** — Record3D scan of a bedroom, real SAM2, real audio |
| `r3d_2026-09-08--00-36-25_run1_region_box_bug/` | Same scan, before the region/box-preview fix below |

## Run 5: `r3d_2026-09-08--00-36-25/` — after fixing the region/box-preview bug
Real Fable + Astra + Jev + real SAM2, `--every 12` (18 of 222 frames segmented), all 28 regions *found*,
first 6 processed (cost cap; the other 22 are listed by id in `report.json.steps.pieces.skipped_regions`,
not silently dropped).

**Result: 23 objects, 11 correctly priced, 12 correctly discarded as `not_an_object`, 2 to HITL, 1 unpriced.**
`pieces/r000_boxes.jpg` shows the 3D boxes projected back onto a real photo — they now land on the actual
mirror cabinet, spray can, cardboard box, jars, and backpack, not floating in blank wall space.

### Bugs a user found by eye and I confirmed + fixed (see main README "Bugs found and fixed" for detail)
1. **Region overview/box-preview picked the wrong photo** — scored by piece *centres* in frame, not by
   visible box size, so labelled images and the crops the models actually saw could show almost nothing of
   the real object. This is very likely why the previous run's descriptions looked like "objects outside the
   region." Fixed in `regions.best_group_still`; both preview paths now use it.
2. **`overlaps` almost never fired** between genuinely adjacent (not literally intersecting) objects, so
   Fable/Astra had no adjacency signal for real clutter. Added a `near (not overlapping)` hint + explicit
   system-prompt guidance not to invent gaps/objects outside the given crops.
3. **`--regions` cap silently dropped most of the room.** Previous runs capped at 3-4 regions with no
   indication more existed. Default is now "process everything found"; a cap is now reported, not hidden.
4. **`ReplicateSAM2` had no rate-limit backoff** and crashed on the first `429`. Added exponential backoff
   with `Retry-After`, matching the pattern already used for Jev.

### Still not measured / still wrong
- Object/price accuracy: no ground truth, so still just "looks plausible" (e.g. 0.89 INR for a tissue).
- `p0022`/`p0025` disagreements: still genuinely hard cases (Jev confidence 0.26-0.72 on some finals) - HITL
  routing is doing its job, not failing.
- Still no books in any test scan - step 6 spine separation, step 10 identity, step 11 book pricing remain
  completely unexercised on real data.

## Run 4: `r3d_2026-09-08--00-36-25_run1_region_box_bug/` (kept as the "before")
Same scan, `--every 16`/`--regions 4`: only 4 of many possible regions processed, boxes projected onto
largely empty wall/ceiling. This is the run the region/box-preview bug was found in.

## Earlier runs (stray scanner, no books)
- `stray_c7d28f72c6_run1_wrong_pose_convention/`: Stray Scanner poses are OpenCV-convention; assuming ARKit
  gave a 15.9cm adjacent-frame depth residual. Fixed in `stray.py`, now 0.65cm.
- `stray_c7d28f72c6_run2_before_is_object_gate/`: before the `is_object` gate - wall patches were priced.
- `stray_c7d28f72c6/`: with the gate - correctly discards non-objects, but the free OpenCV segmenter
  over-segments walls into 1200+ "pieces"; superseded by real SAM2 in run 4/5 above.

## Floor plan vs a hand-measured room (from run 4, unaffected by the region/box fix)
| | Measured | Pipeline (percentile) | Pipeline (wall-plane) |
|---|---|---|---|
| Breadth | 3.54 m | 3.88 m (+34cm) | 0.76 m (locked onto furniture, wrong) |
| Length | 3.77 m | 3.86 m (+9cm) | 3.50 m (-4cm) |
| Height | 2.60 m | 2.78 m (+18cm) | 2.70 m (+10cm) |

Only 1-2 of 6 estimates land inside the diagram's 10cm target. Full detail in `r3d_2026-09-08--00-36-25_run1_region_box_bug/floorplan/vs_ground_truth.json`.

---
## Spine detector fine-tune: yolov8n-obb, 48 epochs
Trained locally (GTX 1650, 4GB) on the 2,003-image Roboflow "Book Spines" export (single class `book-spine`,
real oriented boxes). Crashed once at epoch 23/48 from system RAM exhaustion (8.4GB total, another process
running concurrently) - resumed cleanly from checkpoint with `workers=1`, completed all 48 epochs.

**Final (best.pt, 148-image held-out validation split):**

| Metric | Value |
|---|---|
| Precision | 96.2% |
| Recall | 93.9% |
| mAP50 (IoU>=0.50) | 97.8% |
| mAP50-95 (COCO-style, IoU 0.50-0.95) | 76.3% |
| Inference | 13.0ms/image (~77 FPS) on this GTX 1650 |

**Read this against the paper's numbers carefully - not directly comparable:**
- The paper reports mAP at a single strict IoU=0.75 threshold (90.22% for their DCN+PAFPN+ResNet50
  architecture; 75.19% for plain YOLOv8n-OBB in their own comparison table).
- Our mAP50 (97.8%) uses a much looser IoU>=0.50 threshold - not the same metric, don't compare it to their
  90.22% directly.
- Our mAP50-95 (76.3%) is the fairer comparison (it's an average that includes IoU=0.75 within its range) -
  by that read, this nano model lands close to their own plain-YOLOv8n-OBB baseline (75.19%), well below
  their proposed architecture, which is exactly what you'd expect from a 3M-parameter model vs their much
  larger custom design.

**What this does NOT tell us:** this is validated on Roboflow's own held-out split - same distribution and
photo style as training (library shelves, similar phones/angles). It has not been run on a single real photo
from our own capture pipeline (this project has never processed a real bookshelf at all). Real-world
generalization to our actual LiDAR RGB frames is completely unverified until that happens.

**Done since:** integrated into `pieces.py` as `SpineDetector`, loading this exact weights path, usable
alongside the untrained `split_by_spine_edges` heuristic (pass both to `--spine`). Still unrun on any real
bookshelf frame. The box-prompted-SAM2 idea that was meant to follow is not started.

Weights: `runs/obb/runs_spine/nano48/weights/best.pt` (not committed - 6.4MB, regenerate via
`scripts/train_spine_colab.ipynb`'s local-equivalent settings if needed elsewhere).
