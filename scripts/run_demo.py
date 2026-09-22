"""Run the real pipeline (real Fable/Astra/Jev/web-search pricing) on an imported session and keep all artifacts.

    PYTHONPATH=src python scripts/run_demo.py data/sessions/stray_c7d28f72c6 [--segmenter opencv|sam2] [--every 8] [--regions N]

--regions caps how many of the *found* regions get sent to the models (cost control); the rest are still
geometrically formed and their box-preview images are still saved, but they are marked "skipped_regions" in
the report and never reach the room's scene.json. Omit --regions (or pass 0) to process all of them.
"""
import argparse
import json
import sys
from pathlib import Path

from libpipe.artifacts import Recorder
from libpipe.config import Config
from libpipe.db import DB
from libpipe.identity import ISBNdb
from libpipe.llm import Astra, Fable, JevJudge, transcribe
from libpipe.pieces import OpenCVSegmenter, ReplicateSAM2, SpineDetector
from libpipe.pipeline import Deps, process_session
from libpipe.pricing import Pricer, WebSearchPricer

ap = argparse.ArgumentParser()
ap.add_argument("session_dir")
ap.add_argument("--segmenter", default="opencv", choices=["opencv", "sam2"])
ap.add_argument("--spine", action="store_true",
                help="also run the trained YOLOv8-OBB spine detector alongside --segmenter (see results/README.md "
                     "'Spine detector fine-tune'); on the two non-shelf test scans this should find near-zero "
                     "spines, which is itself the validation until there's a real bookshelf to run on")
ap.add_argument("--spine-weights", default="runs/obb/runs_spine/nano48/weights/best.pt")
ap.add_argument("--spine-conf", type=float, default=0.6)
ap.add_argument("--every", type=int, default=8, help="segment 1 of every N frames (lower = more room coverage, more SAM2 cost)")
ap.add_argument("--regions", type=int, default=0, help="0 = process every region found (default)")
ap.add_argument("--out", default=None)
ap.add_argument("--whisper", action="store_true", help="transcribe the session audio with OpenAI Whisper")
ap.add_argument("--resume", action="store_true",
                help="skip regions that already have objects stored in the DB for this session (e.g. after an "
                     "interrupted run) instead of re-paying for their Fable/Astra/Jev calls")
ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                help="clockwise degrees to correct crops shown to Fable/Astra and saved for review - some "
                     "capture rigs (confirmed: this project's own RoomCapture app) save frames sideways "
                     "relative to gravity; does not affect 3D box positions/sizes, which are unaffected "
                     "(see pieces.rotate_cw)")
a = ap.parse_args()

cfg = Config().ensure()
cfg.crop_rotate_deg = a.rotate
db = DB(cfg.db_path)
name = Path(a.session_dir).name
rec = Recorder(a.out or f"results/{name}")
base_segmenter = OpenCVSegmenter() if a.segmenter == "opencv" else ReplicateSAM2()
segmenter = [base_segmenter, SpineDetector(a.spine_weights, a.spine_conf)] if a.spine else base_segmenter
deps = Deps(Fable(), Astra(), JevJudge(), segmenter,
            transcribe if a.whisper else (lambda p: []), ISBNdb(), Pricer(web=WebSearchPricer()))
report = process_session(cfg, db, a.session_dir, deps, seed=0, rec=rec, every=a.every, max_regions=a.regions or None,
                         resume=a.resume)
print(json.dumps(report, indent=2, default=str))
