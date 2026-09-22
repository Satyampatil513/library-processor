"""libpipe serve | process <session_dir> | hitl | gold"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config


def real_deps():
    from .identity import ISBNdb
    from .llm import Astra, Fable, JevJudge, transcribe
    from .pieces import ReplicateSAM2, SpineDetector
    from .pipeline import Deps
    from .pricing import Pricer, WebSearchPricer

    import os
    from pathlib import Path
    reg = os.environ.get("ASSET_REGISTER")
    pricer = Pricer(asset_register=Path(reg) if reg else None, web=WebSearchPricer())
    # step 6 per the diagram: "SAM2 + spine-boundary detector", not SAM2 alone - this was missing here even
    # after the trained detector was wired into run_demo.py, so a real `libpipe process` on a real app upload
    # would have silently skipped it and fallen back to split_by_spine_edges for every mask, book or not.
    segmenter = [ReplicateSAM2(), SpineDetector()]
    return Deps(Fable(), Astra(), JevJudge(), segmenter, transcribe, ISBNdb(), pricer)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="libpipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    p = sub.add_parser("process")
    p.add_argument("session_dir")
    sub.add_parser("hitl")
    sub.add_parser("gold")
    a = ap.parse_args(argv)
    cfg = Config().ensure()

    from .db import DB
    db = DB(cfg.db_path)
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("libpipe.server:app", host="0.0.0.0", port=8000)
    elif a.cmd == "process":
        from .pipeline import process_session
        print(json.dumps(process_session(cfg, db, a.session_dir, real_deps()), indent=2, default=str))
    elif a.cmd == "hitl":
        from .hitl import open_items
        print(json.dumps(open_items(db), indent=2, default=str))
    elif a.cmd == "gold":
        from .learning import run_gold
        from .llm import JevJudge
        print(json.dumps(run_gold(db, JevJudge(), cfg.drift_pause_agreement)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
